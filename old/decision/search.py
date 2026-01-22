from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple, List

import torch

from envs.env import FactoryLayoutEnv
from .agents import Agent
from .selectors import CandidateSelector
from .candidate_set import CandidateSet


class SearchStrategy(Protocol):
    """Search strategy that chooses an action index among candidates.

    Pipeline owns `agent` and `selector` and passes them in to avoid duplicate injection.
    """

    def select(
        self,
        *,
        env: FactoryLayoutEnv,
        obs: dict,
        agent: Agent,
        selector: CandidateSelector,
    ) -> int: ...


@dataclass(frozen=True)
class MCTSConfig:
    num_simulations: int = 50
    c_puct: float = 1.0
    rollout_depth: int = 5
    # Root Dirichlet noise (parity with `policies/mcts.py`).
    # If epsilon <= 0, noise is disabled.
    dirichlet_epsilon: float = 0.0
    dirichlet_concentration: float = 0.5


@dataclass(frozen=True)
class BeamConfig:
    beam_width: int = 8
    depth: int = 5
    # How many actions to consider per beam node based on priors ranking.
    expansion_topk: int = 16


class _Node:
    def __init__(
        self,
        *,
        snapshot: Dict[str, object],
        selector_state: object,
        candidates: CandidateSet,
        priors: torch.Tensor,  # float32 [N]
        parent: Optional["_Node"] = None,
        action: Optional[int] = None,
        reward: float = 0.0,
        terminal: bool = False,
    ):
        self.snapshot = snapshot
        self.selector_state = selector_state
        self.candidates = candidates
        self.priors = priors
        self.parent = parent
        self.action = action
        self.reward = float(reward)
        self.terminal = bool(terminal)

        self.visits = 0
        self.total_value = 0.0
        self.children: Dict[int, "_Node"] = {}

        valid = candidates.mask
        self.valid_actions = [i for i in range(int(valid.shape[0])) if bool(valid[i].item())]

    def best_action(self, c_puct: float) -> int:
        if not self.valid_actions:
            return -1

        fpu_val = (self.total_value / self.visits) if self.visits > 0 else 0.0
        best_score = float("-inf")
        best_act = -1
        for act in self.valid_actions:
            if act in self.children:
                child = self.children[act]
                q = child.total_value / max(1, child.visits)
                n = child.visits
            else:
                q = fpu_val
                n = 0
            p = float(self.priors[act].item()) if act < int(self.priors.shape[0]) else 0.0
            u = float(c_puct) * p * math.sqrt(self.visits + 1) / (1 + n)
            score = float(q) + float(u)
            if score > best_score:
                best_score = score
                best_act = act
        return best_act


class MCTSSearch:
    """MCTS over (CandidateSelector + Agent priors) using env snapshots."""

    def __init__(self, *, config: MCTSConfig):
        self.config = config

    def select(self, *, env: FactoryLayoutEnv, obs: dict) -> int:
        raise RuntimeError(
            "MCTSSearch.select(env, obs) is not supported. "
            "Use select(env=..., obs=..., agent=..., selector=...)."
        )

    def select(
        self,
        *,
        env: FactoryLayoutEnv,
        obs: dict,
        agent: Agent,
        selector: CandidateSelector,
    ) -> int:
        root_snapshot = env.get_snapshot()
        root_sel_state = selector.get_state()

        root_candidates = selector.build(env)
        priors = self._safe_priors(agent=agent, env=env, obs=obs, candidates=root_candidates)
        priors = self._apply_root_dirichlet(priors=priors, mask=root_candidates.mask)
        root = _Node(
            snapshot=env.get_snapshot(),
            selector_state=selector.get_state(),
            candidates=root_candidates,
            priors=priors,
            parent=None,
            action=None,
            reward=0.0,
            terminal=False,
        )

        for _ in range(int(self.config.num_simulations)):
            self._simulate(env=env, root=root, agent=agent, selector=selector)

        # Pick the most visited child.
        if not root.children:
            env.set_snapshot(root_snapshot)
            selector.set_state(root_sel_state)
            return 0

        best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
        env.set_snapshot(root_snapshot)
        selector.set_state(root_sel_state)
        return int(best_action)

    def _apply_root_dirichlet(self, *, priors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply root Dirichlet noise to priors (valid actions only).

        Logic matches `policies/mcts.py:_apply_root_dirichlet`:
        - eps <= 0 or c <= 0 -> no change
        - alpha = c / valid_count
        - mixed_valid = (1-eps)*priors_valid + eps*dirichlet(alpha)
        - renormalize valid part to sum to 1
        """
        eps = float(self.config.dirichlet_epsilon)
        c = float(self.config.dirichlet_concentration)
        if eps <= 0.0 or c <= 0.0:
            return priors

        if not isinstance(mask, torch.Tensor):
            return priors
        valid = mask.to(dtype=torch.bool, device=priors.device).view(-1)
        valid_count = int(valid.to(torch.int64).sum().item())
        if valid_count <= 0:
            return priors

        alpha = c / float(valid_count)
        if alpha <= 0.0:
            return priors

        # Dirichlet over valid actions only (torch implementation).
        alpha_vec = torch.full((valid_count,), float(alpha), dtype=torch.float32, device=priors.device)
        noise = torch.distributions.Dirichlet(alpha_vec).sample().to(dtype=torch.float32)

        mixed = priors.clone()
        mixed_valid = (1.0 - eps) * mixed[valid] + eps * noise
        total = float(mixed_valid.sum().item())
        if total > 0.0:
            mixed_valid = mixed_valid / total
        mixed[valid] = mixed_valid
        return mixed

    def _safe_priors(self, *, agent: Agent, env: FactoryLayoutEnv, obs: dict, candidates: CandidateSet) -> torch.Tensor:
        pri = agent.priors(env=env, obs=obs, candidates=candidates)
        if not isinstance(pri, torch.Tensor):
            raise TypeError("Agent.priors must return torch.Tensor")
        pri = pri.to(dtype=torch.float32, device=env.device).view(-1)
        if int(pri.shape[0]) != int(candidates.mask.shape[0]):
            # fallback: uniform over valid
            out = torch.zeros((int(candidates.mask.shape[0]),), dtype=torch.float32, device=env.device)
            valid = candidates.mask
            cnt = int(valid.to(torch.int64).sum().item())
            if cnt > 0:
                out[valid] = 1.0 / float(cnt)
            return out
        # enforce non-negative and zero invalid
        pri = torch.clamp(pri, min=0.0)
        pri = pri.masked_fill(~candidates.mask, 0.0)
        s = float(pri.sum().item())
        if s > 0:
            pri = pri / s
        else:
            # uniform fallback
            valid = candidates.mask
            cnt = int(valid.to(torch.int64).sum().item())
            if cnt > 0:
                pri = torch.zeros_like(pri)
                pri[valid] = 1.0 / float(cnt)
        return pri

    def _simulate(self, *, env: FactoryLayoutEnv, root: _Node, agent: Agent, selector: CandidateSelector) -> None:
        node = root
        env.set_snapshot(root.snapshot)
        selector.set_state(root.selector_state)

        path_nodes = [root]
        path_rewards = []

        # Selection/Expansion
        while not node.terminal:
            action = node.best_action(self.config.c_puct)
            if action == -1:
                break

            if action in node.children:
                node = node.children[action]
                env.set_snapshot(node.snapshot)
                selector.set_state(node.selector_state)
                path_nodes.append(node)
                path_rewards.append(node.reward)
            else:
                # Expand new child by taking action in this node.
                env.set_snapshot(node.snapshot)
                selector.set_state(node.selector_state)

                cand = node.candidates
                x = float(cand.xyrot[action, 0].item()) if action < int(cand.xyrot.shape[0]) else 0.0
                y = float(cand.xyrot[action, 1].item()) if action < int(cand.xyrot.shape[0]) else 0.0
                rot = int(round(float(cand.xyrot[action, 2].item()))) if action < int(cand.xyrot.shape[0]) else 0

                obs2, reward, terminated, truncated, _info = env.step_masked(
                    action=int(action),
                    x=float(x),
                    y=float(y),
                    rot=int(rot),
                    mask=cand.mask,
                    action_space_n=int(cand.mask.shape[0]),
                    extra_info=None,
                )
                terminal = bool(terminated or truncated)

                if terminal:
                    next_candidates = CandidateSet(
                        xyrot=torch.zeros((0, 3), dtype=torch.float32, device=env.device),
                        mask=torch.zeros((0,), dtype=torch.bool, device=env.device),
                        meta={"terminal": True},
                    )
                    priors = torch.zeros((0,), dtype=torch.float32, device=env.device)
                else:
                    next_candidates = selector.build(env)
                    priors = self._safe_priors(agent=agent, env=env, obs=obs2, candidates=next_candidates)

                child = _Node(
                    snapshot=env.get_snapshot(),
                    selector_state=selector.get_state(),
                    candidates=next_candidates,
                    priors=priors,
                    parent=node,
                    action=int(action),
                    reward=float(reward),
                    terminal=terminal or (not terminal and int(next_candidates.mask.to(torch.int64).sum().item()) == 0),
                )
                node.children[int(action)] = child
                node = child
                path_nodes.append(node)
                path_rewards.append(float(reward))
                break

        # Rollout from leaf
        rollout_return = 0.0
        if not node.terminal:
            rollout_return = self._rollout(env=env, agent=agent, selector=selector)

        # Backprop
        total = float(rollout_return)
        for reward, path_node in zip(reversed(path_rewards), reversed(path_nodes[1:])):
            total += float(reward)
            path_node.visits += 1
            path_node.total_value += float(total)

        root.visits += 1
        root.total_value += float(total)

    def _rollout(self, *, env: FactoryLayoutEnv, agent: Agent, selector: CandidateSelector) -> float:
        total = 0.0
        for _ in range(int(self.config.rollout_depth)):
            obs = env._build_obs()
            candidates = selector.build(env)
            if int(candidates.mask.to(torch.int64).sum().item()) == 0:
                # trigger engine's no_valid_actions path
                _, reward, terminated, truncated, _ = env.step_masked(
                    action=0,
                    x=0.0,
                    y=0.0,
                    rot=0,
                    mask=candidates.mask,
                    action_space_n=int(candidates.mask.shape[0]),
                )
                total += float(reward)
                break

            a = agent.select_action(env=env, obs=obs, candidates=candidates)
            x = float(candidates.xyrot[a, 0].item())
            y = float(candidates.xyrot[a, 1].item())
            rot = int(round(float(candidates.xyrot[a, 2].item())))
            _obs2, reward, terminated, truncated, _ = env.step_masked(
                action=int(a),
                x=float(x),
                y=float(y),
                rot=int(rot),
                mask=candidates.mask,
                action_space_n=int(candidates.mask.shape[0]),
            )
            total += float(reward)
            if terminated or truncated:
                break
        return float(total)


class BeamSearch:
    """Beam search over discrete candidates using env snapshots.

    Logic:
    - At each depth, expand each beam by top-k actions (by agent priors) among valid candidates.
    - Score beams by cumulative reward (sum of env.step_masked rewards).
    - Return the root action of the best-scoring beam.

    NOTE: Like MCTS, this must restore env+selector state to root before returning.
    """

    def __init__(self, *, config: BeamConfig):
        self.config = config

    def select(
        self,
        *,
        env: FactoryLayoutEnv,
        obs: dict,
        agent: Agent,
        selector: CandidateSelector,
    ) -> int:
        root_snapshot = env.get_snapshot()
        root_sel_state = selector.get_state()

        # Each beam item: (cum_reward, first_action, snapshot, selector_state)
        beams: List[Tuple[float, int, Dict[str, object], object]] = [(0.0, -1, root_snapshot, root_sel_state)]

        for d in range(int(self.config.depth)):
            new_beams: List[Tuple[float, int, Dict[str, object], object]] = []

            for cum_reward, first_action, snap, sel_state in beams:
                env.set_snapshot(snap)
                selector.set_state(sel_state)

                candidates = selector.build(env)
                valid_mask = candidates.mask
                valid_n = int(valid_mask.to(torch.int64).sum().item())
                if valid_n <= 0:
                    # dead-end
                    new_beams.append((cum_reward, first_action if first_action >= 0 else 0, env.get_snapshot(), selector.get_state()))
                    continue

                priors = agent.priors(env=env, obs=obs, candidates=candidates).to(dtype=torch.float32, device=env.device).view(-1)
                priors = priors.masked_fill(~valid_mask, float("-inf"))

                topk = min(int(self.config.expansion_topk), int(priors.numel()))
                if topk <= 0:
                    continue

                top_actions = torch.topk(priors, k=topk).indices.tolist()

                for a in top_actions:
                    a = int(a)
                    x = int(candidates.xyrot[a, 0].item())
                    y = int(candidates.xyrot[a, 1].item())
                    rot = int(candidates.xyrot[a, 2].item())

                    # step from the current node state
                    env.set_snapshot(snap)
                    selector.set_state(sel_state)
                    _obs2, reward, terminated, truncated, _info = env.step_masked(
                        action=a,
                        x=float(x),
                        y=float(y),
                        rot=int(rot),
                        mask=candidates.mask,
                        action_space_n=int(candidates.mask.shape[0]),
                    )

                    new_cum = float(cum_reward) + float(reward)
                    root_a = a if first_action < 0 else int(first_action)
                    new_beams.append((new_cum, root_a, env.get_snapshot(), selector.get_state()))

                    if terminated or truncated:
                        # terminal states can still stay in beam; no further expansion will help.
                        pass

            if not new_beams:
                break

            # Keep top beams
            new_beams.sort(key=lambda t: t[0], reverse=True)
            beams = new_beams[: int(self.config.beam_width)]

        best = beams[0][1] if beams else 0

        # Restore root
        env.set_snapshot(root_snapshot)
        selector.set_state(root_sel_state)
        return int(best) if int(best) >= 0 else 0


if __name__ == "__main__":
    import time

    from envs.json_loader import load_env
    from .agents import GreedyAgent
    from .selectors import TopKSelector

    ENV_JSON = "env_configs/basic_01.json"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    env = loaded.env
    env.log = False
    obs, _ = env.reset(options=loaded.reset_kwargs)

    selector = TopKSelector(k=50, scan_step=10.0, quant_step=10.0, random_seed=0)
    agent = GreedyAgent(prior_temperature=1.0)
    search = MCTSSearch(config=MCTSConfig(num_simulations=50, rollout_depth=5))

    t0 = time.perf_counter()
    a = search.select(env=env, obs=obs, agent=agent, selector=selector)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # Rebuild root candidates for decode (search restores selector/env state).
    cand = selector.build(env)
    x, y, rot = (0.0, 0.0, 0)
    if int(cand.mask.sum().item()) > 0:
        x = float(cand.xyrot[a, 0].item())
        y = float(cand.xyrot[a, 1].item())
        rot = int(round(float(cand.xyrot[a, 2].item())))

    print("[search demo]")
    print(" env=", ENV_JSON, "next_gid=", (env.remaining[0] if env.remaining else None))
    print(" selector: N=", int(cand.mask.shape[0]), "valid=", int(cand.mask.sum().item()))
    print(" output.action=", a, "xyrot=", (x, y, rot))
    print(f" elapsed_ms={dt_ms:.2f} (num_simulations=50, rollout_depth=5)")

