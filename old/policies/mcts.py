from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import numpy as np

from envs.env_old import Candidate, FactoryLayoutEnvOld


class BaseAgent:
    def get_candidates(self, env: FactoryLayoutEnvOld) -> Tuple[List[Candidate], np.ndarray]:
        raise NotImplementedError

    def select_action(
        self,
        env: FactoryLayoutEnvOld,
        candidates: List[Candidate],
        mask: np.ndarray,
    ) -> int:
        raise NotImplementedError

    def get_action_priors(
        self,
        env: FactoryLayoutEnvOld,
        candidates: List[Candidate],
        mask: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def get_selector_state(self) -> object:
        return None

    def set_selector_state(self, state: object) -> None:
        return None


@dataclass(frozen=True)
class MCTSConfig:
    num_simulations: int = 50
    c_puct: float = 1.0
    rollout_depth: int = 5
    dirichlet_epsilon: float = 0.0
    dirichlet_concentration: float = 0.5


class MCTSNode:
    def __init__(
        self,
        snapshot: Dict[str, object],
        candidates: List[Candidate],
        mask: np.ndarray,
        priors: Optional[np.ndarray] = None,
        prior: float = 0.0,
        parent: Optional["MCTSNode"] = None,
        action: Optional[int] = None,
        reward: float = 0.0,
        terminal: bool = False,
    ):
        self.snapshot = snapshot
        self.candidates = candidates
        self.mask = mask
        self.priors = priors if priors is not None else np.zeros((len(candidates),), dtype=np.float32)
        self.prior = float(prior)
        self.parent = parent
        self.action = action
        self.reward = reward
        self.terminal = terminal

        self.visits = 0
        self.total_value = 0.0
        self.children: Dict[int, "MCTSNode"] = {}
        self.valid_actions = [i for i, m in enumerate(mask) if int(m) == 1]

    def best_action(self, c_puct: float) -> int:
        """Select action using PUCT formula (includes FPU for unvisited)."""
        if not self.valid_actions:
            return -1

        # FPU: Use parent's average value for unvisited nodes
        fpu_val = self.total_value / self.visits if self.visits > 0 else 0.0

        best_score = float("-inf")
        best_act = -1

        for act in self.valid_actions:
            if act in self.children:
                child = self.children[act]
                q = child.total_value / child.visits
                n = child.visits
                p = child.prior
            else:
                q = fpu_val
                n = 0
                p = self.priors[act] if act < len(self.priors) else 0.0

            u = c_puct * p * math.sqrt(self.visits + 1) / (1 + n)
            score = q + u

            if score > best_score:
                best_score = score
                best_act = act

        return best_act


class MCTSAgent(BaseAgent):
    """MCTS wrapper that uses a base agent for rollouts."""

    def __init__(self, base_agent: BaseAgent, config: MCTSConfig):
        self.base_agent = base_agent
        self.config = config

    def get_candidates(self, env: FactoryLayoutEnvOld) -> Tuple[List[Candidate], np.ndarray]:
        return self.base_agent.get_candidates(env)

    def select_action(
        self,
        env: FactoryLayoutEnvOld,
        candidates: List[Candidate],
        mask: np.ndarray,
    ) -> int:
        """Perform MCTS simulations and return the best action."""
        root_snapshot = env.get_snapshot()
        selector_state = self.base_agent.get_selector_state()
        root_candidates = candidates
        root_mask = mask
        if not root_candidates:
            root_candidates, root_mask = self.base_agent.get_candidates(env)
        if len(root_candidates) == 0:
            env.set_snapshot(root_snapshot)
            self.base_agent.set_selector_state(selector_state)
            return 0

        priors = self.base_agent.get_action_priors(env, root_candidates, root_mask)
        if len(priors) != len(root_candidates):
            priors = np.zeros((len(root_candidates),), dtype=np.float32)
            valid = root_mask.astype(bool)
            if valid.any():
                priors[valid] = 1.0 / float(np.sum(valid))
        priors = self._apply_root_dirichlet(priors, root_mask)
        root = MCTSNode(env.get_snapshot(), root_candidates, root_mask, priors=priors, prior=1.0)

        for _ in range(self.config.num_simulations):
            self._simulate(env, root)

        if not root.children:
            env.set_snapshot(root_snapshot)
            self.base_agent.set_selector_state(selector_state)
            return 0

        best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
        env.set_snapshot(root_snapshot)
        self.base_agent.set_selector_state(selector_state)
        return best_action

    def _apply_root_dirichlet(self, priors: np.ndarray, mask: np.ndarray) -> np.ndarray:
        eps = float(self.config.dirichlet_epsilon)
        c = float(self.config.dirichlet_concentration)
        if eps <= 0 or c <= 0:
            return priors
        valid = mask.astype(bool)
        valid_count = int(np.sum(valid))
        if valid_count <= 0:
            return priors
        alpha = c / float(valid_count)
        if alpha <= 0:
            return priors
        noise = np.random.dirichlet([alpha] * valid_count).astype(np.float32)
        mixed = priors.copy()
        mixed_valid = (1.0 - eps) * mixed[valid] + eps * noise
        total = float(np.sum(mixed_valid))
        if total > 0:
            mixed_valid = mixed_valid / total
        mixed[valid] = mixed_valid
        return mixed

    def _simulate(self, env: FactoryLayoutEnvOld, root: MCTSNode) -> None:
        node = root
        env.set_snapshot(root.snapshot)

        path_nodes = [root]
        path_rewards: List[float] = []

        # 1. Selection & Expansion Integrated
        while not node.terminal:
            action = node.best_action(self.config.c_puct)
            if action == -1:  # No valid actions
                break

            if action in node.children:
                # Selection: Move to existing child
                node = node.children[action]
                env.set_snapshot(node.snapshot)
                path_nodes.append(node)
                path_rewards.append(node.reward)
            else:
                # Expansion: Create new child and stop descent
                env.set_snapshot(node.snapshot)
                env.set_candidates(node.candidates, node.mask)
                _, reward, terminated, truncated, _ = env.step(action)
                terminal = terminated or truncated
                snapshot = env.get_snapshot()

                if terminal:
                    candidates, mask = [], np.zeros((0,), dtype=np.int8)
                else:
                    candidates, mask = self.base_agent.get_candidates(env)

                priors = self.base_agent.get_action_priors(env, candidates, mask) if candidates else np.zeros((0,))

                child = MCTSNode(
                    snapshot=snapshot,
                    candidates=candidates,
                    mask=mask,
                    priors=priors,
                    prior=node.priors[action] if action < len(node.priors) else 0.0,
                    parent=node,
                    action=action,
                    reward=reward,
                    terminal=terminal or (not terminal and len(candidates) == 0),
                )
                node.children[action] = child
                node = child
                path_nodes.append(node)
                path_rewards.append(reward)
                break  # Expanded, now rollout from this new node

        # 2. Rollout
        rollout_return = 0.0
        if not node.terminal:
            rollout_return = self._rollout(env)

        # 3. Backprop
        total = rollout_return
        for reward, path_node in zip(reversed(path_rewards), reversed(path_nodes[1:])):
            total += reward
            path_node.visits += 1
            path_node.total_value += total

        # Root update
        root.visits += 1
        root.total_value += total

    def _rollout(self, env: FactoryLayoutEnvOld) -> float:
        total = 0.0
        for _ in range(self.config.rollout_depth):
            candidates, mask = self.base_agent.get_candidates(env)
            if len(candidates) == 0:
                break
            action = self.base_agent.select_action(env, candidates, mask)
            env.set_candidates(candidates, mask)
            _, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        return total

