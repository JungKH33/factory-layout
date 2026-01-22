from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from envs.env import FactoryLayoutEnv
from agents.base import Agent
from actionspace.base import CandidateSelector
from actionspace.candidate_set import CandidateSet


@dataclass(frozen=True)
class BeamConfig:
    beam_width: int = 8
    depth: int = 5
    expansion_topk: int = 16


class BeamSearch:
    """Beam search over candidates using env snapshots."""

    def __init__(self, *, config: BeamConfig):
        self.config = config

    def select(
        self,
        *,
        env: FactoryLayoutEnv,
        obs: dict,
        agent: Agent,
        selector: CandidateSelector,
        root_candidates: CandidateSet,
        root_selector_state: object,
    ) -> int:
        root_snapshot = env.get_snapshot()
        root_sel_state = root_selector_state

        # Each beam item:
        # (cum_reward, first_action, snapshot, selector_state, obs_at_snapshot)
        beams: List[Tuple[float, int, Dict[str, object], object, dict]] = [(0.0, -1, root_snapshot, root_sel_state, obs)]

        for depth in range(int(self.config.depth)):
            new_beams: List[Tuple[float, int, Dict[str, object], object, dict]] = []
            for cum_reward, first_action, snap, sel_state, obs_node in beams:
                env.set_snapshot(snap)
                selector.set_state(sel_state)

                # Use provided root candidates at depth=0 on root node to avoid rebuild mismatch.
                if depth == 0 and snap is root_snapshot:
                    candidates = root_candidates
                else:
                    candidates = selector.build(env)
                valid_mask = candidates.mask
                valid_n = int(valid_mask.to(torch.int64).sum().item())
                if valid_n <= 0:
                    new_beams.append(
                        (
                            cum_reward,
                            first_action if first_action >= 0 else 0,
                            env.get_snapshot(),
                            selector.get_state(),
                            obs_node,
                        )
                    )
                    continue

                # IMPORTANT: policy must be computed from the obs corresponding to this node state.
                priors = (
                    agent.policy(env=env, obs=obs_node, candidates=candidates)
                    .to(dtype=torch.float32, device=env.device)
                    .view(-1)
                )
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

                    env.set_snapshot(snap)
                    selector.set_state(sel_state)
                    obs2, reward, terminated, truncated, _info = env.step_masked(
                        action=a,
                        x=float(x),
                        y=float(y),
                        rot=int(rot),
                        mask=candidates.mask,
                        action_space_n=int(candidates.mask.shape[0]),
                    )

                    new_cum = float(cum_reward) + float(reward)
                    root_a = a if first_action < 0 else int(first_action)
                    new_beams.append((new_cum, root_a, env.get_snapshot(), selector.get_state(), obs2))

                    if terminated or truncated:
                        pass

            if not new_beams:
                break

            new_beams.sort(key=lambda t: t[0], reverse=True)
            beams = new_beams[: int(self.config.beam_width)]

        best = beams[0][1] if beams else 0

        env.set_snapshot(root_snapshot)
        selector.set_state(root_sel_state)
        return int(best) if int(best) >= 0 else 0


if __name__ == "__main__":
    import time

    from envs.json_loader import load_env
    from agents.greedy import GreedyAgent
    from actionspace.topk import TopKSelector

    ENV_JSON = "env_configs/basic_01.json"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    env = loaded.env
    env.log = False
    obs, _info = env.reset(options=loaded.reset_kwargs)

    selector = TopKSelector(k=50, scan_step=10.0, quant_step=10.0, random_seed=0)
    agent = GreedyAgent(prior_temperature=1.0)
    search = BeamSearch(config=BeamConfig(beam_width=8, depth=3, expansion_topk=16))

    t0 = time.perf_counter()
    root_candidates = selector.build(env)
    root_sel_state = selector.get_state()
    a = search.select(
        env=env,
        obs=obs,
        agent=agent,
        selector=selector,
        root_candidates=root_candidates,
        root_selector_state=root_sel_state,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0

    valid_n = int(root_candidates.mask.sum().item())
    xyrot = root_candidates.xyrot[a].tolist() if int(root_candidates.xyrot.shape[0]) > 0 else [0, 0, 0]

    print("[search.beam demo]")
    print(" env=", ENV_JSON, "device=", device, "next_gid=", (env.remaining[0] if env.remaining else None))
    print(" action=", a, "valid_candidates=", valid_n, "xyrot=", xyrot)
    print(f" elapsed_ms={dt_ms:.2f}")

