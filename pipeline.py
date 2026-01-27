from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from agents.base import Agent
from envs.wrappers.base import BaseWrapper
from envs.wrappers.candidate_set import CandidateSet
from search.base import SearchStrategy


def _candidates_from_wrapper(env: BaseWrapper, obs: Dict[str, Any]) -> CandidateSet:
    """Build a CandidateSet from a wrapper observation.

    - Prefer `obs["action_xyrot"]` when available (TopK wrapper).
    - Otherwise decode all actions (coarse/dense wrappers).
    """
    mask = obs.get("action_mask", None)
    if not isinstance(mask, torch.Tensor):
        raise ValueError("wrapper obs must include torch.Tensor obs['action_mask']")
    mask = mask.to(dtype=torch.bool, device=env.device).view(-1)
    A = int(mask.shape[0])

    gid = env.engine.remaining[0] if env.engine.remaining else None

    if "action_xyrot" in obs and isinstance(obs["action_xyrot"], torch.Tensor):
        xyrot = obs["action_xyrot"].to(dtype=torch.long, device=env.device)
        if xyrot.ndim != 2 or int(xyrot.shape[0]) != A or int(xyrot.shape[1]) != 3:
            raise ValueError(f"obs['action_xyrot'] must have shape [A,3], got {tuple(xyrot.shape)} for A={A}")
        return CandidateSet(xyrot=xyrot, mask=mask, gid=gid)

    # Decode all actions (A can be large; kept simple for now).
    xyrot = torch.zeros((A, 3), dtype=torch.long, device=env.device)
    for a in range(A):
        x_bl, y_bl, rot, _i, _j = env.decode_action(int(a))  # type: ignore[attr-defined]
        xyrot[a, 0] = int(x_bl)
        xyrot[a, 1] = int(y_bl)
        xyrot[a, 2] = int(rot)
    return CandidateSet(xyrot=xyrot, mask=mask, gid=gid)


@dataclass(frozen=True)
class DecisionPipeline:
    """Orchestrates agent + optional search over a wrapper env (actionspace-free).

    Contract:
    - `env` is a wrapper env (GreedyWrapperEnv / AlphaChipWrapperEnv / MaskPlaceWrapperEnv)
      that returns `obs["action_mask"]` and supports `env.step(action)`.
    - `agent` consumes (engine, obs, candidates) using CandidateSet built from wrapper obs.
    - `search` (if provided) selects an action index among candidates.
    """

    agent: Agent
    search: Optional[SearchStrategy] = None

    def act(self, *, env: BaseWrapper, obs: Dict[str, Any]) -> Tuple[int, Dict[str, Any], CandidateSet]:
        """Return (action_index, debug_info, root_candidates). (Does not step the env.)"""
        root_candidates = _candidates_from_wrapper(env, obs)

        if self.search is None:
            a = int(self.agent.select_action(env=env.engine, obs=obs, candidates=root_candidates))
            return a, {"action": a, "search": "none"}, root_candidates

        a = int(self.search.select(env=env, obs=obs, agent=self.agent, root_candidates=root_candidates))
        return a, {"action": a, "search": type(self.search).__name__}, root_candidates

    def step(
        self, *, env: BaseWrapper, obs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any], Dict[str, Any]]:
        """Select an action and apply `env.step(action)` (wrapper owns decode+mask).

        Returns:
          obs2, reward, terminated, truncated, info, dbg
        """
        action, dbg, root_candidates = self.act(env=env, obs=obs)

        obs2, reward, terminated, truncated, info = env.step(int(action))

        dbg2 = dict(dbg)
        valid_n = int(root_candidates.mask.to(torch.int64).sum().item())
        xy = (
            tuple(int(v) for v in root_candidates.xyrot[int(action)].tolist())
            if int(root_candidates.xyrot.shape[0]) > 0
            else (0, 0, 0)
        )
        dbg2.update({"valid_candidates": valid_n, "xyrot": xy})
        dbg2["candidates"] = root_candidates
        return obs2, float(reward), bool(terminated), bool(truncated), info, dbg2


if __name__ == "__main__":
    import time
    import torch

    from envs.json_loader import load_env
    from agents.greedy import GreedyAgent
    from search.mcts import MCTSConfig, MCTSSearch
    from envs.wrappers.greedy import GreedyWrapperEnv

    ENV_JSON = "env_configs/basic_01.json"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False
    env = GreedyWrapperEnv(engine=engine, k=50, scan_step=10.0, quant_step=10.0, random_seed=0)
    obs, _info = env.reset(options=loaded.reset_kwargs)

    agent = GreedyAgent(prior_temperature=1.0)
    search = MCTSSearch(config=MCTSConfig(num_simulations=50, rollout_enabled=True, rollout_depth=5))
    pipe = DecisionPipeline(agent=agent, search=search)

    t0 = time.perf_counter()
    obs2, reward, terminated, truncated, info, dbg = pipe.step(env=env, obs=obs)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print("[pipeline demo]")
    print(" env=", ENV_JSON, "device=", device, "next_gid=", (env.engine.remaining[0] if env.engine.remaining else None))
    print(" dbg=", dbg)
    print(" reward=", reward, "terminated=", terminated, "truncated=", truncated, "reason=", info.get("reason"))
    print(f" elapsed_ms={dt_ms:.2f}")

