from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from envs.env import FactoryLayoutEnv

from .agents import Agent
from .selectors import CandidateSelector
from .search import SearchStrategy


@dataclass(frozen=True)
class DecisionPipeline:
    """Orchestrates agent + selector + optional search (composition-first).

    A-plan:
    - Pipeline owns `agent` and `selector`.
    - Search (if provided) is only the selection logic and does NOT store agent/selector.
    """

    agent: Agent
    selector: CandidateSelector
    search: Optional[SearchStrategy] = None

    def act(self, *, env: FactoryLayoutEnv, obs: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Return (action_index, debug_info). (Does not step the env.)"""
        if self.search is None:
            candidates = self.selector.build(env)
            a = int(self.agent.select_action(env=env, obs=obs, candidates=candidates))
            return a, {"action": a, "search": "none"}
        a = int(self.search.select(env=env, obs=obs, agent=self.agent, selector=self.selector))
        return a, {"action": a, "search": type(self.search).__name__}

    def step(self, *, env: FactoryLayoutEnv, obs: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any], Dict[str, Any]]:
        """Select an action and apply `env.step_masked` using selector-produced candidates.

        Returns:
          obs2, reward, terminated, truncated, info, dbg
        """
        action, dbg = self.act(env=env, obs=obs)

        # IMPORTANT:
        # Search (MCTS/Beam) must restore env+selector state to root before returning action.
        # Then building candidates here reproduces the same CandidateSet that `action` refers to.
        candidates = self.selector.build(env)
        valid_n = int(candidates.mask.sum().item())

        if int(candidates.xyrot.shape[0]) > 0:
            x = int(candidates.xyrot[action, 0].item())
            y = int(candidates.xyrot[action, 1].item())
            rot = int(candidates.xyrot[action, 2].item())
        else:
            x = y = rot = 0

        obs2, reward, terminated, truncated, info = env.step_masked(
            action=int(action),
            x=float(x),
            y=float(y),
            rot=int(rot),
            mask=candidates.mask,
            action_space_n=int(candidates.mask.shape[0]),
        )

        dbg2 = dict(dbg)
        dbg2.update(
            {
                "valid_candidates": valid_n,
                "xyrot": (x, y, rot),
            }
        )
        return obs2, float(reward), bool(terminated), bool(truncated), info, dbg2


if __name__ == "__main__":
    import time
    import torch

    from envs.json_loader import load_env
    from .agents import GreedyAgent
    from .search import MCTSConfig, MCTSSearch
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
    pipe = DecisionPipeline(agent=agent, selector=selector, search=search)

    t0 = time.perf_counter()
    obs2, reward, terminated, truncated, info, dbg = pipe.step(env=env, obs=obs)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print("[pipeline demo]")
    print(" input.env=", ENV_JSON, "next_gid=", (env.remaining[0] if env.remaining else None))
    print(" output.dbg=", dbg)
    print(" step: reward=", reward, "terminated=", terminated, "truncated=", truncated, "reason=", info.get("reason"))
    print(f" elapsed_ms={dt_ms:.2f}")

