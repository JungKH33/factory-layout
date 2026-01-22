from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from envs.env import FactoryLayoutEnv

from agents.base import Agent
from actionspace.base import CandidateSelector
from search.base import SearchStrategy


@dataclass(frozen=True)
class DecisionPipeline:
    """Orchestrates agent + selector + optional search (composition-first).

    - Pipeline owns `agent` and `selector`.
    - Search (if provided) is only the selection logic and does NOT store agent/selector.
    """

    agent: Agent
    selector: CandidateSelector
    search: Optional[SearchStrategy] = None

    def act(self, *, env: FactoryLayoutEnv, obs: Dict[str, Any]) -> Tuple[int, Dict[str, Any], object]:
        """Return (action_index, debug_info, root_candidates). (Does not step the env.)"""
        root_candidates = self.selector.build(env)
        root_selector_state = self.selector.get_state()

        if self.search is None:
            a = int(self.agent.select_action(env=env, obs=obs, candidates=root_candidates))
            return a, {"action": a, "search": "none"}, root_candidates

        a = int(
            self.search.select(
                env=env,
                obs=obs,
                agent=self.agent,
                selector=self.selector,
                root_candidates=root_candidates,
                root_selector_state=root_selector_state,
            )
        )
        return a, {"action": a, "search": type(self.search).__name__}, root_candidates

    def step(
        self, *, env: FactoryLayoutEnv, obs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any], Dict[str, Any]]:
        """Select an action and apply `env.step_masked` using selector-produced candidates.

        Returns:
          obs2, reward, terminated, truncated, info, dbg
        """
        action, dbg, root_candidates = self.act(env=env, obs=obs)

        # IMPORTANT (C-plan):
        # Do NOT rebuild candidates here. `action` is defined against `root_candidates`.
        candidates = root_candidates
        valid_n = int(candidates.mask.sum().item())

        if int(candidates.xyrot.shape[0]) > 0:
            x = float(candidates.xyrot[action, 0].item())
            y = float(candidates.xyrot[action, 1].item())
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
        dbg2.update({"valid_candidates": valid_n, "xyrot": (x, y, rot)})
        return obs2, float(reward), bool(terminated), bool(truncated), info, dbg2


if __name__ == "__main__":
    import time
    import torch

    from envs.json_loader import load_env
    from agents.greedy import GreedyAgent
    from actionspace.topk import TopKSelector
    from search.mcts import MCTSConfig, MCTSSearch

    ENV_JSON = "env_configs/basic_01.json"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    env = loaded.env
    env.log = False
    obs, _info = env.reset(options=loaded.reset_kwargs)

    agent = GreedyAgent(prior_temperature=1.0)
    selector = TopKSelector(k=50, scan_step=10.0, quant_step=10.0, random_seed=0)
    search = MCTSSearch(config=MCTSConfig(num_simulations=50, rollout_depth=5))
    pipe = DecisionPipeline(agent=agent, selector=selector, search=search)

    t0 = time.perf_counter()
    obs2, reward, terminated, truncated, info, dbg = pipe.step(env=env, obs=obs)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print("[pipeline demo]")
    print(" env=", ENV_JSON, "device=", device, "next_gid=", (env.remaining[0] if env.remaining else None))
    print(" dbg=", dbg)
    print(" reward=", reward, "terminated=", terminated, "truncated=", truncated, "reason=", info.get("reason"))
    print(f" elapsed_ms={dt_ms:.2f}")

