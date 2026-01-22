from __future__ import annotations

"""Run quick demos for the decision package.

Usage:
  python -m decision
  python -m decision.selectors
  python -m decision.search
"""

import time

import torch

from envs.json_loader import load_env

from .agents import GreedyAgent
from .pipeline import DecisionPipeline
from .search import MCTSConfig, MCTSSearch
from .selectors import TopKSelector


def main() -> None:
    ENV_JSON = "env_configs/basic_01.json"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    env = loaded.env
    env.log = False
    obs, _ = env.reset(options=loaded.reset_kwargs)

    print("[decision __main__]")
    print(" input.env=", ENV_JSON, "grid=", (env.grid_width, env.grid_height), "device=", env.device)

    selector = TopKSelector(k=50, scan_step=10.0, quant_step=10.0, random_seed=0)
    agent = GreedyAgent(prior_temperature=1.0)
    search = MCTSSearch(config=MCTSConfig(num_simulations=50, rollout_depth=5))
    pipe = DecisionPipeline(agent=agent, selector=selector, search=search)

    t0 = time.perf_counter()
    a, dbg = pipe.act(env=env, obs=obs)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    cand = selector.build(env)
    valid_n = int(cand.mask.sum().item())
    xyrot = cand.xyrot[a].tolist() if valid_n > 0 else [0.0, 0.0, 0.0]
    print(" output.action=", a, "xyrot=", xyrot, "dbg=", dbg)
    print(" candidates.valid=", valid_n, "N=", int(cand.mask.shape[0]))
    print(f" elapsed_ms={dt_ms:.2f} (MCTS 50 sims, rollout_depth 5)")


if __name__ == "__main__":
    main()

