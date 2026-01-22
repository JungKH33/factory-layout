from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from agents.greedy import GreedyAgent, GreedyConfig
from envs.env_old import FactoryLayoutEnvOld, FacilityGroup, RectMask
from envs.json_loader import load_env_old
from envs.visualizer_old import plot_layout
from policies.mcts import MCTSAgent, MCTSConfig
from policies.beam_search import BeamSearchAgent, BeamConfig
from policies.topk_selector import TopKConfig
import time


# ----- Switches (code-level toggles) -----
MODEL = "greedy"  # legacy script: only "greedy" is supported in this repo
SEARCH_ALGO = "mcts"  # "mcts", "beam", or None
CHECKPOINT_PATH = None  # kept for backward compatibility (unused)
ENV_JSON = "env_configs/hard_01.json"
MCTS_CONFIG = MCTSConfig(num_simulations=3000, c_puct=0.3, rollout_depth=10)
BEAM_CONFIG = BeamConfig(beam_width=7, depth=5)


GroupId = Union[int, str]


def _load_module(checkpoint_path: str) -> object:
    # NOTE: legacy RLlib PPO modules were removed from this repo.
    # Keep this function for backward-compat docs, but fail with a clear message.
    raise RuntimeError(
        "Legacy RLlib PPO inference is not available: `agents/ppo.py` was removed. "
        "Use MODEL='greedy' (default) or migrate inference to the new env/train pipeline."
    )


def main() -> None:
 
    loaded = load_env_old(ENV_JSON)
    env = loaded.env

    topk_cfg = TopKConfig(
        k=30,
        scan_step=5.0,
        quant_step=10.0,
        p_high=0.1,
        p_near=0.8,
        p_coarse=0.0,
        oversample_factor=2,
        diversity_ratio=0.0,
        min_diversity=10,
        random_seed=8,
    )

    agent = None
    if MODEL == "greedy":
        greedy_cfg = GreedyConfig(topk_config=topk_cfg)
        agent = GreedyAgent(greedy_cfg)
    elif MODEL == "ppo_topk":
        raise RuntimeError(
            "MODEL='ppo_topk' is not supported anymore (legacy RLlib PPO code was removed). "
            "Use MODEL='greedy' or use the new `inference.py` pipeline."
        )
    else:
        raise ValueError(f"Unknown MODEL: {MODEL}")

    mcts = MCTSAgent(agent, MCTS_CONFIG) if SEARCH_ALGO == "mcts" else None
    beam = BeamSearchAgent(agent, BEAM_CONFIG) if SEARCH_ALGO == "beam" else None

    obs, _ = env.reset(**loaded.reset_kwargs)
    terminated = truncated = False
    total_reward = 0.0

    start_time = time.perf_counter()

    while not (terminated or truncated):
        if mcts is not None:
            candidates, mask = mcts.get_candidates(env)
            action = mcts.select_action(env, candidates, mask)
        elif beam is not None:
            candidates, mask = beam.get_candidates(env)
            action = beam.select_action(env, candidates, mask)
        else:
            candidates, mask = agent.get_candidates(env)
            action = agent.select_action(env, candidates, mask)

        if not candidates:
            break

        env.set_candidates(candidates, mask)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    end_time = time.perf_counter()
    print(f"Total computation time: {end_time - start_time:.4f} seconds")
    print(f"episode_reward={total_reward:.3f} terminated={terminated} truncated={truncated}")

    out_dir = Path("results") / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"out_infer_{MODEL}_{ts}.png"
    plot_layout(env, show_masks=True, show_flow=True, show_score=True, candidates=None, save_path=str(out_path), show=False)
    print(f"saved_layout={out_path}")


if __name__ == "__main__":
    main()
