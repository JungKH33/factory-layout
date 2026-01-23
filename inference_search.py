from __future__ import annotations

from datetime import datetime
from pathlib import Path

import time
import torch

from envs.json_loader import load_env
from envs.visualizer import plot_layout, save_layout

from agents.greedy import GreedyAgent
from agents.alphachip.agent import AlphaChipAgent
from pipeline import DecisionPipeline
from search import BeamConfig, BeamSearch, MCTSConfig, MCTSSearch
from actionspace.topk import TopKSelector
from actionspace.coarse import CoarseSelector


# --- config (module-level constants, keep simple) ---
ENV_JSON: str = "env_configs/constraints_01.json"
ACTIONSPACE_MODE: str = "topk"  # "topk" | "coarse"
AGENT_MODE: str = "greedy"  # "greedy" | "alphachip"
ALPHACHIP_CHECKPOINT_PATH: str | None = r"D:\developments\Projects\factory-layout\results\checkpoints\2026-01-19_16-17_f34368\best.ckpt"  # required when AGENT_MODE="alphachip"

TOPK_K: int = 50
TOPK_SCAN_STEP: float = 5.0
TOPK_QUANT_STEP: float = 10.0
COARSE_GRID: int = 32
MCTS_SIMS: int = 1000
ROLLOUT_DEPTH: int = 10
SEARCH_MODE: str = "mcts"  # "none" | "mcts" | "beam"

SHOW_FLOW: bool = True
SHOW_SCORE: bool = True
SHOW_MASKS: bool = True


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    env = loaded.env
    # MCTS will call env.step many times; keep logs quiet during search.
    env.log = True

    print("[inference_search]")
    print(
        " ENV_JSON=", ENV_JSON,
        "ACTIONSPACE_MODE=", ACTIONSPACE_MODE,
        "AGENT_MODE=", AGENT_MODE,
        "TOPK_K=", TOPK_K,
        "TOPK_SCAN_STEP=", TOPK_SCAN_STEP,
        "TOPK_QUANT_STEP=", TOPK_QUANT_STEP,
        "COARSE_GRID=", COARSE_GRID,
        "MCTS_SIMS=", MCTS_SIMS,
        "ROLLOUT_DEPTH=", ROLLOUT_DEPTH,
        "device=", device,
    )

    if ACTIONSPACE_MODE == "topk":
        selector = TopKSelector(
            k=TOPK_K,
            scan_step=TOPK_SCAN_STEP,
            quant_step=TOPK_QUANT_STEP,
            random_seed=5,
        )
    elif ACTIONSPACE_MODE == "coarse":
        selector = CoarseSelector(coarse_grid=int(COARSE_GRID), rot=0)
    else:
        raise ValueError(f"Unknown ACTIONSPACE_MODE={ACTIONSPACE_MODE!r} (expected 'topk'|'coarse')")

    if AGENT_MODE == "greedy":
        agent = GreedyAgent(prior_temperature=1.0)
    elif AGENT_MODE == "alphachip":
        if ACTIONSPACE_MODE != "coarse":
            raise ValueError("AlphaChipAgent currently supports ACTIONSPACE_MODE='coarse' only.")
        if not ALPHACHIP_CHECKPOINT_PATH:
            raise ValueError("ALPHACHIP_CHECKPOINT_PATH must be set when AGENT_MODE='alphachip'.")
        agent = AlphaChipAgent(coarse_grid=int(COARSE_GRID), checkpoint_path=str(ALPHACHIP_CHECKPOINT_PATH), device=device)
    else:
        raise ValueError(f"Unknown AGENT_MODE={AGENT_MODE!r} (expected 'greedy'|'alphachip')")
    # A-plan: pipeline owns agent+selector; search is optional.
    # - To disable search: set `search=None` below.
    # - To enable MCTS: keep `search=MCTSSearch(...)`.
    if SEARCH_MODE == "mcts":
        search = MCTSSearch(
            config=MCTSConfig(
                num_simulations=MCTS_SIMS,
                rollout_depth=ROLLOUT_DEPTH,
                dirichlet_epsilon=0.1,
                dirichlet_concentration=0.5,
            )
        )
    elif SEARCH_MODE == "beam":
        search = BeamSearch(config=BeamConfig(beam_width=8, depth=5, expansion_topk=16))
    elif SEARCH_MODE == "none":
        search = None
    else:
        raise ValueError(f"Unknown SEARCH_MODE={SEARCH_MODE!r} (expected 'none'|'mcts'|'beam')")

    pipeline = DecisionPipeline(agent=agent, selector=selector, search=search)

    obs, _info = env.reset(options=loaded.reset_kwargs)
    terminated = truncated = False
    total_reward = 0.0

    start = time.perf_counter()
    step = 0

    while not (terminated or truncated):
        step += 1
        next_gid = env.remaining[0] if env.remaining else None
        obs, reward, terminated, truncated, info, dbg = pipeline.step(env=env, obs=obs)
        total_reward += float(reward)
        print(f"[step] {step} next_gid={next_gid} dbg={dbg}")

        if terminated or truncated:
            reason = info.get("reason", None)
            print(
                f"[env] end: terminated={terminated} truncated={truncated} "
                f"step={step} placed={len(env.placed)} cost={env.cal_obj():.3f} reason={reason}"
            )

    end = time.perf_counter()
    print(f"Total computation time: {end - start:.4f} seconds")
    print(f"episode_reward={total_reward:.3f} terminated={terminated} truncated={truncated}")

    out_dir = Path("results") / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{ts}_{AGENT_MODE}_{ACTIONSPACE_MODE}_{SEARCH_MODE}.png"

    # Preview before saving (interactive; close the window to continue).
    plot_layout(env)

    save_layout(
        env,
        show_masks=SHOW_MASKS,
        show_flow=SHOW_FLOW,
        show_score=SHOW_SCORE,
        show_zones=False,
        candidate_set=None,
        save_path=str(out_path),
    )
    print(f"saved_layout={out_path}")


if __name__ == "__main__":
    main()

