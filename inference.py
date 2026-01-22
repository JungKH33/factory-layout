from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import time
import torch
from torch_geometric.data import Data

from agents.alphachip.model import AlphaChip
from envs.wrappers import CoarseWrapperEnv
from envs.visualizer import save_layout
from envs.json_loader import load_env


# ----- Switches (code-level toggles) -----
# If CHECKPOINT_PATH is set, it takes precedence.
# Otherwise set RUN_ID and USE_BEST to load from results/checkpoints/<run_id>/(best|latest).ckpt
CHECKPOINT_PATH: str | None = r"D:\developments\Projects\factory-layout\results\checkpoints\2026-01-19_16-17_f34368\best.ckpt"
RUN_ID: str | None = None
USE_BEST: bool = True

# Environment / wrapper config (constants)
ENV_JSON: str = "env_configs/basic_01.json"
COARSE_GRID: int = 32
ROT: int = 0
SHOW_FLOW: bool = True
SHOW_SCORE: bool = True
SHOW_MASKS: bool = True

def _obs_to_pyg_data(obs: Dict[str, torch.Tensor]) -> Data:
    # env provides torch tensors already.
    return Data(
        x=obs["x"],
        edge_index=obs["edge_index"],
        edge_attr=obs["edge_attr"],
        netlist_metadata=obs["netlist_metadata"].view(1, -1),
        current_node=obs["current_node"].view(-1),
    )


def _resolve_checkpoint_path() -> Path | None:
    if CHECKPOINT_PATH:
        return Path(CHECKPOINT_PATH)
    if RUN_ID:
        fname = "best.ckpt" if USE_BEST else "latest.ckpt"
        return Path("results") / "checkpoints" / RUN_ID / fname
    return None


def _load_checkpoint(model: AlphaChip, *, device: torch.device) -> Path | None:
    path = _resolve_checkpoint_path()
    if path is None:
        print("[inference] checkpoint: none (random weights)")
        return None
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    obj = torch.load(str(path), map_location=device)
    state = obj.get("model") if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format at {path} (expected state_dict or dict with key 'model').")
    model.load_state_dict(state)
    model.eval()
    meta = obj.get("meta") if isinstance(obj, dict) else None
    print(f"[inference] loaded_checkpoint={path}")
    if isinstance(meta, dict):
        print(f"[inference] checkpoint_meta={meta}")
    return path


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env

    # Wrapper is created separately (as requested).
    env = CoarseWrapperEnv(engine=engine, coarse_grid=int(32), rot=int(0))

    model = AlphaChip(
        metadata_dim=12,
        node_feature_dim=8,
        max_grid_size=int(COARSE_GRID),
        device=device,
    )
    _load_checkpoint(model, device=device)

    # JSON reset options (initial_positions/remaining_order) are passed through.
    obs, _info = env.reset(options=loaded.reset_kwargs)
    terminated = truncated = False
    total_reward = 0.0

    start = time.perf_counter()
    step = 0
    while not (terminated or truncated):
        step += 1

        data = _obs_to_pyg_data(obs)
        mask_flat = obs["mask_flat"].view(1, -1).to(dtype=torch.int32)
        logits_flat, _value = model(data, mask_flat=mask_flat, is_eval=True)  # [1, A]

        action = int(torch.argmax(logits_flat[0]).item())
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            # debug-friendly final line
            reason = info.get("reason", None)
            print(
                f"[env] end: terminated={terminated} truncated={truncated} "
                f"step={step} placed={len(env.engine.placed)} cost={env.engine.cal_obj():.3f} reason={reason}"
            )

    end = time.perf_counter()
    print(f"Total computation time: {end - start:.4f} seconds")
    print(f"episode_reward={total_reward:.3f} terminated={terminated} truncated={truncated}")

    out_dir = Path("results") / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"out_infer_alphachip_{ts}.png"

    save_layout(
        env,
        show_masks=SHOW_MASKS,
        show_flow=SHOW_FLOW,
        show_score=SHOW_SCORE,
        show_zones=False,
        mask_flat=obs.get("mask_flat"),
        save_path=str(out_path),
    )
    print(f"saved_layout={out_path}")


if __name__ == "__main__":
    main()

