from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from envs.env import FactoryLayoutEnv

from .candidate_set import CandidateSet


@dataclass(frozen=True)
class CoarseSelector:
    """Coarse-grid candidates: N=G*G with deterministic (x,y,rot)."""

    coarse_grid: int
    rot: int = 0

    def get_state(self) -> object:
        # Deterministic selector: no RNG/state to snapshot.
        return None

    def set_state(self, state: object) -> None:
        # Deterministic selector: nothing to restore.
        return None

    def _cell_wh(self, env: FactoryLayoutEnv) -> Tuple[int, int]:
        g = int(self.coarse_grid)
        cell_w = int(math.ceil(int(env.grid_width) / float(g)))
        cell_h = int(math.ceil(int(env.grid_height) / float(g)))
        return cell_w, cell_h

    def _xyrot_table(self, env: FactoryLayoutEnv) -> torch.Tensor:
        g = int(self.coarse_grid)
        cell_w, cell_h = self._cell_wh(env)
        device = env.device

        ii = torch.arange(g, device=device).view(-1, 1).expand(g, g)
        jj = torch.arange(g, device=device).view(1, -1).expand(g, g)
        cx = jj * cell_w + int(round(cell_w / 2.0))
        cy = ii * cell_h + int(round(cell_h / 2.0))

        xyrot = torch.zeros((g * g, 3), dtype=torch.long, device=device)
        xyrot[:, 0] = cx.reshape(-1).to(torch.long)
        xyrot[:, 1] = cy.reshape(-1).to(torch.long)
        xyrot[:, 2] = int(self.rot)
        return xyrot

    def _mask(self, env: FactoryLayoutEnv) -> torch.Tensor:
        # Direct copy of `CoarseWrapperEnv.create_mask` logic.
        g = int(self.coarse_grid)
        device = env.device

        if not env.remaining:
            return torch.zeros((g * g,), dtype=torch.bool, device=device)

        gid = env.remaining[0]
        group = env.groups[gid]
        invalid_map = env._invalid  # bool[H,W], True=invalid cell
        H, W = int(invalid_map.shape[0]), int(invalid_map.shape[1])

        kw = max(1, int(math.ceil(float(group.width))))
        kh = max(1, int(math.ceil(float(group.height))))

        inv_f = invalid_map.to(dtype=torch.float32).view(1, 1, H, W)
        kernel = torch.ones((1, 1, kh, kw), device=device, dtype=inv_f.dtype)
        overlap = F.conv2d(inv_f, kernel, padding=0)
        valid_top_left = (overlap == 0).squeeze(0).squeeze(0)  # bool [H2,W2]
        H2, W2 = int(valid_top_left.shape[0]), int(valid_top_left.shape[1])

        cell_w, cell_h = self._cell_wh(env)
        ii = torch.arange(g, device=device).view(-1, 1).expand(g, g)
        jj = torch.arange(g, device=device).view(1, -1).expand(g, g)
        cx = (jj * cell_w).to(torch.float32) + (cell_w / 2.0)
        cy = (ii * cell_h).to(torch.float32) + (cell_h / 2.0)
        tx = torch.round(cx - (kw / 2.0)).to(torch.long)
        ty = torch.round(cy - (kh / 2.0)).to(torch.long)

        inside = (tx >= 0) & (ty >= 0) & (tx < W2) & (ty < H2)
        idx = (ty * W2 + tx).to(torch.long)

        flat_valid = valid_top_left.reshape(-1)
        mask_2d = torch.zeros((g, g), device=device, dtype=torch.bool)
        mask_2d[inside] = flat_valid[idx[inside]]
        return mask_2d.reshape(-1)

    def build(self, env: FactoryLayoutEnv) -> CandidateSet:
        xyrot = self._xyrot_table(env)
        mask = self._mask(env)
        return CandidateSet(xyrot=xyrot, mask=mask, meta={"type": "coarse", "coarse_grid": int(self.coarse_grid)})


if __name__ == "__main__":
    import time

    from envs.json_loader import load_env

    ENV_JSON = "env_configs/basic_01.json"
    COARSE_GRID = 20

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    env = loaded.env
    env.log = False
    obs, _info = env.reset(options=loaded.reset_kwargs)

    selector = CoarseSelector(coarse_grid=COARSE_GRID, rot=0)
    t0 = time.perf_counter()
    candidates = selector.build(env)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print("[actionspace.coarse demo]")
    print(" env=", ENV_JSON, "device=", device, "next_gid=", (env.remaining[0] if env.remaining else None))
    print(" coarse_grid=", COARSE_GRID, "N=", int(candidates.mask.shape[0]), "valid=", int(candidates.mask.sum().item()))
    print(" xyrot.dtype=", candidates.xyrot.dtype, "mask.dtype=", candidates.mask.dtype)
    print(f" elapsed_ms={dt_ms:.3f}")
