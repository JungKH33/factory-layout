from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F

from envs.env import FactoryLayoutEnv, GroupId

from .base import BaseWrapper


class AlphaChipWrapperEnv(BaseWrapper):
    """AlphaChip-style coarse action wrapper: Discrete(G*G) actions.

    Provides:
    - action_mask: bool [G*G]  (valid-action mask for Discrete)
    - graph tensors: x, edge_index, edge_attr, current_node, netlist_metadata
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        engine: FactoryLayoutEnv,
        coarse_grid: int,
        rot: int = 0,
    ):
        super().__init__(engine=engine)
        self.grid_width = int(engine.grid_width)
        self.grid_height = int(engine.grid_height)
        self.coarse_grid = int(coarse_grid)
        self.rot = int(rot)

        self.action_space = gym.spaces.Discrete(self.coarse_grid * self.coarse_grid)
        self.observation_space = gym.spaces.Dict({})

    def cell_wh(self) -> Tuple[int, int]:
        g = int(self.coarse_grid)
        cell_w = int(math.ceil(self.grid_width / float(g)))
        cell_h = int(math.ceil(self.grid_height / float(g)))
        return cell_w, cell_h

    def _next_gid(self) -> Optional[GroupId]:
        return self.engine.remaining[0] if self.engine.remaining else None

    def decode_action(self, mask_index: int) -> Tuple[float, float, int, int, int]:
        """Decode coarse cell index to bottom-left integer coordinates (x_bl,y_bl,rot)."""
        gid = self._next_gid()
        if gid is None:
            return 0.0, 0.0, 0, 0, 0

        g = int(self.coarse_grid)
        a = int(mask_index)
        i = a // g
        j = a % g

        group = self.engine.groups[gid]
        rot = int(self.rot if group.rotatable else 0)
        w, h = self.engine.rotated_size(group, rot)
        w_i = max(1, int(round(float(w))))
        h_i = max(1, int(round(float(h))))

        cell_w, cell_h = self.cell_wh()
        cx = float(j * cell_w) + (cell_w / 2.0)
        cy = float(i * cell_h) + (cell_h / 2.0)
        x_bl = int(round(cx - (w_i / 2.0)))
        y_bl = int(round(cy - (h_i / 2.0)))
        return float(x_bl), float(y_bl), int(rot), int(i), int(j)

    def _valid_top_left_body(self, *, gid: GroupId, rot: int) -> torch.Tensor:
        """Return bool[H2,W2] where True means body window is clear of invalid and clear_invalid."""
        group = self.engine.groups[gid]
        w, h = self.engine.rotated_size(group, rot)
        kw = max(1, int(round(float(w))))
        kh = max(1, int(round(float(h))))

        inv = self.engine._invalid.to(dtype=torch.float32).view(1, 1, self.grid_height, self.grid_width)
        clr = self.engine._clear_invalid.to(dtype=torch.float32).view(1, 1, self.grid_height, self.grid_width)
        kernel = torch.ones((1, 1, kh, kw), device=self.device, dtype=inv.dtype)

        ov_inv = F.conv2d(inv, kernel, padding=0).squeeze(0).squeeze(0)
        ov_clr = F.conv2d(clr, kernel, padding=0).squeeze(0).squeeze(0)
        return (ov_inv == 0) & (ov_clr == 0)

    def _valid_top_left_pad(self, *, gid: GroupId, rot: int) -> Tuple[torch.Tensor, int, int]:
        """Return (bool[H3,W3], cL, cB) for pad window invalid check.

        Indexing rule: pad top-left is (x_bl - cL, y_bl - cB).
        """
        group = self.engine.groups[gid]
        w, h = self.engine.rotated_size(group, rot)
        w_i = max(1, int(round(float(w))))
        h_i = max(1, int(round(float(h))))

        cL, cR, cB, cT = self.engine._clearance_lrtb(group, rot)
        cL_i, cR_i, cB_i, cT_i = int(cL), int(cR), int(cB), int(cT)
        kw = max(1, w_i + cL_i + cR_i)
        kh = max(1, h_i + cB_i + cT_i)

        inv = self.engine._invalid.to(dtype=torch.float32).view(1, 1, self.grid_height, self.grid_width)
        kernel = torch.ones((1, 1, kh, kw), device=self.device, dtype=inv.dtype)
        ov = F.conv2d(inv, kernel, padding=0).squeeze(0).squeeze(0)
        return (ov == 0), int(cL_i), int(cB_i)

    def create_mask(self) -> torch.Tensor:
        gid = self._next_gid()
        if gid is None:
            return torch.zeros((self.coarse_grid * self.coarse_grid,), dtype=torch.bool, device=self.device)

        group = self.engine.groups[gid]
        rot = int(self.rot if group.rotatable else 0)
        body_ok = self._valid_top_left_body(gid=gid, rot=rot)  # bool[H2,W2]
        pad_ok, cL, cB = self._valid_top_left_pad(gid=gid, rot=rot)  # bool[H3,W3]

        H2, W2 = int(body_ok.shape[0]), int(body_ok.shape[1])
        H3, W3 = int(pad_ok.shape[0]), int(pad_ok.shape[1])

        g = int(self.coarse_grid)
        cell_w, cell_h = self.cell_wh()

        ii = torch.arange(g, device=self.device).view(-1, 1).expand(g, g)
        jj = torch.arange(g, device=self.device).view(1, -1).expand(g, g)
        cx = (jj * cell_w).to(torch.float32) + (cell_w / 2.0)
        cy = (ii * cell_h).to(torch.float32) + (cell_h / 2.0)

        w, h = self.engine.rotated_size(group, rot)
        w_i = max(1, int(round(float(w))))
        h_i = max(1, int(round(float(h))))
        x_bl = torch.round(cx - (w_i / 2.0)).to(torch.long)
        y_bl = torch.round(cy - (h_i / 2.0)).to(torch.long)

        # body index
        inside_body = (x_bl >= 0) & (y_bl >= 0) & (x_bl < W2) & (y_bl < H2)
        idx_body = (y_bl * W2 + x_bl).to(torch.long)
        flat_body = body_ok.reshape(-1)

        # pad index (shifted by clearance)
        px = (x_bl - int(cL)).to(torch.long)
        py = (y_bl - int(cB)).to(torch.long)
        inside_pad = (px >= 0) & (py >= 0) & (px < W3) & (py < H3)
        idx_pad = (py * W3 + px).to(torch.long)
        flat_pad = pad_ok.reshape(-1)

        ok = torch.zeros((g, g), device=self.device, dtype=torch.bool)
        ok[inside_body] = flat_body[idx_body[inside_body]]
        # Pad must be inside bounds; otherwise invalid.
        ok = ok & inside_pad
        ok[inside_pad] = ok[inside_pad] & flat_pad[idx_pad[inside_pad]]
        return ok.reshape(-1)

    def _build_obs(self) -> Dict[str, Any]:
        assert self.mask is not None
        obs = dict(self.engine._build_obs())
        # Attach AlphaChip graph tensors from engine caches.
        gid = self._next_gid()
        cur_idx = int(self.engine.gid_to_idx.get(gid, 0)) if gid is not None else 0
        obs["x"] = self.engine._node_feat
        obs["edge_index"] = self.engine._edge_index
        obs["edge_attr"] = self.engine._edge_attr
        obs["current_node"] = torch.tensor([cur_idx], dtype=torch.long, device=self.device)
        obs["netlist_metadata"] = self.engine._netlist_metadata
        obs["action_mask"] = self.mask
        return obs

    def step(self, action: int):
        assert self.mask is not None
        x_bl, y_bl, rot, i, j = self.decode_action(int(action))
        obs_core, reward, terminated, truncated, info = self.engine.step_masked(
            action=int(action),
            x=float(x_bl),
            y=float(y_bl),
            rot=int(rot),
            mask=self.mask,
            action_space_n=int(self.action_space.n),
            extra_info={"cell_i": int(i), "cell_j": int(j)},
        )
        if not (terminated or truncated):
            self.mask = self.create_mask()
            return self._build_obs(), reward, terminated, truncated, info
        return obs_core, reward, terminated, truncated, info


if __name__ == "__main__":
    import time

    import torch

    from actionspace.candidate_set import CandidateSet
    from envs.json_loader import load_env
    from envs.visualizer import plot_layout

    ENV_JSON = "env_configs/constraints_01.json"
    device = torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False

    env = AlphaChipWrapperEnv(engine=engine, coarse_grid=32, rot=0)

    t0 = time.perf_counter()
    obs, _info = env.reset(options=loaded.reset_kwargs)
    dt_reset_ms = (time.perf_counter() - t0) * 1000.0

    valid = int(obs["action_mask"].sum().item())
    a = int(torch.where(obs["action_mask"])[0][0].item()) if valid > 0 else 0

    # Build BL candidates for visualization (avoid plotting all invalid points).
    gid = env.engine.remaining[0] if env.engine.remaining else None
    g = int(env.coarse_grid)
    idxs = torch.where(obs["action_mask"])[0]
    xy = torch.zeros((int(idxs.numel()), 3), dtype=torch.long, device=device)
    for t, ai in enumerate(idxs.tolist()):
        x_bl, y_bl, rot, _i, _j = env.decode_action(int(ai))
        xy[t, 0] = int(x_bl)
        xy[t, 1] = int(y_bl)
        xy[t, 2] = int(rot)

    cand0 = CandidateSet(xyrot=xy, mask=torch.ones((xy.shape[0],), dtype=torch.bool, device=device), gid=gid, meta={"g": g})
    plot_layout(env, candidate_set=cand0)

    t1 = time.perf_counter()
    obs2, _r, _term, _trunc, _info2 = env.step(a)
    dt_step_ms = (time.perf_counter() - t1) * 1000.0

    # Plot after one placement (new candidates)
    if isinstance(obs2, dict) and ("action_mask" in obs2):
        gid2 = env.engine.remaining[0] if env.engine.remaining else None
        idxs2 = torch.where(obs2["action_mask"])[0]
        xy2 = torch.zeros((int(idxs2.numel()), 3), dtype=torch.long, device=device)
        for t, ai in enumerate(idxs2.tolist()):
            x_bl, y_bl, rot, _i, _j = env.decode_action(int(ai))
            xy2[t, 0] = int(x_bl)
            xy2[t, 1] = int(y_bl)
            xy2[t, 2] = int(rot)
        cand1 = CandidateSet(xyrot=xy2, mask=torch.ones((xy2.shape[0],), dtype=torch.bool, device=device), gid=gid2, meta={"g": g})
        plot_layout(env, candidate_set=cand1)
    else:
        plot_layout(env, candidate_set=None)

    print("[AlphaChipWrapperEnv demo]")
    print(" env=", ENV_JSON, "device=", device, "G=", env.coarse_grid)
    print(" valid_actions=", valid, "first_valid_action=", a)
    print(f" reset_ms={dt_reset_ms:.3f} step_ms={dt_step_ms:.3f}")

