from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F

from envs.env import FactoryLayoutEnv, GroupId

from .base import BaseWrapper


class MaskPlaceWrapperEnv(BaseWrapper):
    """MaskPlace dense-grid wrapper: Discrete(G*G) actions (default G=224).

    Obs provides:
    - action_mask: bool [G*G] (True means valid)
    - state: float32 [1 + 5*G*G + 2]  (pos_idx + 5 maps + extra2)
      maps are flattened in order: canvas, net_img, mask, net_img_2, mask_2
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        engine: FactoryLayoutEnv,
        grid: int = 224,
        rot: int = 0,
        soft_coefficient: float = 1.0,
    ):
        super().__init__(engine=engine)
        self.grid_width = int(engine.grid_width)
        self.grid_height = int(engine.grid_height)
        self.grid = int(grid)
        self.rot = int(rot)
        self.soft_coefficient = float(soft_coefficient)

        self.action_space = gym.spaces.Discrete(self.grid * self.grid)
        self.observation_space = gym.spaces.Dict({})

        # cached last-built maps (for debugging/visualization)
        self._last_maps: Optional[torch.Tensor] = None  # float32 [5,G,G]

    def cell_wh(self) -> Tuple[int, int]:
        g = int(self.grid)
        cell_w = int(math.ceil(self.grid_width / float(g)))
        cell_h = int(math.ceil(self.grid_height / float(g)))
        return cell_w, cell_h

    def _next_gid(self) -> Optional[GroupId]:
        return self.engine.remaining[0] if self.engine.remaining else None

    def decode_action(self, action: int) -> Tuple[int, int, int, int, int]:
        """Decode dense cell index to bottom-left integer coords (x_bl,y_bl,rot)."""
        gid = self._next_gid()
        if gid is None:
            return 0, 0, 0, 0, 0
        g = int(self.grid)
        a = int(action)
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
        return int(x_bl), int(y_bl), int(rot), int(i), int(j)

    def _valid_top_left_body(self, *, gid: GroupId, rot: int) -> torch.Tensor:
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
            return torch.zeros((self.grid * self.grid,), dtype=torch.bool, device=self.device)

        group = self.engine.groups[gid]
        rot = int(self.rot if group.rotatable else 0)
        body_ok = self._valid_top_left_body(gid=gid, rot=rot)  # bool[H2,W2]
        pad_ok, cL, cB = self._valid_top_left_pad(gid=gid, rot=rot)  # bool[H3,W3]
        H2, W2 = int(body_ok.shape[0]), int(body_ok.shape[1])
        H3, W3 = int(pad_ok.shape[0]), int(pad_ok.shape[1])

        g = int(self.grid)
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

        inside_body = (x_bl >= 0) & (y_bl >= 0) & (x_bl < W2) & (y_bl < H2)
        idx_body = (y_bl * W2 + x_bl).to(torch.long)
        flat_body = body_ok.reshape(-1)

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

    def _build_maps_and_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (maps[5,G,G], state[1+5*G*G+2])."""
        gid = self._next_gid()
        g = int(self.grid)
        g2 = g * g

        # --- hard mask map (1=invalid) derived from current wrapper mask ---
        assert self.mask is not None
        mask_valid = self.mask.view(g, g)
        mask_map = (~mask_valid).to(torch.float32)  # 1 where invalid

        # --- canvas: downsample of occupancy/static (1=occupied/blocked) ---
        canvas_src = (self.engine._occ_invalid | self.engine._static_invalid).to(torch.float32).view(1, 1, self.grid_height, self.grid_width)
        canvas = F.interpolate(canvas_src, size=(g, g), mode="nearest").squeeze(0).squeeze(0)

        # --- net_img: simple weighted L1-to-placed-centers heuristic (lower is better) ---
        net_img = torch.zeros((g, g), dtype=torch.float32, device=self.device)
        if gid is not None and self.engine.placed:
            cell_w, cell_h = self.cell_wh()
            ii = torch.arange(g, device=self.device).view(-1, 1).expand(g, g)
            jj = torch.arange(g, device=self.device).view(1, -1).expand(g, g)
            cx = (jj * cell_w).to(torch.float32) + (cell_w / 2.0)
            cy = (ii * cell_h).to(torch.float32) + (cell_h / 2.0)
            # Use symmetric flow weights (next->j + j->next).
            wmap = {}
            for j in self.engine.placed:
                w = float(self.engine.group_flow.get(gid, {}).get(j, 0.0)) + float(self.engine.group_flow.get(j, {}).get(gid, 0.0))
                if w != 0.0:
                    wmap[j] = w
            if wmap:
                for j, w in wmap.items():
                    jcx, jcy = self.engine.pose_center(j)
                    net_img = net_img + float(w) * (torch.abs(cx - float(jcx)) + torch.abs(cy - float(jcy)))

        net_img_2 = torch.zeros((g, g), dtype=torch.float32, device=self.device)
        mask_2 = mask_map.clone()

        maps = torch.stack([canvas, net_img, mask_map, net_img_2, mask_2], dim=0)  # [5,G,G]

        # state = [pos_idx] + [5 maps flat] + [extra2]
        pos_idx = float(self.engine.gid_to_idx.get(gid, 0)) if gid is not None else 0.0
        extra2 = self.engine._build_obs().get("next_group_wh", torch.zeros((2,), device=self.device, dtype=torch.float32))
        state = torch.empty((1 + 5 * g2 + 2,), dtype=torch.float32, device=self.device)
        state[0] = float(pos_idx)
        state[1 : 1 + 5 * g2] = maps.reshape(-1)
        state[1 + 5 * g2 :] = extra2.to(dtype=torch.float32).view(-1)[:2]
        return maps, state

    def _build_obs(self) -> Dict[str, Any]:
        assert self.mask is not None
        obs = dict(self.engine._build_obs())
        obs["action_mask"] = self.mask
        maps, state = self._build_maps_and_state()
        self._last_maps = maps
        obs["state"] = state
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

    env = MaskPlaceWrapperEnv(engine=engine, grid=224, rot=0)

    t0 = time.perf_counter()
    obs, _info = env.reset(options=loaded.reset_kwargs)
    dt_reset_ms = (time.perf_counter() - t0) * 1000.0

    valid = int(obs["action_mask"].sum().item())
    a = int(torch.where(obs["action_mask"])[0][0].item()) if valid > 0 else 0

    # For visualization, subsample valid points (224^2 can be large).
    gid = env.engine.remaining[0] if env.engine.remaining else None
    idxs = torch.where(obs["action_mask"])[0][:5000]
    xy = torch.zeros((int(idxs.numel()), 3), dtype=torch.long, device=device)
    for t, ai in enumerate(idxs.tolist()):
        x_bl, y_bl, rot, _i, _j = env.decode_action(int(ai))
        xy[t, 0] = int(x_bl)
        xy[t, 1] = int(y_bl)
        xy[t, 2] = int(rot)
    cand0 = CandidateSet(
        xyrot=xy,
        mask=torch.ones((xy.shape[0],), dtype=torch.bool, device=device),
        gid=gid,
        meta={"grid": int(env.grid)},
    )
    plot_layout(env, candidate_set=cand0)

    t1 = time.perf_counter()
    obs2, _r, _term, _trunc, _info2 = env.step(a)
    dt_step_ms = (time.perf_counter() - t1) * 1000.0

    # Plot after one placement (subsample new valid points)
    if isinstance(obs2, dict) and ("action_mask" in obs2):
        gid2 = env.engine.remaining[0] if env.engine.remaining else None
        idxs2 = torch.where(obs2["action_mask"])[0][:5000]
        xy2 = torch.zeros((int(idxs2.numel()), 3), dtype=torch.long, device=device)
        for t, ai in enumerate(idxs2.tolist()):
            x_bl, y_bl, rot, _i, _j = env.decode_action(int(ai))
            xy2[t, 0] = int(x_bl)
            xy2[t, 1] = int(y_bl)
            xy2[t, 2] = int(rot)
        cand1 = CandidateSet(
            xyrot=xy2,
            mask=torch.ones((xy2.shape[0],), dtype=torch.bool, device=device),
            gid=gid2,
            meta={"grid": int(env.grid)},
        )
        plot_layout(env, candidate_set=cand1)
    else:
        plot_layout(env, candidate_set=None)

    print("[MaskPlaceWrapperEnv demo]")
    print(" env=", ENV_JSON, "device=", device, "grid=", env.grid)
    print(" valid_actions=", valid, "first_valid_action=", a, "plotted=", int(xy.shape[0]))
    print(f" reset_ms={dt_reset_ms:.3f} step_ms={dt_step_ms:.3f}")

