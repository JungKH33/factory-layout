from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import torch
import torch.nn.functional as F

from abc import ABC, abstractmethod

from envs.env import FactoryLayoutEnv


class BaseWrapperEnv(gym.Env, ABC):
    """Base wrapper: holds an engine and provides mask lifecycle."""

    metadata = {"render_modes": []}

    def __init__(self, engine: FactoryLayoutEnv):
        super().__init__()
        self.engine = engine
        self.device = engine.device
        self.mask: Optional[torch.Tensor] = None

    @abstractmethod
    def create_mask(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int):
        """Wrapper step must be implemented by subclasses."""
        raise NotImplementedError

    def _build_obs(self) -> Dict[str, torch.Tensor]:
        obs = self.engine._build_obs()
        assert self.mask is not None
        obs["mask_flat"] = self.mask
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = self.engine.reset(seed=seed, options=options)
        self.mask = self.create_mask()
        return self._build_obs(), info


class CoarseWrapperEnv(BaseWrapperEnv):
    """Gymnasium env wrapper for coarse-grid Discrete(G*G) actions.

    Design:
    - "Codec" is no longer used. This wrapper *is* the action adapter.
    - The underlying engine is `FactoryLayoutEnv` (logic engine).
    - mask is always a 1D torch.BoolTensor [G*G].
    - decode is deterministic -> no caching of xyrot table.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        engine: FactoryLayoutEnv,
        coarse_grid: int,
        rot: int = 0,
    ):
        super().__init__(engine)
        self.grid_width = int(engine.grid_width)
        self.grid_height = int(engine.grid_height)
        self.coarse_grid = int(coarse_grid)
        self.rot = int(rot)

        self.action_space = gym.spaces.Discrete(self.coarse_grid * self.coarse_grid)
        self.observation_space = gym.spaces.Dict({})

    # --- deterministic decode (no cache needed) ---
    def cell_wh(self) -> Tuple[int, int]:
        g = int(self.coarse_grid)
        cell_w = int(math.ceil(self.grid_width / float(g)))
        cell_h = int(math.ceil(self.grid_height / float(g)))
        return cell_w, cell_h

    def decode_action(self, mask_index: int) -> Tuple[float, float, int, int, int]:
        g = int(self.coarse_grid)
        a = int(mask_index)
        i = a // g
        j = a % g
        cell_w, cell_h = self.cell_wh()
        # Center coordinates must be integer (project convention).
        x = int(round(j * cell_w + (cell_w / 2.0)))
        y = int(round(i * cell_h + (cell_h / 2.0)))
        return float(x), float(y), int(self.rot), int(i), int(j)

    # --- mask ---
    def create_mask(self) -> torch.Tensor:
        if not self.engine.remaining:
            return torch.zeros((self.coarse_grid * self.coarse_grid,), dtype=torch.bool, device=self.device)

        gid = self.engine.remaining[0]
        group = self.engine.groups[gid]
        invalid_map = self.engine._invalid  # [H,W] bool (engine-owned cached map)

        g = int(self.coarse_grid)
        H, W = int(invalid_map.shape[0]), int(invalid_map.shape[1])
        # assume engine invariant: invalid_map is bool[H,W] matching grid

        kw = max(1, int(math.ceil(float(group.width))))
        kh = max(1, int(math.ceil(float(group.height))))

        inv_f = invalid_map.to(dtype=torch.float32).view(1, 1, H, W)
        kernel = torch.ones((1, 1, kh, kw), device=self.device, dtype=inv_f.dtype)
        overlap = F.conv2d(inv_f, kernel, padding=0)  # [1,1,H-kh+1,W-kw+1]
        valid_top_left = (overlap == 0).squeeze(0).squeeze(0)  # bool [H2,W2]
        H2, W2 = int(valid_top_left.shape[0]), int(valid_top_left.shape[1])

        cell_w, cell_h = self.cell_wh()
        ii = torch.arange(g, device=self.device).view(-1, 1).expand(g, g)
        jj = torch.arange(g, device=self.device).view(1, -1).expand(g, g)
        cx = (jj * cell_w).to(torch.float32) + (cell_w / 2.0)
        cy = (ii * cell_h).to(torch.float32) + (cell_h / 2.0)
        tx = torch.round(cx - (kw / 2.0)).to(torch.long)
        ty = torch.round(cy - (kh / 2.0)).to(torch.long)

        inside = (tx >= 0) & (ty >= 0) & (tx < W2) & (ty < H2)
        idx = (ty * W2 + tx).to(torch.long)

        flat_valid = valid_top_left.reshape(-1)
        mask_2d = torch.zeros((g, g), device=self.device, dtype=torch.bool)
        mask_2d[inside] = flat_valid[idx[inside]]
        return mask_2d.reshape(-1)

    def step(self, action: int):
        x, y, rot, i, j = self.decode_action(int(action))
        obs, reward, terminated, truncated, info = self.engine.step_masked(
            action=int(action),
            x=float(x),
            y=float(y),
            rot=int(rot),
            mask=self.mask,
            action_space_n=int(self.action_space.n),
            extra_info={"cell_i": i, "cell_j": j},
        )
        if not (terminated or truncated):
            self.mask = self.create_mask()
            obs = self._build_obs()
        return obs, reward, terminated, truncated, info


class TopKWrapperEnv(BaseWrapperEnv):
    """Gymnasium env wrapper for Top-K candidate actions: Discrete(K).

    This is a *direct* in-file transplant of `policies/topk_selector.py` behavior,
    but adapted to:
    - torch outputs: `mask_flat` is torch.BoolTensor[K]
    - action table: `action_xyrot` is torch.FloatTensor[K,3] (raw coords)
    - engine: `FactoryLayoutEnv` (logic engine)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        engine: FactoryLayoutEnv,
        k: int,
        scan_step: float = 2000.0,
        quant_step: Optional[float] = None,
        # Defaults MUST match `policies/topk_selector.py:TopKConfig` for parity.
        p_high: float = 0.55,
        p_near: float = 0.25,
        p_coarse: float = 0.12,
        oversample_factor: int = 4,
        diversity_ratio: float = 0.15,  # kept for parity with original config (not used in current impl)
        min_diversity: int = 10,  # kept for parity with original config (not used in current impl)
        random_seed: Optional[int] = None,
    ):
        super().__init__(engine)
        self.k = int(k)
        self.scan_step = float(scan_step)
        self.quant_step = float(quant_step) if quant_step is not None else None
        self.p_high = float(p_high)
        self.p_near = float(p_near)
        self.p_coarse = float(p_coarse)
        self.oversample_factor = int(oversample_factor)
        self.diversity_ratio = float(diversity_ratio)
        self.min_diversity = int(min_diversity)
        self._rng = random.Random(random_seed)

        self.action_space = gym.spaces.Discrete(self.k)
        self.observation_space = gym.spaces.Dict({})

        # Cached action table for the current step.
        self.action_xyrot: Optional[torch.Tensor] = None  # float32 [K,3]
        self._candidates: List[Tuple[Union[int, str], float, float, int]] = []

    def _build_obs(self) -> Dict[str, torch.Tensor]:
        obs = super()._build_obs()
        assert self.action_xyrot is not None
        obs["action_xyrot"] = self.action_xyrot
        return obs

    def decode_action(self, mask_index: int) -> Tuple[float, float, int, int, int]:
        """Decode candidate index to (x,y,rot) for visualization; i/j are placeholders."""
        a = int(mask_index)
        if self.action_xyrot is None or a < 0 or a >= self.k:
            return 0.0, 0.0, 0, 0, 0
        xyz = self.action_xyrot[a]
        return float(xyz[0].item()), float(xyz[1].item()), int(round(float(xyz[2].item()))), 0, a

    # --- wrapper lifecycle ---
    def create_mask(self) -> torch.Tensor:
        if not self.engine.remaining:
            self._candidates = []
            # TopK action table must use integer center positions.
            self.action_xyrot = torch.zeros((self.k, 3), dtype=torch.long, device=self.device)
            return torch.zeros((self.k,), dtype=torch.bool, device=self.device)

        gid = self.engine.remaining[0]
        candidates, mask = self._generate(self.engine, gid)
        self._candidates = candidates

        # IMPORTANT: store integer center coordinates.
        xyrot = torch.zeros((self.k, 3), dtype=torch.long, device=self.device)
        for i, c in enumerate(candidates[: self.k]):
            _, x, y, rot = c
            xyrot[i, 0] = int(round(float(x)))
            xyrot[i, 1] = int(round(float(y)))
            xyrot[i, 2] = int(rot)
        self.action_xyrot = xyrot
        return mask

    def step(self, action: int):
        a = int(action)
        if self.action_xyrot is None or a < 0 or a >= self.k:
            x = 0.0
            y = 0.0
            rot = 0
        else:
            x = float(self.action_xyrot[a, 0].item())
            y = float(self.action_xyrot[a, 1].item())
            rot = int(round(float(self.action_xyrot[a, 2].item())))

        obs, reward, terminated, truncated, info = self.engine.step_masked(
            action=int(a),
            x=float(x),
            y=float(y),
            rot=int(rot),
            mask=self.mask,
            action_space_n=int(self.k),
            extra_info={"cand_idx": a},
        )
        if not (terminated or truncated):
            self.mask = self.create_mask()
            obs = self._build_obs()
        return obs, reward, terminated, truncated, info

    # --- TopK transplant (adapted from policies/topk_selector.py) ---
    def _generate(
        self, env: FactoryLayoutEnv, next_group_id: Union[int, str]
    ) -> Tuple[List[Tuple[Union[int, str], float, float, int]], torch.Tensor]:
        q = self.quant_step if self.quant_step is not None else self.scan_step

        if len(env.placed) == 0:
            return self._generate_initial(env, next_group_id, q)

        n_high, n_near, n_coarse, n_rand = self._quota(self.k)
        group = env.groups[next_group_id]

        raw_tagged: List[Tuple[int, Tuple[Union[int, str], float, float, int]]] = []
        raw_tagged.extend((0, c) for c in self._source_high(env, next_group_id, n_high * self.oversample_factor))
        raw_tagged.extend((1, c) for c in self._source_near(env, next_group_id, n_near * self.oversample_factor))
        raw_tagged.extend((2, c) for c in self._source_coarse(env, next_group_id, n_coarse * self.oversample_factor))
        raw_tagged.extend((3, c) for c in self._source_random(env, next_group_id, n_rand * self.oversample_factor))

        unique_tagged = self._dedup_tagged(raw_tagged, q, group)
        valid_tagged = [
            (src, c) for src, c in unique_tagged if env.is_placeable(next_group_id, float(c[1]), float(c[2]), int(c[3]))
        ]

        pools: Dict[int, List[Tuple[Union[int, str], float, float, int]]] = {0: [], 1: [], 2: [], 3: []}
        for src, c in valid_tagged:
            pools[src].append(c)

        final: List[Tuple[Union[int, str], float, float, int]] = []
        final.extend(self._score_sorted(env, next_group_id, pools[0])[:n_high])
        final.extend(self._score_sorted(env, next_group_id, pools[1])[:n_near])
        final.extend(self._random_take(pools[2], n_coarse))
        final.extend(self._random_take(pools[3], n_rand))

        final = final[: self.k]
        mask = torch.zeros((self.k,), dtype=torch.bool, device=self.device)
        if final:
            mask[: len(final)] = True
        if len(final) < self.k:
            final.extend(self._pad_candidates(next_group_id, self.k - len(final)))

        return final, mask

    def _generate_initial(
        self,
        env: FactoryLayoutEnv,
        gid: Union[int, str],
        quant_step: float,
    ) -> Tuple[List[Tuple[Union[int, str], float, float, int]], torch.Tensor]:
        group = env.groups[gid]
        total_k = self.k * self.oversample_factor
        n_strat_target = round(total_k * 0.9)
        n_rand = total_k - n_strat_target

        raw_tagged: List[Tuple[int, Tuple[Union[int, str], float, float, int]]] = []
        raw_tagged.extend((0, c) for c in self._source_stratified(env, gid, n_strat_target))
        raw_tagged.extend((1, c) for c in self._source_random(env, gid, n_rand))

        unique_tagged = self._dedup_tagged(raw_tagged, quant_step, group)
        valid_candidates = [c for _, c in unique_tagged if env.is_placeable(gid, float(c[1]), float(c[2]), int(c[3]))]

        final = valid_candidates[: self.k]
        mask = torch.zeros((self.k,), dtype=torch.bool, device=self.device)
        if final:
            mask[: len(final)] = True
        if len(final) < self.k:
            final.extend(self._pad_candidates(gid, self.k - len(final)))
        return final, mask

    def _source_stratified(
        self, env: FactoryLayoutEnv, gid: Union[int, str], count: int
    ) -> List[Tuple[Union[int, str], float, float, int]]:
        if count <= 0:
            return []

        group = env.groups[gid]
        rotations = (0, 90) if group.rotatable else (0,)

        count_per_rot = max(1, count // len(rotations))
        aspect_ratio = float(env.grid_width) / float(env.grid_height)
        nx = max(1, round(math.sqrt(count_per_rot * aspect_ratio)))
        ny = max(1, round(count_per_rot / nx))

        dx = float(env.grid_width) / float(nx)
        dy = float(env.grid_height) / float(ny)

        candidates: List[Tuple[Union[int, str], float, float, int]] = []
        for rot in rotations:
            for i in range(nx):
                for j in range(ny):
                    x = (i + 0.5) * dx
                    y = (j + 0.5) * dy
                    candidates.append((gid, float(x), float(y), int(rot)))
        return candidates

    def _quota(self, k: int) -> Tuple[int, int, int, int]:
        n_high = round(k * self.p_high)
        n_near = round(k * self.p_near)
        n_coarse = round(k * self.p_coarse)
        n_rand = k - (n_high + n_near + n_coarse)
        if n_rand < 0:
            n_rand = 0
        return int(n_high), int(n_near), int(n_coarse), int(n_rand)

    def _score_sorted(
        self,
        env: FactoryLayoutEnv,
        gid: Union[int, str],
        pool: List[Tuple[Union[int, str], float, float, int]],
    ) -> List[Tuple[Union[int, str], float, float, int]]:
        if not pool:
            return []
        scored = [(env.estimate_delta_obj(gid=gid, x=c[1], y=c[2], rot=c[3]), c) for c in pool]
        scored.sort(key=lambda item: item[0])
        return [c for _, c in scored]

    def _random_take(
        self, pool: List[Tuple[Union[int, str], float, float, int]], count: int
    ) -> List[Tuple[Union[int, str], float, float, int]]:
        if count <= 0 or not pool:
            return []
        if len(pool) <= count:
            return list(pool)
        return self._rng.sample(pool, count)

    def _source_high(
        self, env: FactoryLayoutEnv, gid: Union[int, str], count: int
    ) -> List[Tuple[Union[int, str], float, float, int]]:
        group = env.groups[gid]
        rotations = (0, 90) if group.rotatable else (0,)
        step = max(int(self.scan_step), 1)
        max_scan = 50000

        def _jump_x(x: float, y: float, w: float, h: float, direction: int) -> Optional[float]:
            y_min = y - h / 2.0
            y_max = y + h / 2.0
            best: Optional[float] = None
            for pid in env.placed:
                px, py, prot = env.positions[pid]
                pw, ph = env.rotated_size(env.groups[pid], prot)
                left, right, bottom, top = env.rect_from_center(px, py, pw, ph)
                if top <= y_min or bottom >= y_max:
                    continue
                if direction > 0 and left >= x:
                    cand = right + w / 2.0
                    if best is None or cand < best:
                        best = cand
                if direction < 0 and right <= x:
                    cand = left - w / 2.0
                    if best is None or cand > best:
                        best = cand
            return best

        results: List[Tuple[Union[int, str], float, float, int]] = []
        bounds = (0.0, float(env.grid_width), 0.0, float(env.grid_height))

        for rot in rotations:
            w, h = env.rotated_size(group, rot)
            min_x, max_x, min_y, max_y = bounds
            x_start = min_x + w / 2.0
            y_start = min_y + h / 2.0
            x = x_start
            y = y_start

            for _ in range(max_scan):
                results.append((gid, float(x), float(y), int(rot)))

                if x + w / 2.0 < max_x:
                    jump = _jump_x(x, y, w, h, 1)
                    if jump is None:
                        x += step
                    else:
                        x = max(jump, x + step)
                else:
                    y += step
                    if y + h / 2.0 > max_y:
                        break
                    x = x_start

                if len(results) > count * 10:
                    break

        return results

    def _source_near(
        self, env: FactoryLayoutEnv, gid: Union[int, str], count: int
    ) -> List[Tuple[Union[int, str], float, float, int]]:
        if not env.placed:
            return []
        group = env.groups[gid]
        rotations = (0, 90) if group.rotatable else (0,)
        candidates: List[Tuple[Union[int, str], float, float, int]] = []

        for rot in rotations:
            w, h = env.rotated_size(group, rot)
            for pid in env.placed:
                px, py, prot = env.positions[pid]
                pw, ph = env.rotated_size(env.groups[pid], prot)
                lB, rB, bB, tB = env.rect_from_center(px, py, pw, ph)

                x_events = [lB - w / 2.0, lB + w / 2.0, rB - w / 2.0, rB + w / 2.0]
                y_events = [bB - h / 2.0, bB + h / 2.0, tB - h / 2.0, tB + h / 2.0]

                for x in x_events:
                    for y in y_events:
                        candidates.append((gid, float(x), float(y), int(rot)))
        return candidates

    def _source_coarse(
        self, env: FactoryLayoutEnv, gid: Union[int, str], count: int
    ) -> List[Tuple[Union[int, str], float, float, int]]:
        step = max(self.scan_step * 3, 1.0)
        group = env.groups[gid]
        w, h = env.rotated_size(group, 0)
        xs = self._scan_axis(float(env.grid_width), float(w), float(step))
        ys = self._scan_axis(float(env.grid_height), float(h), float(step))
        candidates: List[Tuple[Union[int, str], float, float, int]] = []
        for x in xs:
            for y in ys:
                candidates.append((gid, float(x), float(y), 0))
        return candidates

    def _source_random(
        self, env: FactoryLayoutEnv, gid: Union[int, str], count: int
    ) -> List[Tuple[Union[int, str], float, float, int]]:
        if count <= 0:
            return []
        group = env.groups[gid]
        candidates: List[Tuple[Union[int, str], float, float, int]] = []
        for _ in range(count):
            rot = 0 if not group.rotatable else self._rng.choice([0, 90])
            w, h = env.rotated_size(group, rot)
            x = self._rng.uniform(w / 2.0, float(env.grid_width) - w / 2.0)
            y = self._rng.uniform(h / 2.0, float(env.grid_height) - h / 2.0)
            candidates.append((gid, float(x), float(y), int(rot)))
        return candidates

    def _dedup_tagged(
        self,
        candidates: List[Tuple[int, Tuple[Union[int, str], float, float, int]]],
        q: float,
        group: object,
    ) -> List[Tuple[int, Tuple[Union[int, str], float, float, int]]]:
        if q <= 0:
            return candidates
        seen = set()
        unique: List[Tuple[int, Tuple[Union[int, str], float, float, int]]] = []
        for src, c in candidates:
            qx = int(round(float(c[1]) / q))
            qy = int(round(float(c[2]) / q))
            key = (qx, qy, int(c[3]))
            if key not in seen:
                seen.add(key)
                unique.append((src, c))
        return unique

    def _pad_candidates(self, gid: Union[int, str], count: int) -> List[Tuple[Union[int, str], float, float, int]]:
        return [(gid, 0.0, 0.0, 0) for _ in range(count)]

    def _scan_axis(self, limit: float, size: float, step: float) -> List[float]:
        if step <= 0:
            step = 1.0
        start = size / 2.0
        end = limit - size / 2.0
        if end < start:
            return []
        n = max(int((end - start) // step) + 1, 1)
        return [start + i * step for i in range(n)]

