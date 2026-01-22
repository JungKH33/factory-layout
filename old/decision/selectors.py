from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import torch
import torch.nn.functional as F

from envs.env import FactoryLayoutEnv, GroupId
from .candidate_set import CandidateSet


class CandidateSelector(Protocol):
    """Build candidates for the *current* env state."""

    def build(self, env: FactoryLayoutEnv) -> CandidateSet: ...

    def get_state(self) -> object:  # RNG etc.
        return None

    def set_state(self, state: object) -> None:
        return None


@dataclass(frozen=True)
class CoarseSelector:
    """Coarse-grid candidates: N=G*G with deterministic (x,y,rot)."""

    coarse_grid: int
    rot: int = 0

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
        cx = (jj * cell_w).to(torch.float32) + (cell_w / 2.0)
        cy = (ii * cell_h).to(torch.float32) + (cell_h / 2.0)

        xyrot = torch.zeros((g * g, 3), dtype=torch.float32, device=device)
        xyrot[:, 0] = cx.reshape(-1)
        xyrot[:, 1] = cy.reshape(-1)
        xyrot[:, 2] = float(int(self.rot))
        return xyrot

    def _mask(self, env: FactoryLayoutEnv) -> torch.Tensor:
        # NOTE: This is a direct copy of `CoarseWrapperEnv.create_mask` logic.
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


class TopKSelector:
    """Top-K candidates selector (ported from `TopKWrapperEnv`).

    Output:
      - xyrot: float32 [K,3] padded with (0,0,0)
      - mask:  bool [K] where True means VALID candidate exists
    """

    def __init__(
        self,
        *,
        k: int,
        # Defaults MUST match `policies/topk_selector.py:TopKConfig`
        scan_step: float = 2000.0,
        quant_step: Optional[float] = None,
        p_high: float = 0.55,
        p_near: float = 0.25,
        p_coarse: float = 0.12,
        oversample_factor: int = 4,
        diversity_ratio: float = 0.15,  # kept for parity (currently unused)
        min_diversity: int = 10,  # kept for parity (currently unused)
        random_seed: Optional[int] = None,
    ):
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

    def get_state(self) -> object:
        # Match `policies/topk_selector.py` shape: {"rng_state": ...}
        return {"rng_state": self._rng.getstate()}

    def set_state(self, state: object) -> None:
        if isinstance(state, dict) and "rng_state" in state:
            self._rng.setstate(state["rng_state"])  # type: ignore[arg-type]

    def build(self, env: FactoryLayoutEnv) -> CandidateSet:
        device = env.device
        if not env.remaining:
            xyrot = torch.zeros((self.k, 3), dtype=torch.long, device=device)
            mask = torch.zeros((self.k,), dtype=torch.bool, device=device)
            return CandidateSet(xyrot=xyrot, mask=mask, meta={"type": "topk", "k": int(self.k), "empty": True})

        gid = env.remaining[0]
        candidates, mask = self._generate(env, gid)

        # IMPORTANT: TopK coordinates must be integer center positions (user requirement).
        xyrot = torch.zeros((self.k, 3), dtype=torch.long, device=device)
        for i, c in enumerate(candidates[: self.k]):
            _, x, y, rot = c
            xyrot[i, 0] = int(x)
            xyrot[i, 1] = int(y)
            xyrot[i, 2] = int(rot)

        return CandidateSet(xyrot=xyrot, mask=mask, meta={"type": "topk", "k": int(self.k), "gid": gid})

    # ---- TopK transplant (adapted from `TopKWrapperEnv._generate`) ----
    def _generate(
        self, env: FactoryLayoutEnv, next_group_id: GroupId
    ) -> Tuple[List[Tuple[GroupId, int, int, int]], torch.Tensor]:
        device = env.device
        q = self.quant_step if self.quant_step is not None else self.scan_step

        if len(env.placed) == 0:
            return self._generate_initial(env, next_group_id, q)

        n_high, n_near, n_coarse, n_rand = self._quota(self.k)
        group = env.groups[next_group_id]

        raw_tagged: List[Tuple[int, Tuple[GroupId, int, int, int]]] = []
        raw_tagged.extend((0, c) for c in self._source_high(env, next_group_id, n_high * self.oversample_factor))
        raw_tagged.extend((1, c) for c in self._source_near(env, next_group_id, n_near * self.oversample_factor))
        raw_tagged.extend((2, c) for c in self._source_coarse(env, next_group_id, n_coarse * self.oversample_factor))
        raw_tagged.extend((3, c) for c in self._source_random(env, next_group_id, n_rand * self.oversample_factor))

        unique_tagged = self._dedup_tagged(raw_tagged, q, group)
        valid_tagged = [
            (src, c) for src, c in unique_tagged if env.is_placeable(next_group_id, float(c[1]), float(c[2]), int(c[3]))
        ]

        pools: Dict[int, List[Tuple[GroupId, int, int, int]]] = {0: [], 1: [], 2: [], 3: []}
        for src, c in valid_tagged:
            pools[src].append(c)

        final: List[Tuple[GroupId, int, int, int]] = []
        final.extend(self._score_sorted(env, next_group_id, pools[0])[:n_high])
        final.extend(self._score_sorted(env, next_group_id, pools[1])[:n_near])
        final.extend(self._random_take(pools[2], n_coarse))
        final.extend(self._random_take(pools[3], n_rand))

        final = final[: self.k]
        mask = torch.zeros((self.k,), dtype=torch.bool, device=device)
        if final:
            mask[: len(final)] = True
        if len(final) < self.k:
            final.extend(self._pad_candidates(next_group_id, self.k - len(final)))

        return final, mask

    def _generate_initial(
        self, env: FactoryLayoutEnv, gid: GroupId, quant_step: float
    ) -> Tuple[List[Tuple[GroupId, int, int, int]], torch.Tensor]:
        device = env.device
        group = env.groups[gid]
        total_k = self.k * self.oversample_factor
        n_strat_target = round(total_k * 0.9)
        n_rand = total_k - n_strat_target

        raw_tagged: List[Tuple[int, Tuple[GroupId, int, int, int]]] = []
        raw_tagged.extend((0, c) for c in self._source_stratified(env, gid, n_strat_target))
        raw_tagged.extend((1, c) for c in self._source_random(env, gid, n_rand))

        unique_tagged = self._dedup_tagged(raw_tagged, quant_step, group)
        valid_candidates = [c for _, c in unique_tagged if env.is_placeable(gid, float(c[1]), float(c[2]), int(c[3]))]

        final = valid_candidates[: self.k]
        mask = torch.zeros((self.k,), dtype=torch.bool, device=device)
        if final:
            mask[: len(final)] = True
        if len(final) < self.k:
            final.extend(self._pad_candidates(gid, self.k - len(final)))
        return final, mask

    def _source_stratified(self, env: FactoryLayoutEnv, gid: GroupId, count: int) -> List[Tuple[GroupId, int, int, int]]:
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

        candidates: List[Tuple[GroupId, int, int, int]] = []
        for rot in rotations:
            for i in range(nx):
                for j in range(ny):
                    x = int(round((i + 0.5) * dx))
                    y = int(round((j + 0.5) * dy))
                    candidates.append((gid, int(x), int(y), int(rot)))
        return candidates

    def _quota(self, k: int) -> Tuple[int, int, int, int]:
        n_high = round(k * self.p_high)
        n_near = round(k * self.p_near)
        n_coarse = round(k * self.p_coarse)
        n_rand = k - (n_high + n_near + n_coarse)
        if n_rand < 0:
            n_rand = 0
        return int(n_high), int(n_near), int(n_coarse), int(n_rand)

    def _score_sorted(self, env: FactoryLayoutEnv, gid: GroupId, pool: List[Tuple[GroupId, int, int, int]]) -> List[Tuple[GroupId, int, int, int]]:
        if not pool:
            return []
        scored = [(env.estimate_delta_obj(gid=gid, x=c[1], y=c[2], rot=c[3]), c) for c in pool]
        scored.sort(key=lambda item: item[0])
        return [c for _, c in scored]

    def _random_take(self, pool: List[Tuple[GroupId, int, int, int]], count: int) -> List[Tuple[GroupId, int, int, int]]:
        if count <= 0 or not pool:
            return []
        if len(pool) <= count:
            return list(pool)
        return self._rng.sample(pool, count)

    def _source_high(self, env: FactoryLayoutEnv, gid: GroupId, count: int) -> List[Tuple[GroupId, int, int, int]]:
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

        results: List[Tuple[GroupId, int, int, int]] = []
        bounds = (0.0, float(env.grid_width), 0.0, float(env.grid_height))

        for rot in rotations:
            w, h = env.rotated_size(group, rot)
            min_x, max_x, min_y, max_y = bounds
            x_start = min_x + w / 2.0
            y_start = min_y + h / 2.0
            x = x_start
            y = y_start

            for _ in range(max_scan):
                results.append((gid, int(round(float(x))), int(round(float(y))), int(rot)))

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

    def _source_near(self, env: FactoryLayoutEnv, gid: GroupId, count: int) -> List[Tuple[GroupId, int, int, int]]:
        if not env.placed:
            return []
        group = env.groups[gid]
        rotations = (0, 90) if group.rotatable else (0,)
        candidates: List[Tuple[GroupId, int, int, int]] = []

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
                        candidates.append((gid, int(round(float(x))), int(round(float(y))), int(rot)))
        return candidates

    def _source_coarse(self, env: FactoryLayoutEnv, gid: GroupId, count: int) -> List[Tuple[GroupId, int, int, int]]:
        step = max(self.scan_step * 3, 1.0)
        group = env.groups[gid]
        w, h = env.rotated_size(group, 0)
        xs = self._scan_axis(float(env.grid_width), float(w), float(step))
        ys = self._scan_axis(float(env.grid_height), float(h), float(step))
        candidates: List[Tuple[GroupId, int, int, int]] = []
        for x in xs:
            for y in ys:
                candidates.append((gid, int(round(float(x))), int(round(float(y))), 0))
        return candidates

    def _source_random(self, env: FactoryLayoutEnv, gid: GroupId, count: int) -> List[Tuple[GroupId, int, int, int]]:
        if count <= 0:
            return []
        group = env.groups[gid]
        candidates: List[Tuple[GroupId, int, int, int]] = []
        for _ in range(count):
            rot = 0 if not group.rotatable else self._rng.choice([0, 90])
            w, h = env.rotated_size(group, rot)
            x = self._rng.uniform(w / 2.0, float(env.grid_width) - w / 2.0)
            y = self._rng.uniform(h / 2.0, float(env.grid_height) - h / 2.0)
            candidates.append((gid, int(round(float(x))), int(round(float(y))), int(rot)))
        return candidates

    def _dedup_tagged(
        self,
        candidates: List[Tuple[int, Tuple[GroupId, int, int, int]]],
        q: float,
        group: object,
    ) -> List[Tuple[int, Tuple[GroupId, int, int, int]]]:
        if q <= 0:
            return candidates
        seen = set()
        unique: List[Tuple[int, Tuple[GroupId, float, float, int]]] = []
        for src, c in candidates:
            qx = int(round(float(c[1]) / q))
            qy = int(round(float(c[2]) / q))
            key = (qx, qy, int(c[3]))
            if key not in seen:
                seen.add(key)
                unique.append((src, c))
        return unique

    def _pad_candidates(self, gid: GroupId, count: int) -> List[Tuple[GroupId, int, int, int]]:
        return [(gid, 0, 0, 0) for _ in range(count)]

    def _scan_axis(self, limit: float, size: float, step: float) -> List[float]:
        if step <= 0:
            step = 1.0
        start = size / 2.0
        end = limit - size / 2.0
        if end < start:
            return []
        n = max(int((end - start) // step) + 1, 1)
        return [start + i * step for i in range(n)]


if __name__ == "__main__":
    import time

    from envs.json_loader import load_env

    ENV_JSON = "env_configs/basic_01.json"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cuda")
    loaded = load_env(ENV_JSON, device=device)
    env = loaded.env
    env.log = False
    env.reset(options=loaded.reset_kwargs)

    print("[selectors demo]")
    print(" env=", ENV_JSON, "grid=", (env.grid_width, env.grid_height), "device=", env.device)
    print(" next_gid=", (env.remaining[0] if env.remaining else None))

    # Coarse demo
    t0 = time.perf_counter()
    coarse = CoarseSelector(coarse_grid=32, rot=0).build(env)
    t1 = time.perf_counter()
    print(" CoarseSelector:")
    print("  N=", int(coarse.mask.shape[0]), "valid=", int(coarse.mask.sum().item()), "elapsed_ms=", (t1 - t0) * 1000.0)
    print("  first_xyrot=", coarse.xyrot[0].tolist())

    # TopK demo (use small step/quant for a realistic run; defaults match original but may be too coarse)
    topk = TopKSelector(k=50, scan_step=10.0, quant_step=10.0, random_seed=0)
    t0 = time.perf_counter()
    cand = topk.build(env)
    t1 = time.perf_counter()
    print(" TopKSelector:")
    print("  N=", int(cand.mask.shape[0]), "valid=", int(cand.mask.sum().item()), "elapsed_ms=", (t1 - t0) * 1000.0)
    if int(cand.mask.sum().item()) > 0:
        print("  first_valid_xyrot=", cand.xyrot[int(torch.nonzero(cand.mask)[0].item())].tolist())

