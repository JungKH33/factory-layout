from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import torch

from envs.env import FactoryLayoutEnv, GroupId

from .candidate_set import CandidateSet


class TopKSelector:
    """Top-K candidates selector (ported from legacy `policies/topk_selector.py`).

    IMPORTANT (new engine contract):
    - (x,y) are integer bottom-left coordinates of the rotated AABB footprint.
    - mask must match `env.is_placeable(gid, x_bl, y_bl, rot)` (includes clearance/pad/zone).

    Output:
      - xyrot: int64 [K,3] padded with (0,0,0) (integer bottom-left coordinates)
      - mask:  bool [K] where True means VALID candidate exists
    """

    def __init__(
        self,
        *,
        k: int,
        # Defaults MUST match `policies/topk_selector.py:TopKConfig`
        scan_step: float = 2000.0,
        quant_step: Optional[float] = 10.0,
        p_high: float = 0.1,
        p_near: float = 0.8,
        p_coarse: float = 0.0,
        oversample_factor: int = 2,
        diversity_ratio: float = 0.0,  # parity (unused)
        min_diversity: int = 0,  # parity (unused)
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

        xyrot = torch.zeros((self.k, 3), dtype=torch.long, device=device)
        for i, c in enumerate(candidates[: self.k]):
            _, x_bl, y_bl, rot = c
            xyrot[i, 0] = int(x_bl)
            xyrot[i, 1] = int(y_bl)
            xyrot[i, 2] = int(rot)

        return CandidateSet(xyrot=xyrot, mask=mask, meta={"type": "topk", "k": int(self.k), "gid": gid})

    # ---- candidate generation (BL int coords) ----
    def _cheap_reject_body(
        self,
        env: FactoryLayoutEnv,
        *,
        gid: GroupId,
        x_bl: int,
        y_bl: int,
        rot: int,
        wh_cache: Dict[int, Tuple[int, int]],
    ) -> bool:
        """Return True if candidate is certainly invalid by cheap checks (no false rejects).

        We only do O(1) checks here:
        - Boundary check for the BODY rect (pad/clearance is still validated by env.is_placeable()).
        - Sample a handful of cells inside the body rect (corners + center). If any sampled cell is
          invalid (core invalid or others' clearance), the placement is certainly invalid.

        NOTE: This is an optimization to reduce calls to env.is_placeable(). We still call
        env.is_placeable() for exact validation to avoid false accepts.
        """
        w, h = wh_cache[int(rot)]
        gw = int(env.grid_width)
        gh = int(env.grid_height)

        # Body boundary (half-open): [x_bl, x_bl+w) x [y_bl, y_bl+h)
        if x_bl < 0 or y_bl < 0 or (x_bl + w) > gw or (y_bl + h) > gh:
            return True
        if w <= 0 or h <= 0:
            return True

        invalid = env._invalid  # torch.BoolTensor[H,W]
        clear_invalid = env._clear_invalid  # torch.BoolTensor[H,W]

        x0 = int(x_bl)
        y0 = int(y_bl)
        x1 = x0 + int(w) - 1
        y1 = y0 + int(h) - 1
        xc = (x0 + x1) // 2
        yc = (y0 + y1) // 2

        # 5-point sampling: corners + center (all guaranteed in-bounds by boundary check above)
        pts = ((x0, y0), (x1, y0), (x0, y1), (x1, y1), (xc, yc))
        for px, py in pts:
            if bool(invalid[py, px].item()):
                return True
            if bool(clear_invalid[py, px].item()):
                return True
        return False

    def _generate(
        self, env: FactoryLayoutEnv, next_group_id: GroupId
    ) -> Tuple[List[Tuple[GroupId, int, int, int]], torch.Tensor]:
        device = env.device
        q = self.quant_step if self.quant_step is not None else self.scan_step

        if len(env.placed) == 0:
            return self._generate_initial(env, next_group_id, q)

        n_high, n_near, n_coarse, n_rand = self._quota(self.k)
        group = env.groups[next_group_id]
        rotations = (0, 90) if getattr(group, "rotatable", False) else (0,)
        wh_cache = {int(r): self._wh_int(env, next_group_id, int(r)) for r in rotations}

        raw_tagged: List[Tuple[int, Tuple[GroupId, int, int, int]]] = []
        raw_tagged.extend((0, c) for c in self._source_high(env, next_group_id, n_high * self.oversample_factor))
        raw_tagged.extend((1, c) for c in self._source_near(env, next_group_id, n_near * self.oversample_factor))
        raw_tagged.extend((2, c) for c in self._source_coarse(env, next_group_id, n_coarse * self.oversample_factor))
        raw_tagged.extend((3, c) for c in self._source_random(env, next_group_id, n_rand * self.oversample_factor))

        unique_tagged = self._dedup_tagged(raw_tagged, q, group)
        valid_tagged = [
            (src, c)
            for src, c in unique_tagged
            if (not self._cheap_reject_body(env, gid=next_group_id, x_bl=int(c[1]), y_bl=int(c[2]), rot=int(c[3]), wh_cache=wh_cache))
            and env.is_placeable(next_group_id, float(c[1]), float(c[2]), int(c[3]))
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

    # ---- BL helpers ----
    def _wh_int(self, env: FactoryLayoutEnv, gid: GroupId, rot: int) -> Tuple[int, int]:
        g = env.groups[gid]
        w, h = env.rotated_size(g, int(rot))
        return int(round(float(w))), int(round(float(h)))

    def _clamp_bl(self, env: FactoryLayoutEnv, x_bl: int, y_bl: int, w: int, h: int) -> Tuple[int, int]:
        max_x = int(env.grid_width) - int(w)
        max_y = int(env.grid_height) - int(h)
        if max_x < 0 or max_y < 0:
            return 0, 0
        x2 = max(0, min(int(x_bl), max_x))
        y2 = max(0, min(int(y_bl), max_y))
        return int(x2), int(y2)

    def _generate_initial(
        self, env: FactoryLayoutEnv, gid: GroupId, quant_step: float
    ) -> Tuple[List[Tuple[GroupId, int, int, int]], torch.Tensor]:
        device = env.device
        total_k = self.k * self.oversample_factor
        n_strat_target = round(total_k * 0.9)
        n_rand = total_k - n_strat_target

        raw_tagged: List[Tuple[int, Tuple[GroupId, int, int, int]]] = []
        raw_tagged.extend((0, c) for c in self._source_stratified(env, gid, n_strat_target))
        raw_tagged.extend((1, c) for c in self._source_random(env, gid, n_rand))

        group = env.groups[gid]
        rotations = (0, 90) if getattr(group, "rotatable", False) else (0,)
        wh_cache = {int(r): self._wh_int(env, gid, int(r)) for r in rotations}
        unique_tagged = self._dedup_tagged(raw_tagged, quant_step, group)
        valid_candidates = [
            c
            for _, c in unique_tagged
            if (not self._cheap_reject_body(env, gid=gid, x_bl=int(c[1]), y_bl=int(c[2]), rot=int(c[3]), wh_cache=wh_cache))
            and env.is_placeable(gid, float(c[1]), float(c[2]), int(c[3]))
        ]

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
            w, h = self._wh_int(env, gid, int(rot))
            if int(env.grid_width) - w < 0 or int(env.grid_height) - h < 0:
                continue
            for i in range(nx):
                for j in range(ny):
                    x_bl = int(round(i * dx))
                    y_bl = int(round(j * dy))
                    x_bl, y_bl = self._clamp_bl(env, x_bl, y_bl, w, h)
                    candidates.append((gid, int(x_bl), int(y_bl), int(rot)))
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
        step = max(int(round(self.scan_step)), 1)
        max_scan = 50000

        def _jump_x_bl(x_bl: int, y_bl: int, w: int, h: int, direction: int) -> Optional[int]:
            y0 = int(y_bl)
            y1 = int(y_bl) + int(h)
            best: Optional[int] = None
            for pid in env.placed:
                px0, py0, px1, py1 = env.placed_body_rect_bl(pid)
                if py1 <= y0 or py0 >= y1:
                    continue
                if direction > 0 and px0 >= x_bl:
                    cand = int(px1)
                    if best is None or cand < best:
                        best = cand
                if direction < 0 and px1 <= x_bl:
                    cand = int(px0) - int(w)
                    if best is None or cand > best:
                        best = cand
            return best

        results: List[Tuple[GroupId, int, int, int]] = []
        for rot in rotations:
            w, h = self._wh_int(env, gid, int(rot))
            if int(env.grid_width) - w < 0 or int(env.grid_height) - h < 0:
                continue
            x_bl = 0
            y_bl = 0
            for _ in range(max_scan):
                x_bl, y_bl = self._clamp_bl(env, x_bl, y_bl, w, h)
                results.append((gid, int(x_bl), int(y_bl), int(rot)))

                if x_bl < (int(env.grid_width) - w):
                    jump = _jump_x_bl(x_bl, y_bl, w, h, 1)
                    if jump is None:
                        x_bl += step
                    else:
                        x_bl = max(int(jump), x_bl + step)
                else:
                    y_bl += step
                    if y_bl > (int(env.grid_height) - h):
                        break
                    x_bl = 0

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
            w, h = self._wh_int(env, gid, int(rot))
            if int(env.grid_width) - w < 0 or int(env.grid_height) - h < 0:
                continue
            for pid in env.placed:
                px0, py0, px1, py1 = env.placed_body_rect_bl(pid)
                x_events = [int(px0) - w, int(px0), int(px1) - w, int(px1)]
                y_events = [int(py0) - h, int(py0), int(py1) - h, int(py1)]
                for x_bl in x_events:
                    for y_bl in y_events:
                        x2, y2 = self._clamp_bl(env, int(x_bl), int(y_bl), w, h)
                        candidates.append((gid, int(x2), int(y2), int(rot)))
        return candidates

    def _source_coarse(self, env: FactoryLayoutEnv, gid: GroupId, count: int) -> List[Tuple[GroupId, int, int, int]]:
        step = max(int(round(float(self.scan_step * 3))), 1)
        w, h = self._wh_int(env, gid, 0)
        xs = self._scan_axis_bl(int(env.grid_width), w, step)
        ys = self._scan_axis_bl(int(env.grid_height), h, step)
        candidates: List[Tuple[GroupId, int, int, int]] = []
        for x_bl in xs:
            for y_bl in ys:
                candidates.append((gid, int(x_bl), int(y_bl), 0))
        return candidates

    def _source_random(self, env: FactoryLayoutEnv, gid: GroupId, count: int) -> List[Tuple[GroupId, int, int, int]]:
        if count <= 0:
            return []
        group = env.groups[gid]
        candidates: List[Tuple[GroupId, int, int, int]] = []
        for _ in range(count):
            rot = 0 if not group.rotatable else self._rng.choice([0, 90])
            w, h = self._wh_int(env, gid, int(rot))
            if int(env.grid_width) - w < 0 or int(env.grid_height) - h < 0:
                continue
            x_bl = int(round(self._rng.uniform(0.0, float(int(env.grid_width) - w))))
            y_bl = int(round(self._rng.uniform(0.0, float(int(env.grid_height) - h))))
            x_bl, y_bl = self._clamp_bl(env, x_bl, y_bl, w, h)
            candidates.append((gid, int(x_bl), int(y_bl), int(rot)))
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
        unique: List[Tuple[int, Tuple[GroupId, int, int, int]]] = []
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

    def _scan_axis_bl(self, limit: int, size: int, step: int) -> List[int]:
        if step <= 0:
            step = 1
        end = int(limit) - int(size)
        if end < 0:
            return []
        return list(range(0, end + 1, int(step)))


if __name__ == "__main__":
    import time

    from envs.json_loader import load_env
    from envs.visualizer import plot_layout

    ENV_JSON = "env_configs/basic_01.json"
    K = 50
    SCAN_STEP = 10.0
    QUANT_STEP = 10.0

    device = torch.device("cpu")

    loaded = load_env(ENV_JSON, device=device)
    env = loaded.env
    env.log = False
    _obs, _info = env.reset(options=loaded.reset_kwargs)

    selector = TopKSelector(k=K, scan_step=SCAN_STEP, quant_step=QUANT_STEP, random_seed=0)
    t0 = time.perf_counter()
    candidates0 = selector.build(env)
    dt0_ms = (time.perf_counter() - t0) * 1000.0

    print("[actionspace.topk demo]")
    print(" env=", ENV_JSON, "device=", device, "next_gid=", (env.remaining[0] if env.remaining else None))
    print(" k=", K, "scan_step=", SCAN_STEP, "quant_step=", QUANT_STEP)
    print(" [case A] empty layout (placed=0)")
    print("  N=", int(candidates0.mask.shape[0]), "valid=", int(candidates0.mask.sum().item()))
    print("  xyrot.dtype=", candidates0.xyrot.dtype, "mask.dtype=", candidates0.mask.dtype)
    if int(candidates0.mask.sum().item()) > 0:
        first_valid = int(torch.where(candidates0.mask)[0][0].item())
        print("  first_valid_idx=", first_valid, "first_valid_xyrot=", candidates0.xyrot[first_valid].tolist())
    print(f"  build_ms={dt0_ms:.3f}")
    plot_layout(env, candidate_set=candidates0)

    # ---- case B: hand-crafted example (NOT using TopK for placement) ----
    # We reset the env with a fixed set of initial placements (bottom-left integer coords).
    # NOTE: reset() will validate feasibility and raise ValueError if any pose is invalid.
    fixed_initial_positions = {
        # (x_bl, y_bl, rot)
        "A": (170, 20, 0),   # A: 160x80
        "B": (170, 120, 0),  # B: 120x120 (not rotatable)
        "D": (330, 120, 0),  # D: 80x80
    }
    opts_b = dict(loaded.reset_kwargs)
    opts_b["initial_positions"] = fixed_initial_positions
    _obs_b, _info_b = env.reset(options=opts_b)

    t1 = time.perf_counter()
    candidates1 = selector.build(env) if env.remaining else candidates0
    dt1_ms = (time.perf_counter() - t1) * 1000.0
    print(" [case B] hand-crafted placements via reset(initial_positions=...)")
    print("  initial_positions=", fixed_initial_positions)
    print("  placed=", len(env.placed), "remaining=", len(env.remaining), "next_gid=", (env.remaining[0] if env.remaining else None))
    print("  N=", int(candidates1.mask.shape[0]), "valid=", int(candidates1.mask.sum().item()))
    if int(candidates1.mask.sum().item()) > 0:
        first_valid = int(torch.where(candidates1.mask)[0][0].item())
        print("  first_valid_idx=", first_valid, "first_valid_xyrot=", candidates1.xyrot[first_valid].tolist())
    print(f"  build_ms={dt1_ms:.3f}")
    plot_layout(env, candidate_set=candidates1)
