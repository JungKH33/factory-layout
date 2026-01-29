from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from envs.env_old import Candidate, FactoryLayoutEnvOld, FacilityGroup

Point = Tuple[float, float]


@dataclass(frozen=True)
class TopKConfig:
    k: int
    scan_step: float = 2000.0
    quant_step: Optional[float] = None
    p_high: float = 0.55
    p_near: float = 0.25
    p_coarse: float = 0.12
    oversample_factor: int = 4
    diversity_ratio: float = 0.15
    min_diversity: int = 10
    random_seed: Optional[int] = None


class TopKSelector:
    """Top-K candidate generator aligned to the current env state."""

    def __init__(self, config: TopKConfig):
        """Initialize selector with quotas and sampling configuration."""
        self.config = config
        self._rng = random.Random(config.random_seed)

    def get_state(self) -> Dict[str, object]:
        """Return RNG state for deterministic replay."""
        return {
            "rng_state": self._rng.getstate(),
        }

    def set_state(self, state: Dict[str, object]) -> None:
        """Restore RNG state."""
        if isinstance(state, dict) and "rng_state" in state:
            self._rng.setstate(state["rng_state"])

    def generate(self, env: FactoryLayoutEnvOld, next_group_id: Union[int, str]) -> Tuple[List[Candidate], np.ndarray]:
        """Generate K candidates and a mask for the next group."""
        quant_step = self.config.quant_step if self.config.quant_step is not None else self.config.scan_step

        if len(env.placed) == 0:
            return self._generate_initial(env, next_group_id, quant_step)

        n_high, n_near, n_coarse, n_rand = self._quota(self.config.k)
        group = env.groups[next_group_id]

        raw_tagged: List[Tuple[int, Candidate]] = []
        raw_tagged.extend((0, c) for c in self._source_high(env, next_group_id, n_high * self.config.oversample_factor))
        raw_tagged.extend((1, c) for c in self._source_near(env, next_group_id, n_near * self.config.oversample_factor))
        raw_tagged.extend((2, c) for c in self._source_coarse(env, next_group_id, n_coarse * self.config.oversample_factor))
        raw_tagged.extend((3, c) for c in self._source_random(env, next_group_id, n_rand * self.config.oversample_factor))

        unique_tagged = self._dedup_tagged(raw_tagged, quant_step, group)
        valid_tagged = [(src, c) for src, c in unique_tagged if env.is_placeable(next_group_id, c.x, c.y, c.rot)]

        pools: Dict[int, List[Candidate]] = {0: [], 1: [], 2: [], 3: []}
        for src, c in valid_tagged:
            pools[src].append(c)

        final: List[Candidate] = []
        final.extend(self._score_sorted(env, next_group_id, pools[0])[:n_high])
        final.extend(self._score_sorted(env, next_group_id, pools[1])[:n_near])
        final.extend(self._random_take(pools[2], n_coarse))
        final.extend(self._random_take(pools[3], n_rand))

        final = final[: self.config.k]
        mask = np.zeros((self.config.k,), dtype=np.int8)
        mask[: len(final)] = 1
        if len(final) < self.config.k:
            final.extend(self._pad_candidates(next_group_id, self.config.k - len(final)))

        return final, mask

    def _generate_initial(
        self,
        env: FactoryLayoutEnvOld,
        gid: Union[int, str],
        quant_step: float,
    ) -> Tuple[List[Candidate], np.ndarray]:
        """Generate candidates for empty layouts using stratified grid (90%) + random (10%)."""
        group = env.groups[gid]
        total_k = self.config.k * self.config.oversample_factor
        n_strat_target = round(total_k * 0.9)
        n_rand = total_k - n_strat_target

        raw_tagged: List[Tuple[int, Candidate]] = []
        raw_tagged.extend((0, c) for c in self._source_stratified(env, gid, n_strat_target))
        raw_tagged.extend((1, c) for c in self._source_random(env, gid, n_rand))

        unique_tagged = self._dedup_tagged(raw_tagged, quant_step, group)
        valid_candidates = [c for _, c in unique_tagged if env.is_placeable(gid, c.x, c.y, c.rot)]

        final = valid_candidates[: self.config.k]
        mask = np.zeros((self.config.k,), dtype=np.int8)
        mask[: len(final)] = 1
        if len(final) < self.config.k:
            final.extend(self._pad_candidates(gid, self.config.k - len(final)))
        return final, mask

    def _source_stratified(self, env: FactoryLayoutEnvOld, gid: Union[int, str], count: int) -> List[Candidate]:
        if count <= 0:
            return []
        group = env.groups[gid]
        rotations = (0, 90) if group.rotatable else (0,)

        count_per_rot = max(1, count // len(rotations))
        aspect_ratio = env.grid_width / env.grid_height
        nx = max(1, round(math.sqrt(count_per_rot * aspect_ratio)))
        ny = max(1, round(count_per_rot / nx))

        dx = env.grid_width / nx
        dy = env.grid_height / ny

        candidates: List[Candidate] = []
        for rot in rotations:
            for i in range(nx):
                for j in range(ny):
                    x = (i + 0.5) * dx
                    y = (j + 0.5) * dy
                    candidates.append(Candidate(gid, x, y, rot))
        return candidates

    def _quota(self, k: int) -> Tuple[int, int, int, int]:
        n_high = round(k * self.config.p_high)
        n_near = round(k * self.config.p_near)
        n_coarse = round(k * self.config.p_coarse)
        n_rand = k - (n_high + n_near + n_coarse)
        if n_rand < 0:
            n_rand = 0
        return n_high, n_near, n_coarse, n_rand

    def _score_sorted(self, env: FactoryLayoutEnvOld, gid: Union[int, str], pool: List[Candidate]) -> List[Candidate]:
        if not pool:
            return []
        scored = [(env.estimate_delta_obj(gid, c.x, c.y, c.rot), c) for c in pool]
        scored.sort(key=lambda item: item[0])
        return [c for _, c in scored]

    def _random_take(self, pool: List[Candidate], count: int) -> List[Candidate]:
        if count <= 0 or not pool:
            return []
        if len(pool) <= count:
            return list(pool)
        return self._rng.sample(pool, count)

    def _source_high(self, env: FactoryLayoutEnvOld, gid: Union[int, str], count: int) -> List[Candidate]:
        group = env.groups[gid]
        rotations = (0, 90) if group.rotatable else (0,)
        step = max(int(self.config.scan_step), 1)
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

        results: List[Candidate] = []
        bounds = (0.0, float(env.grid_width), 0.0, float(env.grid_height))

        for rot in rotations:
            w, h = env.rotated_size(group, rot)
            min_x, max_x, min_y, max_y = bounds
            x_start = min_x + w / 2.0
            y_start = min_y + h / 2.0
            x = x_start
            y = y_start

            for _ in range(max_scan):
                results.append(Candidate(gid, x, y, rot))

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

    def _source_near(self, env: FactoryLayoutEnvOld, gid: Union[int, str], count: int) -> List[Candidate]:
        if not env.placed:
            return []
        group = env.groups[gid]
        rotations = (0, 90) if group.rotatable else (0,)
        candidates: List[Candidate] = []

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
                        candidates.append(Candidate(gid, x, y, rot))
        return candidates

    def _source_coarse(self, env: FactoryLayoutEnvOld, gid: Union[int, str], count: int) -> List[Candidate]:
        step = max(self.config.scan_step * 3, 1.0)
        group = env.groups[gid]
        w, h = env.rotated_size(group, 0)
        xs = self._scan_axis(env.grid_width, w, step)
        ys = self._scan_axis(env.grid_height, h, step)
        candidates: List[Candidate] = []
        for x in xs:
            for y in ys:
                candidates.append(Candidate(gid, x, y, 0))
        return candidates

    def _source_random(self, env: FactoryLayoutEnvOld, gid: Union[int, str], count: int) -> List[Candidate]:
        if count <= 0:
            return []
        group = env.groups[gid]
        candidates: List[Candidate] = []
        for _ in range(count):
            rot = 0 if not group.rotatable else self._rng.choice([0, 90])
            w, h = env.rotated_size(group, rot)
            x = self._rng.uniform(w / 2.0, env.grid_width - w / 2.0)
            y = self._rng.uniform(h / 2.0, env.grid_height - h / 2.0)
            candidates.append(Candidate(gid, x, y, rot))
        return candidates

    def _dedup_tagged(self, candidates: List[Tuple[int, Candidate]], q: float, group: FacilityGroup) -> List[Tuple[int, Candidate]]:
        if q <= 0:
            return candidates
        seen = set()
        unique = []
        for src, c in candidates:
            qx = int(round(c.x / q))
            qy = int(round(c.y / q))
            key = (qx, qy, c.rot)
            if key not in seen:
                seen.add(key)
                unique.append((src, c))
        return unique

    def _pad_candidates(self, gid: Union[int, str], count: int) -> List[Candidate]:
        return [Candidate(gid, 0.0, 0.0, 0) for _ in range(count)]

    def _scan_axis(self, limit: float, size: float, step: float) -> List[float]:
        if step <= 0:
            step = 1.0
        start = size / 2.0
        end = limit - size / 2.0
        if end < start:
            return []
        n = max(int((end - start) // step) + 1, 1)
        return [start + i * step for i in range(n)]

