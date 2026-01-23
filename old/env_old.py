from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

Point = Tuple[float, float]
Rect = Tuple[float, float, float, float]  # (left, right, bottom, top)


@dataclass(frozen=True)
class FacilityGroup:
    id: Union[int, str]
    width: float
    height: float
    movable: bool = True
    rotatable: bool = True
    dry_level: float = 0.0
    weight: float = 0.0
    type: Optional[str] = None
    ent_rel: Point = (0.0, 0.0)
    exi_rel: Point = (0.0, 0.0)


class RectMask:
    """Grid mask where True means allowed placement for that cell."""

    def __init__(self, allowed: List[List[bool]]):
        """Create a mask with True for allowed grid cells."""
        self.allowed = allowed
        self.height = len(allowed)
        self.width = len(allowed[0]) if self.height > 0 else 0

    def is_rect_allowed(self, rect: Rect) -> bool:
        """Return True if rect fits entirely within allowed cells."""
        left, right, bottom, top = rect
        min_x = int(math.floor(left))
        max_x = int(math.ceil(right)) - 1
        min_y = int(math.floor(bottom))
        max_y = int(math.ceil(top)) - 1
        if min_x < 0 or min_y < 0 or max_x >= self.width or max_y >= self.height:
            return False
        for y in range(min_y, max_y + 1):
            row = self.allowed[y]
            for x in range(min_x, max_x + 1):
                if not row[x]:
                    return False
        return True

    def is_rect_forbidden(self, rect: Rect) -> bool:
        """Return True if rect intersects any forbidden cell."""
        return not self.is_rect_allowed(rect)


@dataclass(frozen=True)
class Candidate:
    group_id: Union[int, str]
    x: float
    y: float
    rot: int


@dataclass(frozen=True)
class ObservationSpec:
    max_candidates: int
    cand_feat_dim: int
    state_dim: int


class FactoryLayoutEnvOld(gym.Env):
    """Gymnasium environment with candidate-based actions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        grid_width: int,
        grid_height: int,
        grid_size: float,
        groups: Dict[Union[int, str], FacilityGroup],
        group_flow: Dict[Union[int, str], Dict[Union[int, str], float]],
        max_candidates: int,
        forbidden_mask: Optional[RectMask] = None,
        column_mask: Optional[RectMask] = None,
        dry_mask: Optional[RectMask] = None,
        weight_mask: Optional[RectMask] = None,
        exclude_compact_types: Optional[Iterable[str]] = None,
        max_steps: Optional[int] = None,
        reward_scale: float = 100.0,
        penalty_scale: float = 30.0,
        seed: Optional[int] = None,
        log: bool = False,
    ):
        """Initialize env with grid geometry, groups, and masks."""
        super().__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid_size = grid_size
        self.groups = groups
        self.group_flow = group_flow
        self.forbidden_mask = forbidden_mask
        self.column_mask = column_mask
        self.dry_mask = dry_mask
        self.weight_mask = weight_mask
        self.exclude_compact_types = set(exclude_compact_types or [])
        self.positions: Dict[Union[int, str], Tuple[float, float, int]] = {}
        self.placed: set[Union[int, str]] = set()
        self.remaining: List[Union[int, str]] = []
        self.max_candidates = max_candidates
        self._candidates: List[Candidate] = []
        # Current candidate mask (0/1). If an external generator provides a mask, we store it as-is.
        # If not provided, we compute it via is_placeable() for correctness.
        self._cand_mask: np.ndarray = np.zeros((self.max_candidates,), dtype=np.int8)
        self._rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.penalty_scale = float(penalty_scale)
        self._step_count = 0
        self.log = log

        self._obs_spec = ObservationSpec(
            max_candidates=max_candidates,
            cand_feat_dim=11,
            state_dim=4,
        )

        self.action_space = spaces.Discrete(self.max_candidates)
        self.observation_space = spaces.Dict(
            {
                "cand_feat": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self._obs_spec.max_candidates, self._obs_spec.cand_feat_dim),
                    dtype=np.float32,
                ),
                "cand_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self._obs_spec.max_candidates,),
                    dtype=np.int8,
                ),
                "state_summary": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self._obs_spec.state_dim,),
                    dtype=np.float32,
                ),
            }
        )

    def _reset_state(
        self,
        *,
        initial_positions: Optional[Dict[Union[int, str], Tuple[float, float, int]]] = None,
        remaining_order: Optional[List[Union[int, str]]] = None,
    ) -> None:
        """Initialize placement state with optional pre-placed groups."""
        self.positions = dict(initial_positions or {})
        self.placed = set(self.positions.keys())
        self._step_count = 0
        if remaining_order is not None:
            self.remaining = list(remaining_order)
        else:
            self.remaining = [gid for gid in self.groups.keys() if gid not in self.placed]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        initial_positions: Optional[Dict[Union[int, str], Tuple[float, float, int]]] = None,
        remaining_order: Optional[List[Union[int, str]]] = None,
        candidates: Optional[List[Candidate]] = None,
    ):
        """Reset env state and optionally load initial positions and candidates."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state(initial_positions=initial_positions, remaining_order=remaining_order)
        if candidates is not None:
            self.set_candidates(candidates)
        obs = self._build_observation()
        info: Dict[str, Union[bool, str]] = {}
        return obs, info

    def _set_cand_mask(self, mask: Optional[np.ndarray]) -> None:
        """Normalize and set cand_mask to shape (max_candidates,) with dtype int8."""
        self._cand_mask = np.zeros((self.max_candidates,), dtype=np.int8)
        if mask is None:
            return
        m = np.asarray(mask, dtype=np.int8)
        n = min(len(m), self.max_candidates)
        if n > 0:
            self._cand_mask[:n] = m[:n]

    def _remaining_area_ratio(self) -> float:
        """Return remaining area / total area in [0, 1] (best effort)."""
        total_area = 0.0
        for g in self.groups.values():
            total_area += float(g.width) * float(g.height)
        if total_area <= 0:
            return 0.0

        rem_area = 0.0
        for gid in self.remaining:
            g = self.groups.get(gid)
            if g is None:
                continue
            rem_area += float(g.width) * float(g.height)

        ratio = rem_area / total_area
        if ratio < 0.0:
            return 0.0
        if ratio > 1.0:
            return 1.0
        return ratio

    def _failure_penalty(self) -> float:
        """Penalty reward for failed placement or no valid actions (negative)."""
        return -self.penalty_scale * self._remaining_area_ratio()

    def set_candidates(self, candidates: List[Candidate], mask: Optional[np.ndarray] = None) -> None:
        """Set current candidate list for discrete actions and update cand_mask.

        - If mask is provided: store it as-is (no extra validation here).
        - If mask is not provided: compute it via is_placeable() for correctness.
        """
        self._candidates = list(candidates)

        if mask is not None:
            self._set_cand_mask(mask)
            return

        # No mask provided -> compute feasibility mask.
        self._set_cand_mask(None)
        for i, cand in enumerate(self._candidates[: self.max_candidates]):
            if self.is_placeable(cand.group_id, cand.x, cand.y, cand.rot):
                self._cand_mask[i] = 1

    def get_snapshot(self) -> Dict[str, object]:
        """Return a lightweight snapshot for MCTS rollouts."""
        return {
            "positions": dict(self.positions),
            "placed": set(self.placed),
            "remaining": list(self.remaining),
            "candidates": list(self._candidates),
            "cand_mask": self._cand_mask.copy(),
            "step_count": self._step_count,
            "rng_state": self._rng.bit_generator.state,
        }

    def set_snapshot(self, snapshot: Dict[str, object]) -> None:
        """Restore env state from a snapshot."""
        self.positions = dict(snapshot.get("positions", {}))
        self.placed = set(snapshot.get("placed", set()))
        self.remaining = list(snapshot.get("remaining", []))
        self._candidates = list(snapshot.get("candidates", []))
        self._set_cand_mask(snapshot.get("cand_mask"))
        self._step_count = int(snapshot.get("step_count", 0))
        rng_state = snapshot.get("rng_state")
        if rng_state is not None:
            self._rng = np.random.default_rng()
            self._rng.bit_generator.state = rng_state

    def step(self, action: int):
        """Apply selected candidate placement and return Gymnasium step tuple."""
        info: Dict[str, Union[bool, str]] = {}
        self._step_count += 1

        reward = 0.0
        terminated = False
        truncated = False

        # (1) Failure cases: set flags/reward only (single exit)
        if len(self.remaining) > 0 and int(np.sum(self._cand_mask)) == 0:
            info["invalid"] = True
            info["reason"] = "no_valid_actions"
            reward = self._failure_penalty()
            truncated = True

        elif action < 0 or action >= self.max_candidates:
            info["invalid"] = True
            info["reason"] = "action_out_of_range"
            reward = self._failure_penalty()
            truncated = True

        elif self._cand_mask[action] == 0:
            info["invalid"] = True
            info["reason"] = "masked_action"
            reward = self._failure_penalty()
            truncated = True

        else:
            # (2) Normal placement path
            cand = self._candidates[action]
            if not self.is_placeable(cand.group_id, cand.x, cand.y, cand.rot):
                info["invalid"] = True
                info["reason"] = "not_placeable"
                reward = self._failure_penalty()
                truncated = True
            else:
                cost_prev = self.cal_obj()
                self.try_place(cand.group_id, cand.x, cand.y, cand.rot)
                cost_new = self.cal_obj()
                reward = -(cost_new - cost_prev) / self.reward_scale
                info["invalid"] = False
                terminated = len(self.remaining) == 0
                truncated = self.max_steps is not None and self._step_count >= self.max_steps

        obs = self._build_observation()

        # (3) Single logging point
        if self.log and (terminated or truncated):
            cost_now = self.cal_obj()
            reason = info.get("reason", "")
            print(
                f"[env] end: terminated={terminated} truncated={truncated} "
                f"remaining={len(self.remaining)} placed={len(self.placed)} step={self._step_count} "
                f"cost={cost_now:.3f} reason={reason} reward={float(reward):.3f}"
            )

        return obs, reward, terminated, truncated, info

    def _build_observation(self) -> Dict[str, np.ndarray]:
        """Build observation dict with candidate features, mask, and state summary."""
        cand_feat = np.zeros(
            (self._obs_spec.max_candidates, self._obs_spec.cand_feat_dim),
            dtype=np.float32,
        )
        cand_mask = self._cand_mask

        for i, cand in enumerate(self._candidates[: self._obs_spec.max_candidates]):
            if cand_mask[i] == 0:
                continue
            group = self.groups[cand.group_id]
            w, h = self.rotated_size(group, cand.rot)
            rect = self.rect_from_center(cand.x, cand.y, w, h)

            boundary_ok = 1.0 if self._rect_in_bounds(rect) else 0.0
            forbidden_ok = 1.0
            if self.forbidden_mask and self.forbidden_mask.is_rect_forbidden(rect):
                forbidden_ok = 0.0
            column_ok = 1.0
            if self.column_mask and self.column_mask.is_rect_forbidden(rect):
                column_ok = 0.0
            dry_ok = 1.0
            if self.dry_mask:
                if group.dry_level > 0:
                    dry_ok = 1.0 if self.dry_mask.is_rect_allowed(rect) else 0.0
                else:
                    dry_ok = 0.0 if self.dry_mask.is_rect_allowed(rect) else 1.0
            weight_ok = 1.0
            if self.weight_mask and group.weight > 0:
                weight_ok = 1.0 if self.weight_mask.is_rect_allowed(rect) else 0.0

            delta_obj = self.estimate_delta_obj(cand.group_id, cand.x, cand.y, cand.rot)
            cand_feat[i] = np.array(
                [
                    cand.x / self.grid_width,
                    cand.y / self.grid_height,
                    (cand.rot % 360) / 360.0,
                    w / self.grid_width,
                    h / self.grid_height,
                    boundary_ok,
                    forbidden_ok,
                    column_ok,
                    dry_ok,
                    weight_ok,
                    np.tanh(delta_obj),
                ],
                dtype=np.float32,
            )

        state_summary = self._build_state_summary()
        return {
            "cand_feat": cand_feat,
            "cand_mask": cand_mask,
            "state_summary": state_summary,
        }

    def _build_state_summary(self) -> np.ndarray:
        """Return compact state summary for observation."""
        total = max(len(self.groups), 1)
        placed = len(self.placed)
        remaining = len(self.remaining)
        placed_ratio = placed / total
        remaining_ratio = remaining / total
        current_obj = self.cal_obj()
        current_obj_norm = np.tanh(current_obj) if current_obj > 0 else 0.0
        grid_fill = min(placed / total, 1.0)
        return np.array(
            [placed_ratio, remaining_ratio, current_obj_norm, grid_fill],
            dtype=np.float32,
        )

    def _rect_in_bounds(self, rect: Rect) -> bool:
        """Check if rect lies within grid boundaries."""
        left, right, bottom, top = rect
        if left < 0 or right > self.grid_width:
            return False
        if bottom < 0 or top > self.grid_height:
            return False
        return True

    def try_place(self, group_id: Union[int, str], x: float, y: float, rot: int) -> bool:
        """Place a group if feasible; returns True on success."""
        if not self.is_placeable(group_id, x, y, rot):
            return False
        self.positions[group_id] = (x, y, rot)
        self.placed.add(group_id)
        if group_id in self.remaining:
            self.remaining.remove(group_id)
        return True

    def is_placeable(self, group_id: Union[int, str], x: float, y: float, rot: int) -> bool:
        """Check boundary, overlap, and mask constraints for a placement."""
        if group_id not in self.groups:
            return False
        if group_id in self.placed:
            return False
        group = self.groups[group_id]
        if not group.movable:
            return False
        if not group.rotatable:
            rot = 0
        w, h = self.rotated_size(group, rot)
        rect = self.rect_from_center(x, y, w, h)
        if rect[0] < 0 or rect[1] > self.grid_width:
            return False
        if rect[2] < 0 or rect[3] > self.grid_height:
            return False
        for pid in self.placed:
            px, py, prot = self.positions[pid]
            pw, ph = self.rotated_size(self.groups[pid], prot)
            if self.rect_overlap(rect, self.rect_from_center(px, py, pw, ph)):
                return False
        if self.forbidden_mask and self.forbidden_mask.is_rect_forbidden(rect):
            return False
        if self.column_mask and self.column_mask.is_rect_forbidden(rect):
            return False
        if self.dry_mask:
            if group.dry_level > 0:
                if not self.dry_mask.is_rect_allowed(rect):
                    return False
            else:
                if self.dry_mask.is_rect_allowed(rect):
                    return False
        if self.weight_mask and group.weight > 0:
            if not self.weight_mask.is_rect_allowed(rect):
                return False
        return True

    def cal_obj(self) -> float:
        """Compute total cost (flow distance + compactness)."""
        if not self.placed:
            return 0.0
        total_distance = 0.0
        min_x = float("inf")
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")
        enex_cache: Dict[Union[int, str], Tuple[float, float, float, float]] = {}
        for gid in self.placed:
            enex_cache[gid] = self.compute_enex(gid)
        for g1 in self.placed:
            neighbors = self.group_flow.get(g1, {})
            for g2, weight in neighbors.items():
                if g2 not in self.placed:
                    continue
                en_x, en_y, ex_x, ex_y = enex_cache[g1]
                en2_x, en2_y, _, _ = enex_cache[g2]
                total_distance += weight * (abs(ex_x - en2_x) + abs(ex_y - en2_y))
        for gid in self.placed:
            g = self.groups[gid]
            if g.type in self.exclude_compact_types:
                continue
            x, y, rot = self.positions[gid]
            w, h = self.rotated_size(g, rot)
            min_x = min(min_x, x - w / 2.0)
            max_x = max(max_x, x + w / 2.0)
            min_y = min(min_y, y - h / 2.0)
            max_y = max(max_y, y + h / 2.0)
        compactness = 0.0
        if min_x != float("inf"):
            compactness = 0.5 * ((max_x - min_x) + (max_y - min_y))
        return total_distance + compactness

    def estimate_delta_obj(self, group_id: Union[int, str], x: float, y: float, rot: int) -> float:
        """Estimate cost change if placing the group at the given pose."""
        if not self.is_placeable(group_id, x, y, rot):
            return 0.0
        cost_prev = self.cal_obj()
        prev_state = (
            dict(self.positions),
            set(self.placed),
            list(self.remaining),
        )
        self.try_place(group_id, x, y, rot)
        cost_new = self.cal_obj()
        self.positions, self.placed, self.remaining = prev_state
        return (cost_new - cost_prev) / self.reward_scale

    def compute_enex(self, group_id: Union[int, str]) -> Tuple[float, float, float, float]:
        """Compute entry/exit coordinates for a placed group."""
        x, y, rot = self.positions[group_id]
        g = self.groups[group_id]
        ent_x, ent_y = self.rotate_point(g.ent_rel[0], g.ent_rel[1], rot)
        exi_x, exi_y = self.rotate_point(g.exi_rel[0], g.exi_rel[1], rot)
        return (x + ent_x, y + ent_y, x + exi_x, y + exi_y)

    @staticmethod
    def rotate_point(x: float, y: float, rot: int) -> Point:
        """Rotate a point by multiples of 90 degrees."""
        rot = rot % 360
        if rot == 0:
            return (x, y)
        if rot == 90:
            return (y, -x)
        if rot == 180:
            return (-x, -y)
        if rot == 270:
            return (-y, x)
        raise ValueError(f"Unsupported rotation: {rot}")

    @staticmethod
    def rotated_size(group: FacilityGroup, rot: int) -> Tuple[float, float]:
        """Return width/height after rotation."""
        rot = rot % 180
        if rot == 90:
            return (group.height, group.width)
        return (group.width, group.height)

    @staticmethod
    def rect_from_center(x: float, y: float, w: float, h: float) -> Rect:
        """Convert center/size to rectangle bounds."""
        left = x - w / 2.0
        right = x + w / 2.0
        bottom = y - h / 2.0
        top = y + h / 2.0
        return (left, right, bottom, top)

    @staticmethod
    def rect_overlap(a: Rect, b: Rect) -> bool:
        """Return True if rectangles overlap (touching is non-overlap)."""
        left_a, right_a, bottom_a, top_a = a
        left_b, right_b, bottom_b, top_b = b
        if right_a <= left_b or right_b <= left_a:
            return False
        if top_a <= bottom_b or top_b <= bottom_a:
            return False
        return True
