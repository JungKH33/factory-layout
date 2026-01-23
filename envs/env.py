from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import torch
import torch.nn.functional as F


GroupId = Union[int, str]
RectI = Tuple[int, int, int, int]  # (x0,y0,x1,y1) half-open (json-style)


@dataclass
class FacilityGroup:
    id: GroupId
    width: float   # footprint width
    height: float  # footprint height
    movable: bool = True
    rotatable: bool = True
    # Constraint attributes (do NOT conflict with footprint width/height):
    # - facility_height: vertical height for ceiling constraint
    # - facility_weight: weight for weight-zone constraint
    # - facility_dry: dry/air-condition requirement level
    facility_height: float = 0.0
    facility_weight: float = 0.0
    facility_dry: float = 0.0
    # IO relative offsets (CENTER-based local coordinates).
    # These are rotated with the facility (multiples of 90 degrees) and added to the world center.
    ent_rel_x: float = 0.0
    ent_rel_y: float = 0.0
    exi_rel_x: float = 0.0
    exi_rel_y: float = 0.0
    # Clearance (asymmetric): L/R/B/T paddings for placement constraints.
    # NOTE: Clearance is in grid-cell units and MUST be integer-valued.
    facility_clearance_left: int = 0
    facility_clearance_right: int = 0
    facility_clearance_bottom: int = 0
    facility_clearance_top: int = 0


class FactoryLayoutEnv(gym.Env):
    """Tensor-first Gymnasium env for AlphaChip-style coarse-cell placement.

    Key ideas:
    - action is 1D index over coarse grid cells: Discrete(G*G) (decoder-defined)
    - decoding (cell -> center x,y, rot=0) is owned by a decoder object
    - mask is computed by decoder from env's cached invalid_map + next_group size
    - observations are torch.Tensors (torchrl-friendly)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        grid_width: int,
        grid_height: int,
        groups: Dict[GroupId, FacilityGroup],
        group_flow: Optional[Dict[GroupId, Dict[GroupId, float]]] = None,
        # Masks are torch.BoolTensor[H,W] where True means forbidden/invalid.
        forbidden_mask: Optional[torch.Tensor] = None,
        column_mask: Optional[torch.Tensor] = None,
        # --- zone/constraint configs (optional; fully map-based) ---
        # For each constraint, we define a per-cell float map:
        # - map is initialized from env.default_*
        # - areas override map values on their rects
        # Constraint invalidation is map comparison (same shape for all three):
        # - Weight/Height: invalid if map < facility_value
        # - Dry (reverse): invalid if map > facility_value
        default_weight: float = float("inf"),
        default_height: float = float("inf"),
        default_dry: float = -float("inf"),
        weight_areas: Optional[List[Dict[str, Any]]] = None,  # [{"rect":[x0,y0,x1,y1], "value": float}, ...]
        height_areas: Optional[List[Dict[str, Any]]] = None,  # [{"rect":[...], "value": float}, ...]
        dry_areas: Optional[List[Dict[str, Any]]] = None,  # [{"rect":[...], "value": float}, ...]
        device: Optional[torch.device] = None,
        max_steps: Optional[int] = None,
        reward_scale: float = 100.0,
        penalty_scale: float = 30.0,
        log: bool = False,
    ):
        super().__init__()
        self.grid_width = int(grid_width)
        self.grid_height = int(grid_height)
        self.groups = dict(groups)
        self.group_flow = group_flow or {}
        self.forbidden_mask = forbidden_mask.to(torch.bool) if forbidden_mask is not None else None
        self.column_mask = column_mask.to(torch.bool) if column_mask is not None else None
        self.default_weight = float(default_weight)
        self.default_height = float(default_height)
        self.default_dry = float(default_dry)
        self.weight_areas = list(weight_areas or [])
        self.height_areas = list(height_areas or [])
        self.dry_areas = list(dry_areas or [])
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.max_steps = max_steps
        self.reward_scale = float(reward_scale)
        self.penalty_scale = float(penalty_scale)
        self.log = bool(log)

        # Engine does not own action semantics. Wrappers define action_space/obs additions.
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Dict({})

        # --- static graph cache (obs will include these tensors every step) ---
        self.node_ids: List[GroupId] = sorted(self.groups.keys(), key=lambda x: str(x))
        self.gid_to_idx: Dict[GroupId, int] = {gid: i for i, gid in enumerate(self.node_ids)}
        self._edge_index, self._edge_attr = self._build_graph_static()

        # --- node features cache (in-place updates per placement) ---
        # Minimal 8-dim feature (dims will be refined later):
        # [w_norm, h_norm, placed, x_norm, y_norm, rot_norm, 0, 0]
        self._node_feat = torch.zeros((len(self.node_ids), 8), dtype=torch.float32, device=self.device)
        for gid, i in self.gid_to_idx.items():
            gg = self.groups[gid]
            self._node_feat[i, 0] = float(gg.width) / float(self.grid_width)
            self._node_feat[i, 1] = float(gg.height) / float(self.grid_height)

        # --- metadata cache ---
        self._netlist_metadata = torch.zeros((12,), dtype=torch.float32, device=self.device)

        # NOTE: positions store (x_bl, y_bl, rot) where (x_bl,y_bl) is the bottom-left
        # of the rotated AABB footprint, in integer grid-cell coordinates.
        self.positions: Dict[GroupId, Tuple[int, int, int]] = {}
        self.placed: set[GroupId] = set()
        self.remaining: List[GroupId] = []
        self._step_count = 0

        # Cached tensor maps (performance).
        self._static_invalid = self._build_static_invalid()
        self._occ_invalid = torch.zeros((self.grid_height, self.grid_width), dtype=torch.bool, device=self.device)
        # Clearance invalid layer accumulated from already-placed facilities (body padded by their own clearance).
        self._clear_invalid = torch.zeros((self.grid_height, self.grid_width), dtype=torch.bool, device=self.device)
        # next-gid dependent invalid layer (zones / constraints)
        self._zone_invalid = torch.zeros((self.grid_height, self.grid_width), dtype=torch.bool, device=self.device)
        self._invalid = self._static_invalid.clone()

        # Pre-rasterized per-cell value maps (float32) for fast updates.
        self._weight_map = self._build_value_map(self.default_weight, self.weight_areas)
        self._height_map = self._build_value_map(self.default_height, self.height_areas)
        self._dry_map = self._build_value_map(self.default_dry, self.dry_areas)

        # Ensure invalid is consistent even before first reset.
        self._recompute_invalid()

    # ---- rect/zone helpers ----
    def _build_value_map(self, default_value: float, areas: List[Dict[str, Any]]) -> torch.Tensor:
        """Return float32[H,W] value map. Unspecified cells use `default_value`.

        Expected area format: {"rect": (x0,y0,x1,y1), "value": float}
        """
        m = torch.full(
            (self.grid_height, self.grid_width),
            float(default_value),
            dtype=torch.float32,
            device=self.device,
        )
        for a in areas:
            if not isinstance(a, dict):
                continue
            rect = a.get("rect", None)
            v = a.get("value", None)
            if rect is None or v is None:
                continue
            x0, y0, x1, y1 = rect
            x0 = max(0, min(self.grid_width, int(x0)))
            x1 = max(0, min(self.grid_width, int(x1)))
            y0 = max(0, min(self.grid_height, int(y0)))
            y1 = max(0, min(self.grid_height, int(y1)))
            if x1 > x0 and y1 > y0:
                m[y0:y1, x0:x1] = float(v)
        return m

    def _clearance_lrtb(self, g: FacilityGroup, rot: int) -> Tuple[float, float, float, float]:
        """Return (cL,cR,cB,cT) clearance for a group, accounting for rotation.

        Clearance is integer-valued in grid-cell units. Rotation is restricted to multiples of 90
        degrees; 180/270 are supported for correctness even if callers primarily use 0/90.
        """
        cL = self._as_int(g.facility_clearance_left, name="facility_clearance_left")
        cR = self._as_int(g.facility_clearance_right, name="facility_clearance_right")
        cB = self._as_int(g.facility_clearance_bottom, name="facility_clearance_bottom")
        cT = self._as_int(g.facility_clearance_top, name="facility_clearance_top")
        r = self._norm_rot(rot)
        if r == 90:
            # left'  <- bottom, right' <- top, bottom' <- right, top' <- left
            return (cB, cT, cR, cL)
        if r == 180:
            return (cR, cL, cT, cB)
        if r == 270:
            return (cT, cB, cL, cR)
        return (cL, cR, cB, cT)

    def _update_zone_invalid_for_next(self) -> None:
        """Update `_zone_invalid` based on the *next* group (self.remaining[0])."""
        self._zone_invalid.zero_()
        if not self.remaining:
            self._recompute_invalid()
            return

        gid = self.remaining[0]
        g = self.groups.get(gid, None)
        if g is None:
            self._recompute_invalid()
            return

        # Unified map-based invalidation:
        # - Weight/Height: invalid where map < facility_value
        # - Dry (reverse): invalid where map > facility_value
        self._zone_invalid |= (self._weight_map < float(g.facility_weight))
        self._zone_invalid |= (self._height_map < float(g.facility_height))
        self._zone_invalid |= (self._dry_map > float(g.facility_dry))

        self._recompute_invalid()

    def _zone_invalid_for_gid(self, gid: GroupId) -> torch.Tensor:
        """Return zone-invalid map (bool[H,W]) as if `gid` were the next group.

        Used for reset-time ordering heuristics (does not mutate env state).
        """
        z = torch.zeros((self.grid_height, self.grid_width), dtype=torch.bool, device=self.device)
        g = self.groups.get(gid, None)
        if g is None:
            return z

        z |= (self._weight_map < float(g.facility_weight))
        z |= (self._height_map < float(g.facility_height))
        z |= (self._dry_map > float(g.facility_dry))

        return z

    def _placeable_top_left_count(self, *, invalid: torch.Tensor, kw: int, kh: int) -> Tuple[int, int]:
        """Return (placeable_count, total_top_left) for a footprint kw*kh on invalid map."""
        kw = max(1, int(kw))
        kh = max(1, int(kh))
        H, W = int(invalid.shape[0]), int(invalid.shape[1])
        H2 = H - kh + 1
        W2 = W - kw + 1
        if H2 <= 0 or W2 <= 0:
            return 0, 0

        inv_f = invalid.to(dtype=torch.float32).view(1, 1, H, W)
        kernel = torch.ones((1, 1, kh, kw), device=self.device, dtype=inv_f.dtype)
        overlap = F.conv2d(inv_f, kernel, padding=0)
        valid_top_left = (overlap == 0).squeeze(0).squeeze(0)
        placeable = int(valid_top_left.to(torch.int64).sum().item())
        total = int(H2 * W2)

        # Reference-only (NOT used): naive free-cell count ignores footprint connectivity.
        # naive_free_cells = int((~invalid).to(torch.int64).sum().item())
        return placeable, total

    @staticmethod
    def _as_int(v: object, *, name: str) -> int:
        """Coerce integer-valued inputs (int or float with .0) to int, else raise.

        This avoids silent rounding bugs when we commit to integer grid coordinates.
        """
        if isinstance(v, bool):
            raise ValueError(f"{name} must be int-like, got bool")
        if isinstance(v, int):
            return int(v)
        if isinstance(v, float) and v.is_integer():
            return int(v)
        raise ValueError(f"{name} must be int-like (e.g., 3 or 3.0), got {v!r}")

    @staticmethod
    def _norm_rot(rot: int) -> int:
        r = int(rot) % 360
        if r % 90 != 0:
            raise ValueError(f"rot must be a multiple of 90 degrees, got {rot!r}")
        return r

    @classmethod
    def rotated_size(cls, group: FacilityGroup, rot: int) -> Tuple[float, float]:
        """Rotated AABB footprint size (w_r, h_r) in grid units.

        Supports 0/90/180/270; 180 behaves like 0, 270 like 90.
        """
        r = cls._norm_rot(rot)
        if r in (90, 270):
            return (float(group.height), float(group.width))
        return (float(group.width), float(group.height))

    @staticmethod
    def rotate_point(dx: float, dy: float, rot: int) -> Tuple[float, float]:
        """Rotate a local point (dx,dy) by multiples of 90 degrees (counter-clockwise)."""
        r = int(rot) % 360
        if r % 90 != 0:
            raise ValueError(f"rot must be a multiple of 90 degrees, got {rot!r}")
        if r == 0:
            return (float(dx), float(dy))
        if r == 90:
            return (float(dy), -float(dx))
        if r == 180:
            return (-float(dx), -float(dy))
        if r == 270:
            return (-float(dy), float(dx))
        # unreachable due to mod 360
        return (float(dx), float(dy))

    def _recti_hits_invalid(self, rect: RectI, invalid: torch.Tensor) -> bool:
        """Return True if rect intersects any invalid cell in `invalid` (bool[H,W], True=invalid)."""
        x0, y0, x1, y1 = rect
        if x0 < 0 or y0 < 0 or x1 > self.grid_width or y1 > self.grid_height:
            return True
        if x0 >= x1 or y0 >= y1:
            return False
        return bool(torch.any(invalid[y0:y1, x0:x1]).item())

    def _body_rect_from_bl(self, *, gid: GroupId, x_bl: int, y_bl: int, rot: int) -> RectI:
        g = self.groups[gid]
        w, h = self.rotated_size(g, rot)
        kw = self._as_int(w, name="width")
        kh = self._as_int(h, name="height")
        x0 = self._as_int(x_bl, name="x")
        y0 = self._as_int(y_bl, name="y")
        return (x0, y0, x0 + kw, y0 + kh)

    # ---- public rect helpers (BL, integer, half-open) ----
    def body_rect_bl(self, *, gid: GroupId, x_bl: int, y_bl: int, rot: int) -> RectI:
        """Return body rect (x0,y0,x1,y1) for a pose in bottom-left integer coordinates."""
        rot_n = self._norm_rot(int(rot))
        return self._body_rect_from_bl(gid=gid, x_bl=int(x_bl), y_bl=int(y_bl), rot=rot_n)

    def placed_body_rect_bl(self, gid: GroupId) -> RectI:
        """Return placed body's rect (x0,y0,x1,y1) from stored BL pose in `self.positions`."""
        x_bl, y_bl, rot = self.positions[gid]
        return self.body_rect_bl(gid=gid, x_bl=int(x_bl), y_bl=int(y_bl), rot=int(rot))

    @staticmethod
    def _pad_rect_i(rect: RectI, *, cL: int, cR: int, cB: int, cT: int) -> RectI:
        x0, y0, x1, y1 = rect
        return (x0 - int(cL), y0 - int(cB), x1 + int(cR), y1 + int(cT))

    def center_from_bl(self, *, gid: GroupId, x_bl: int, y_bl: int, rot: int) -> Tuple[float, float]:
        """Return (cx,cy) center from bottom-left pose (x_bl,y_bl,rot) of rotated AABB.

        NOTE: Engine stores positions as bottom-left integer coordinates. Center is derived
        only for objective/features/visualization. Keeping this in one place avoids drift.
        """
        g = self.groups[gid]
        w, h = self.rotated_size(g, rot)
        return (float(x_bl) + float(w) / 2.0, float(y_bl) + float(h) / 2.0)

    def pose_center(self, gid: GroupId) -> Tuple[float, float]:
        """Return (cx,cy) center for an already-placed group from `self.positions`."""
        x_bl, y_bl, rot = self.positions[gid]
        return self.center_from_bl(gid=gid, x_bl=int(x_bl), y_bl=int(y_bl), rot=int(rot))

    def compute_enex(self, gid: GroupId) -> Tuple[float, float, float, float]:
        """Compute entry/exit coordinates for a placed group (center-based IO offsets)."""
        if gid not in self.positions:
            raise KeyError(f"compute_enex: gid not placed: {gid!r}")
        x_bl, y_bl, rot = self.positions[gid]
        r = self._norm_rot(int(rot))
        cx, cy = self.center_from_bl(gid=gid, x_bl=int(x_bl), y_bl=int(y_bl), rot=int(r))
        g = self.groups[gid]
        ent_dx, ent_dy = self.rotate_point(float(g.ent_rel_x), float(g.ent_rel_y), int(r))
        exi_dx, exi_dy = self.rotate_point(float(g.exi_rel_x), float(g.exi_rel_y), int(r))
        return (float(cx) + float(ent_dx), float(cy) + float(ent_dy), float(cx) + float(exi_dx), float(cy) + float(exi_dy))

    def estimate_delta_obj(self, *, gid: GroupId, x: float, y: float, rot: int) -> float:
        """Estimate *scaled* objective delta if placing `gid` at (x,y,rot).

        Behavior is aligned to `env.py`:
        - If not placeable -> return 0.0 (caller can still rank/ignore).
        - Temporarily place using `try_place`, compute delta, then restore.
        """
        x_bl = self._as_int(x, name="x")
        y_bl = self._as_int(y, name="y")
        r = self._norm_rot(int(rot))
        if not self.is_placeable(gid, float(x_bl), float(y_bl), int(r)):
            return 0.0
        cost_prev = float(self.cal_obj())
        prev_state = (dict(self.positions), set(self.placed), list(self.remaining))
        try:
            self.try_place(gid, float(x_bl), float(y_bl), int(r))
            cost_new = float(self.cal_obj())
        finally:
            self.positions, self.placed, self.remaining = prev_state
        return (cost_new - cost_prev) / float(self.reward_scale)

    def try_place(self, gid: GroupId, x: float, y: float, rot: int) -> bool:
        """Place a group if feasible; returns True on success.

        This mirrors `env.py` semantics (positions/placed/remaining only).
        NOTE: Engine-internal caches (occ/invalid/node_feat) are NOT updated here.
        Use `_apply_place(..., update_caches=True)` when you need cache consistency.
        """
        x_bl = self._as_int(x, name="x")
        y_bl = self._as_int(y, name="y")
        r = self._norm_rot(int(rot))
        if not self.is_placeable(gid, float(x_bl), float(y_bl), int(r)):
            return False
        self.positions[gid] = (int(x_bl), int(y_bl), int(r))
        self.placed.add(gid)
        if gid in self.remaining:
            self.remaining.remove(gid)
        return True

    def _apply_place(self, gid: GroupId, x: float, y: float, rot: int, *, update_caches: bool) -> None:
        """Internal: apply a placement and optionally update engine caches."""
        x_bl = self._as_int(x, name="x")
        y_bl = self._as_int(y, name="y")
        r = self._norm_rot(int(rot))
        self.positions[gid] = (int(x_bl), int(y_bl), int(r))
        self.placed.add(gid)
        if gid in self.remaining:
            self.remaining.remove(gid)
        if update_caches:
            self._paint_occupancy(gid, int(x_bl), int(y_bl), int(r))
            idx = self.gid_to_idx.get(gid, None)
            if idx is not None:
                cx, cy = self.center_from_bl(gid=gid, x_bl=int(x_bl), y_bl=int(y_bl), rot=int(r))
                self._node_feat[idx, 2] = 1.0
                self._node_feat[idx, 3] = float(cx) / float(self.grid_width)
                self._node_feat[idx, 4] = float(cy) / float(self.grid_height)
                self._node_feat[idx, 5] = float(int(r) % 360) / 360.0

    def _build_graph_static(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build sparse graph tensors from group_flow.

        Returns:
          edge_index: int64 [2,E]
          edge_attr: float32 [E,1]  (#nets/weight)
        """
        edges: List[List[int]] = []
        attrs: List[List[float]] = []
        for src, dsts in self.group_flow.items():
            if src not in self.gid_to_idx:
                continue
            for dst, w in dsts.items():
                if dst not in self.gid_to_idx:
                    continue
                edges.append([self.gid_to_idx[src], self.gid_to_idx[dst]])
                attrs.append([float(w)])
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous()
            edge_attr = torch.tensor(attrs, dtype=torch.float32, device=self.device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.zeros((0, 1), dtype=torch.float32, device=self.device)
        return edge_index, edge_attr

    # ---- feasibility / objective ----
    def is_placeable(self, gid: GroupId, x: float, y: float, rot: int) -> bool:
        if gid not in self.groups:
            return False
        g = self.groups[gid]
        if not g.movable:
            return False
        if (not g.rotatable) and int(rot) != 0:
            rot = 0
        rot = self._norm_rot(int(rot))

        # IMPORTANT: (x,y) is bottom-left of rotated AABB in integer grid coordinates.
        x_bl = self._as_int(x, name="x")
        y_bl = self._as_int(y, name="y")

        rect = self._body_rect_from_bl(gid=gid, x_bl=x_bl, y_bl=y_bl, rot=rot)  # (x0,y0,x1,y1)

        # clearance rotation (int)
        cL, cR, cB, cT = self._clearance_lrtb(g, rot)
        rect_pad = self._pad_rect_i(rect, cL=int(cL), cR=int(cR), cB=int(cB), cT=int(cT))

        # boundary: BOTH body and pad must stay inside.
        # rect = (x0, y0, x1, y1)
        if rect[0] < 0 or rect[1] < 0 or rect[2] > self.grid_width or rect[3] > self.grid_height:
            return False
        if rect_pad[0] < 0 or rect_pad[1] < 0 or rect_pad[2] > self.grid_width or rect_pad[3] > self.grid_height:
            return False

        # body hits core invalid (static/occ/zone)
        if self._recti_hits_invalid(rect, self._invalid):
            return False
        # body hits others' clearance
        if self._recti_hits_invalid(rect, self._clear_invalid):
            return False
        # my clearance/pad hits static/occ/zone
        if self._recti_hits_invalid(rect_pad, self._invalid):
            return False

        return True

    def cal_obj(self) -> float:
        """Simple objective: weighted L1 flow distance + compactness.

        NOTE: This is a minimal version; you can replace with your notebook-accurate objective later.
        """
        if not self.placed:
            return 0.0

        total_distance = 0.0
        # Flow distance: EXIT(g1) -> ENTRY(g2) Manhattan distance (legacy behavior).
        enex_cache: Dict[GroupId, Tuple[float, float, float, float]] = {}
        for gid in self.placed:
            enex_cache[gid] = self.compute_enex(gid)
        for g1 in self.placed:
            _en1_x, _en1_y, ex1_x, ex1_y = enex_cache[g1]
            for g2, w in self.group_flow.get(g1, {}).items():
                if g2 not in self.placed:
                    continue
                en2_x, en2_y, _ex2_x, _ex2_y = enex_cache[g2]
                total_distance += float(w) * (abs(ex1_x - en2_x) + abs(ex1_y - en2_y))

        min_x = float("inf")
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")
        for gid in self.placed:
            x_bl, y_bl, rot = self.positions[gid]
            w, h = self.rotated_size(self.groups[gid], rot)
            min_x = min(min_x, float(x_bl))
            max_x = max(max_x, float(x_bl) + float(w))
            min_y = min(min_y, float(y_bl))
            max_y = max(max_y, float(y_bl) + float(h))
        compactness = 0.5 * ((max_x - min_x) + (max_y - min_y))
        return float(total_distance + compactness)

    def _remaining_area_ratio(self) -> float:
        """Remaining area / total area ratio in [0,1]. Mirrors env.py."""
        total_area = 0.0
        remaining_area = 0.0
        for gid, g in self.groups.items():
            a = float(g.width) * float(g.height)
            total_area += a
            if gid in self.remaining:
                remaining_area += a
        if total_area <= 0.0:
            return 1.0
        ratio = remaining_area / total_area
        if ratio < 0.0:
            return 0.0
        if ratio > 1.0:
            return 1.0
        return float(ratio)

    def _failure_penalty(self) -> float:
        """Penalty reward for failed placement or no valid actions (negative)."""
        return -float(self.penalty_scale) * self._remaining_area_ratio()

    # ---- cached maps ----
    def _build_static_invalid(self) -> torch.Tensor:
        inv = torch.zeros((self.grid_height, self.grid_width), dtype=torch.bool, device=self.device)
        if self.forbidden_mask is not None:
            inv |= self.forbidden_mask.to(self.device)
        if self.column_mask is not None:
            inv |= self.column_mask.to(self.device)
        return inv

    def _recompute_invalid(self) -> None:
        # invalid = static | occ | zone (NO new allocation)
        # Keep `self._invalid` as a persistent buffer and update it in-place to reduce
        # large temporary tensors / memory traffic.
        self._invalid.copy_(self._static_invalid)
        self._invalid.logical_or_(self._occ_invalid)
        self._invalid.logical_or_(self._zone_invalid)

    def _paint_occupancy(self, gid: GroupId, x_bl: int, y_bl: int, rot: int) -> None:
        """Paint occupancy and clearance maps for an already-validated placement.

        Inputs are bottom-left integer coordinates of the rotated AABB footprint.
        """
        rot = self._norm_rot(int(rot))
        rect = self._body_rect_from_bl(gid=gid, x_bl=int(x_bl), y_bl=int(y_bl), rot=rot)
        x0, y0, x1, y1 = rect
        if x0 < x1 and y0 < y1:
            self._occ_invalid[y0:y1, x0:x1] = True

        g = self.groups[gid]
        cL, cR, cB, cT = self._clearance_lrtb(g, rot)
        rect_pad = self._pad_rect_i(rect, cL=int(cL), cR=int(cR), cB=int(cB), cT=int(cT))
        px0, py0, px1, py1 = rect_pad
        if px0 < px1 and py0 < py1:
            self._clear_invalid[py0:py1, px0:px1] = True

        self._recompute_invalid()

    def _build_obs(self) -> Dict[str, torch.Tensor]:
        """Return *core* observation (model-agnostic).

        Policy-specific wrappers (AlphaChip/MaskPlace/TopK) should attach their own
        extra observation fields on top of this core dict.
        """
        gid = self.remaining[0] if self.remaining else (self.node_ids[0] if self.node_ids else list(self.groups.keys())[0])
        g = self.groups[gid]

        # metadata (keep dim=12 for now; refine later)
        placed_ratio = float(len(self.placed)) / float(max(1, len(self.node_ids)))
        remaining_ratio = float(len(self.remaining)) / float(max(1, len(self.node_ids)))
        cost = float(self.cal_obj())
        self._netlist_metadata[:] = 0.0
        self._netlist_metadata[0] = placed_ratio
        self._netlist_metadata[1] = remaining_ratio
        self._netlist_metadata[2] = cost
        self._netlist_metadata[3] = float(self.grid_width)
        self._netlist_metadata[4] = float(self.grid_height)

        N = int(len(self.node_ids))
        placed_mask = torch.zeros((N,), dtype=torch.bool, device=self.device)
        pos = torch.full((N, 3), -1, dtype=torch.long, device=self.device)  # (x_bl,y_bl,rot) or -1
        for gid2 in self.placed:
            idx = self.gid_to_idx.get(gid2, None)
            if idx is None:
                continue
            placed_mask[int(idx)] = True
            x_bl, y_bl, rot = self.positions[gid2]
            pos[int(idx), 0] = int(x_bl)
            pos[int(idx), 1] = int(y_bl)
            pos[int(idx), 2] = int(rot)

        # current gid index (static node list)
        cur_idx = int(self.gid_to_idx.get(gid, 0))

        obs: Dict[str, torch.Tensor] = {
            "netlist_metadata": self._netlist_metadata,
            "current_gid_idx": torch.tensor([cur_idx], dtype=torch.long, device=self.device),
            # NOTE: normalized to canvas (grid) size for scale-stable learning.
            "next_group_wh": torch.tensor(
                [float(g.width) / float(self.grid_width), float(g.height) / float(self.grid_height)],
                dtype=torch.float32,
                device=self.device,
            ),
            "placed_mask": placed_mask,
            "positions_bl": pos,
            "step_count": torch.tensor([int(self._step_count)], dtype=torch.long, device=self.device),
        }
        return obs

    def step_place(self, *, x: float, y: float, rot: int) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        """Engine step: place current group at raw (x,y,rot)."""
        self._step_count += 1
        info: Dict[str, Any] = {}

        if not self.remaining:
            return self._build_obs(), 0.0, True, False, {"reason": "done"}

        gid = self.remaining[0]
        x_bl = self._as_int(x, name="x")
        y_bl = self._as_int(y, name="y")
        r = self._norm_rot(int(rot))
        info.update({"gid": gid, "x": int(x_bl), "y": int(y_bl), "rot": int(r)})

        cost_prev = self.cal_obj()
        terminated = False
        truncated = False
        reward = 0.0

        if not self.is_placeable(gid, float(x_bl), float(y_bl), int(r)):
            reward = -1.0 / float(self.reward_scale)
            truncated = True
            info["reason"] = "not_placeable"
        else:
            self._apply_place(gid, float(x_bl), float(y_bl), int(r), update_caches=True)
            # next-gid dependent zones
            self._update_zone_invalid_for_next()

            cost_new = self.cal_obj()
            reward = -(cost_new - cost_prev) / float(self.reward_scale)
            terminated = len(self.remaining) == 0
            if self.max_steps is not None and self._step_count >= self.max_steps:
                truncated = True
            info["reason"] = "placed"

        return self._build_obs(), float(reward), bool(terminated), bool(truncated), info

    def step_masked(
        self,
        *,
        action: int,
        x: float,
        y: float,
        rot: int,
        mask: Optional[torch.Tensor],
        action_space_n: int,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        """Engine step for discrete action selection with action-masking (env.py 270-328 equivalent).

        Wrapper supplies:
        - action: chosen discrete index
        - (x,y,rot): decoded placement for that index
        - mask: torch.BoolTensor[N] where True means valid (can be None)
        - action_space_n: N (range check)
        """
        info: Dict[str, Any] = {}
        if extra_info:
            info.update(dict(extra_info))

        self._step_count += 1
        reward = 0.0
        terminated = False
        truncated = False

        # (1) Failure cases: set flags/reward only (single exit)
        if len(self.remaining) > 0 and mask is not None and int(mask.to(torch.int64).sum().item()) == 0:
            info["invalid"] = True
            info["reason"] = "no_valid_actions"
            reward = float(self._failure_penalty())
            truncated = True

        elif int(action) < 0 or int(action) >= int(action_space_n):
            info["invalid"] = True
            info["reason"] = "action_out_of_range"
            reward = float(self._failure_penalty())
            truncated = True

        elif mask is not None and (not bool(mask[int(action)].item())):
            info["invalid"] = True
            info["reason"] = "masked_action"
            reward = float(self._failure_penalty())
            truncated = True

        else:
            # (2) Normal placement path
            if not self.remaining:
                terminated = True
                info["invalid"] = False
                info["reason"] = "done"
            else:
                gid = self.remaining[0]
                x_bl = self._as_int(x, name="x")
                y_bl = self._as_int(y, name="y")
                r = self._norm_rot(int(rot))
                if not self.is_placeable(gid, float(x_bl), float(y_bl), int(r)):
                    info["invalid"] = True
                    info["reason"] = "not_placeable"
                    reward = float(self._failure_penalty())
                    truncated = True
                else:
                    cost_prev = float(self.cal_obj())
                    self._apply_place(gid, float(x_bl), float(y_bl), int(r), update_caches=True)
                    cost_new = float(self.cal_obj())
                    reward = -(cost_new - cost_prev) / float(self.reward_scale)
                    info["invalid"] = False
                    terminated = len(self.remaining) == 0
                    truncated = self.max_steps is not None and self._step_count >= self.max_steps

        obs = self._build_obs()

        # (3) Single logging point
        if getattr(self, "log", False) and (terminated or truncated):
            cost_now = float(self.cal_obj())
            reason = info.get("reason", "")
            print(
                f"[env] end: terminated={terminated} truncated={truncated} "
                f"remaining={len(self.remaining)} placed={len(self.placed)} step={self._step_count} "
                f"cost={cost_now:.3f} reason={reason} reward={float(reward):.3f}"
            )

        return obs, float(reward), bool(terminated), bool(truncated), info

    # ---- gym api ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        options = dict(options or {})
        initial_positions = options.get("initial_positions", None)
        remaining_order = options.get("remaining_order", None)

        self.positions = {}
        self.placed = set()

        # Base order: "harder-first" heuristic.
        #
        # TEMP (requested): difficulty = facility_area / free_area.
        # - invalid_area: inv.sum() where inv = static_only | zone_invalid_for_gid(gid)
        # - free_area: total_area - invalid_area
        # - facility_area: footprint area (w*h)
        # Larger difficulty => bigger footprint relative to available free space => placed earlier.
        #
        # TODO(ord1): Restore placeable(K/T) ordering using conv2d-based top-left feasibility.
        ordering: List[Tuple[float, float, GroupId, int]] = []
        static_only = self._static_invalid  # no occupancy at reset-time
        for gid in self.groups.keys():
            gg = self.groups[gid]
            inv = static_only | self._zone_invalid_for_gid(gid)
            facility_area = float(gg.width) * float(gg.height)
            # invalid_area (requested): true invalid area on grid (cell units; 1 cell == 1 area unit here)
            invalid_area = int(inv.to(torch.int64).sum().item())
            total_area = int(self.grid_width) * int(self.grid_height)
            free_area = max(1, int(total_area) - int(invalid_area))
            denom = float(free_area)
            difficulty = float(facility_area) / float(denom)
            ordering.append((difficulty, facility_area, gid, invalid_area))

        # Sort:
        # 1) difficulty descending (harder first)
        # 2) facility_area descending (bigger first among equally hard)
        # 3) gid for stable order
        ordering.sort(key=lambda t: (-t[0], -t[1], str(t[2])))
        base_remaining = [gid for _diff, _area, gid, _inv in ordering]

        if self.log:
            print("[reset_order] harder-first by difficulty (=facility_area/free_area)")
            for rank, (diff, area, gid, invalid_area) in enumerate(ordering, start=1):
                total_area = int(self.grid_width) * int(self.grid_height)
                free_area = max(1, int(total_area) - int(invalid_area))
                print(
                    f"  {rank}/{len(ordering)} gid={gid} "
                    f"facility_area={area:.1f} "
                    f"invalid_area={invalid_area} "
                    f"free_area={free_area} "
                    f"difficulty={diff:.6g}"
                )
        if remaining_order is not None:
            if not isinstance(remaining_order, list):
                raise ValueError("reset(options): remaining_order must be a list of group ids")
            # Validate remaining_order elements and uniqueness.
            seen = set()
            for gid in remaining_order:
                if gid not in self.groups:
                    raise ValueError(f"reset(options): remaining_order contains unknown group id: {gid!r}")
                if gid in seen:
                    raise ValueError(f"reset(options): remaining_order contains duplicate group id: {gid!r}")
                seen.add(gid)
            # Keep order provided, append missing groups by base order.
            rest = [gid for gid in base_remaining if gid not in seen]
            self.remaining = list(remaining_order) + rest
        else:
            self.remaining = list(base_remaining)

        self._step_count = 0
        self._occ_invalid.zero_()
        self._clear_invalid.zero_()
        self._zone_invalid.zero_()
        self._recompute_invalid()

        # reset dynamic node features (keep static w/h)
        self._node_feat[:, 2:] = 0.0

        # Apply validated initial placements (and sync caches).
        if initial_positions is not None:
            if not isinstance(initial_positions, dict):
                raise ValueError("reset(options): initial_positions must be a dict {gid: (x,y,rot)}")
            for gid, pose in initial_positions.items():
                if gid not in self.groups:
                    raise ValueError(f"reset(options): initial_positions has unknown group id: {gid!r}")
                if (not isinstance(pose, (tuple, list))) or len(pose) != 3:
                    raise ValueError(f"reset(options): initial_positions[{gid!r}] must be (x,y,rot)")
                x = self._as_int(pose[0], name="x")
                y = self._as_int(pose[1], name="y")
                rot = self._norm_rot(int(pose[2]))
                if gid in self.placed:
                    raise ValueError(f"reset(options): initial_positions contains duplicate gid: {gid!r}")
                if not self.is_placeable(gid, float(x), float(y), int(rot)):
                    raise ValueError(
                        f"reset(options): invalid initial placement gid={gid!r} pose=({x},{y},{rot})"
                    )
                self._apply_place(gid, float(x), float(y), int(rot), update_caches=True)

            # Ensure invalid map is consistent (paint already recomputed, but keep invariant explicit).
            self._update_zone_invalid_for_next()

        # Apply next-gid dependent zones on fresh reset too.
        self._update_zone_invalid_for_next()
        return self._build_obs(), {}

    # ---- snapshot api (for search/MCTS) ----
    def get_snapshot(self) -> Dict[str, object]:
        """Return a deep-ish snapshot for deterministic restore in search algorithms.

        Notes:
        - This is additive (does not change training/inference behavior).
        - Tensors are cloned to ensure isolation across rollouts.
        """
        return {
            "positions": dict(self.positions),
            "placed": set(self.placed),
            "remaining": list(self.remaining),
            "_step_count": int(self._step_count),
            "_occ_invalid": self._occ_invalid.clone(),
            "_clear_invalid": self._clear_invalid.clone(),
            "_invalid": self._invalid.clone(),
            "_zone_invalid": self._zone_invalid.clone(),
            "_node_feat": self._node_feat.clone(),
        }

    def set_snapshot(self, snapshot: Dict[str, object]) -> None:
        """Restore a snapshot produced by `get_snapshot`."""
        self.positions = dict(snapshot.get("positions", {}))  # type: ignore[arg-type]
        self.placed = set(snapshot.get("placed", set()))  # type: ignore[arg-type]
        self.remaining = list(snapshot.get("remaining", []))  # type: ignore[arg-type]
        self._step_count = int(snapshot.get("_step_count", 0))

        occ = snapshot.get("_occ_invalid", None)
        clr = snapshot.get("_clear_invalid", None)
        inv = snapshot.get("_invalid", None)
        zinv = snapshot.get("_zone_invalid", None)
        nf = snapshot.get("_node_feat", None)
        if isinstance(occ, torch.Tensor):
            self._occ_invalid = occ.to(device=self.device, dtype=torch.bool).clone()
        if isinstance(clr, torch.Tensor):
            self._clear_invalid = clr.to(device=self.device, dtype=torch.bool).clone()
        if isinstance(zinv, torch.Tensor):
            self._zone_invalid = zinv.to(device=self.device, dtype=torch.bool).clone()
        if isinstance(nf, torch.Tensor):
            self._node_feat = nf.to(device=self.device, dtype=torch.float32).clone()
        # Recompute invalid from layers to keep invariant.
        self._recompute_invalid()

    def step(self, action: int):
        x, y, rot, i, j = self.decode_action(action)
        obs, reward, terminated, truncated, info = self.step_place(x=x, y=y, rot=rot)
        info.update({"cell_i": i, "cell_j": j})
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    # Timing-focused smoke demo:
    # - env init time
    # - reset time
    # - one step time (any action; validity not required)
    import time

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dev = torch.device("cpu")

    # --- groups: footprint + constraint attributes ---
    groups = {
        "A": FacilityGroup(id="A", width=20, height=10, rotatable=True, facility_weight=3.0, facility_height=2.0, facility_dry=0.0),
        "B": FacilityGroup(id="B", width=15, height=15, rotatable=True, facility_weight=4.0, facility_height=2.0, facility_dry=0.0),
        # Make C "heavy + tall + dry-sensitive" to exercise constraints.
        "C": FacilityGroup(id="C", width=18, height=12, rotatable=True, facility_weight=12.0, facility_height=10.0, facility_dry=2.0),
    }
    group_flow = {"A": {"B": 1.0}, "B": {"C": 0.7}}

    # Base forbidden area.
    forbidden = torch.zeros((80, 120), dtype=torch.bool, device=dev)
    forbidden[0:20, 0:30] = True

    # Constraints (optional; unified default + area override):
    default_weight = 10.0
    # Right-half has higher allowable weight.
    weight_areas = [{"rect": (60, 0, 120, 80), "value": 20.0}]
    default_height = 20.0
    # Top strip has low ceiling height.
    height_areas = [{"rect": (0, 60, 120, 80), "value": 5.0}]
    default_dry = 0.0
    # Middle-left zone has higher dry requirement (reverse inequality).
    dry_areas = [{"rect": (0, 40, 60, 80), "value": 2.0}]

    t0 = time.perf_counter()
    env = FactoryLayoutEnv(
        grid_width=120,
        grid_height=80,
        groups=groups,
        group_flow=group_flow,
        forbidden_mask=forbidden,
        device=dev,
        max_steps=10,
        log=True,
        weight_areas=weight_areas,
        height_areas=height_areas,
        dry_areas=dry_areas,
        default_weight=default_weight,
        default_height=default_height,
        default_dry=default_dry,
    )
    init_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    obs, _ = env.reset()
    reset_ms = (time.perf_counter() - t1) * 1000.0

    # Step with any arbitrary raw placement (we only care about step runtime here).
    # NOTE: This may be invalid/not_placeable; that's fine for this timing demo.
    t2 = time.perf_counter()
    obs2, reward, terminated, truncated, info = env.step_place(x=0.0, y=0.0, rot=0)
    step_ms = (time.perf_counter() - t2) * 1000.0

    print("[env_demo]")
    print(" device=", dev)
    print(f" init_ms={init_ms:.2f} reset_ms={reset_ms:.2f} step_ms={step_ms:.2f}")
    print(" placed_now=", len(env.placed), "remaining_now=", len(env.remaining))
    print(" step_info.reason=", info.get("reason"), "reward=", reward, "terminated=", terminated, "truncated=", truncated)
    print(" obs_keys=", list(obs.keys()))
    print(" obs2_keys=", list(obs2.keys()))

    # Optional visualization (interactive toggles).
    from envs.visualizer import plot_flow_graph, plot_layout

    plot_layout(env, candidate_set=None)
    plot_flow_graph(env)

