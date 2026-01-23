from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from envs.env import FactoryLayoutEnv, FacilityGroup
try:
    # Optional legacy support (may be removed in new-only setups).
    from envs.env_old import FacilityGroup as FacilityGroupOld  # type: ignore
    from envs.env_old import FactoryLayoutEnvOld, RectMask  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    FacilityGroupOld = None  # type: ignore[assignment]
    FactoryLayoutEnvOld = None  # type: ignore[assignment]
    RectMask = None  # type: ignore[assignment]

GroupId = Union[int, str]
RectI = Tuple[int, int, int, int]  # (x0, y0, x1, y1) half-open


@dataclass(frozen=True)
class LoadedEnv:
    """Result of loading an environment from a JSON spec."""

    env: FactoryLayoutEnv
    reset_kwargs: Dict[str, Any]


def _edges_to_adj(edges: List[List[Any]]) -> Dict[GroupId, Dict[GroupId, float]]:
    adj: Dict[GroupId, Dict[GroupId, float]] = {}
    for e in edges:
        if len(e) != 3:
            raise ValueError(f"flow edge must be [src, dst, weight], got: {e}")
        src, dst, w = e
        adj.setdefault(src, {})[dst] = float(w)
    return adj


def _mask_from_forbidden_rects(grid_w: int, grid_h: int, rects: List[RectI]):
    """Build a RectMask where True means allowed and given rects are forbidden (set to False)."""
    if RectMask is None:
        raise RuntimeError("Legacy env_old support is not available (envs/env_old.py is missing).")
    allowed = [[True for _ in range(grid_w)] for _ in range(grid_h)]
    for x0, y0, x1, y1 in rects:
        x0 = max(0, min(grid_w, int(x0)))
        x1 = max(0, min(grid_w, int(x1)))
        y0 = max(0, min(grid_h, int(y0)))
        y1 = max(0, min(grid_h, int(y1)))
        for y in range(y0, y1):
            row = allowed[y]
            for x in range(x0, x1):
                row[x] = False
    return RectMask(allowed)


def _torch_forbidden_mask_from_rects(
    grid_w: int, grid_h: int, rects: List[RectI], *, device: torch.device
) -> torch.Tensor:
    """Build torch.BoolTensor[H,W] where True means forbidden/invalid."""
    m = torch.zeros((grid_h, grid_w), dtype=torch.bool, device=device)
    for x0, y0, x1, y1 in rects:
        x0 = max(0, min(grid_w, int(x0)))
        x1 = max(0, min(grid_w, int(x1)))
        y0 = max(0, min(grid_h, int(y0)))
        y1 = max(0, min(grid_h, int(y1)))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return m


def _torch_rect_union_mask_from_rects(
    grid_w: int, grid_h: int, rects: List[RectI], *, device: torch.device
) -> torch.Tensor:
    """Build torch.BoolTensor[H,W] where True means inside (union of rects)."""
    m = torch.zeros((grid_h, grid_w), dtype=torch.bool, device=device)
    for x0, y0, x1, y1 in rects:
        x0 = max(0, min(grid_w, int(x0)))
        x1 = max(0, min(grid_w, int(x1)))
        y0 = max(0, min(grid_h, int(y0)))
        y1 = max(0, min(grid_h, int(y1)))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return m


def load_env(json_path: str, *, device: torch.device | None = None) -> LoadedEnv:
    """Load a FactoryLayoutEnv (new engine) from a JSON spec file."""
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    grid = data["grid"]
    env_cfg = data["env"]
    groups_cfg: Dict[str, Dict[str, Any]] = data["groups"]

    grid_w = int(grid["width"])
    grid_h = int(grid["height"])
    # NOTE: new engine assumes 1 cell == 1 unit. If grid_size != 1, caller must handle scaling.
    # We keep this value for compatibility with existing JSONs but do not apply it here.
    _grid_size = float(grid.get("grid_size", 1.0))

    groups: Dict[GroupId, FacilityGroup] = {}
    for gid, g in groups_cfg.items():
        # NOTE: Engine expects integer grid units for geometry and clearance.
        # We convert JSON values to ints here to avoid float/half-cell ambiguity.
        def _to_int(v: Any) -> int:
            try:
                return int(round(float(v)))
            except Exception as e:
                raise ValueError(f"groups[{gid}]: expected number, got {v!r}") from e

        groups[gid] = FacilityGroup(
            id=gid,
            width=_to_int(g["width"]),
            height=_to_int(g["height"]),
            movable=bool(g.get("movable", True)),
            rotatable=bool(g.get("rotatable", True)),
            facility_weight=float(g.get("facility_weight", 0.0)),
            facility_height=float(g.get("facility_height", 0.0)),
            facility_dry=float(g.get("facility_dry", 0.0)),
            # IO offsets (center-based local coords; optional)
            ent_rel_x=float(g.get("ent_rel_x", 0.0)),
            ent_rel_y=float(g.get("ent_rel_y", 0.0)),
            exi_rel_x=float(g.get("exi_rel_x", 0.0)),
            exi_rel_y=float(g.get("exi_rel_y", 0.0)),
            facility_clearance_left=_to_int(g.get("facility_clearance_left", 0)),
            facility_clearance_right=_to_int(g.get("facility_clearance_right", 0)),
            facility_clearance_bottom=_to_int(g.get("facility_clearance_bottom", 0)),
            facility_clearance_top=_to_int(g.get("facility_clearance_top", 0)),
        )

    flow_raw = data.get("flow", {})
    if isinstance(flow_raw, list):
        flow = _edges_to_adj(flow_raw)
    elif isinstance(flow_raw, dict):
        flow = {src: {dst: float(w) for dst, w in dsts.items()} for src, dsts in flow_raw.items()}
    else:
        raise ValueError("flow must be a dict adjacency or an edge list")

    masks = data.get("masks", {})
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    forbidden_mask = None
    if "forbidden_rects" in masks:
        forbidden_mask = _torch_forbidden_mask_from_rects(
            grid_w, grid_h, masks["forbidden_rects"], device=dev
        )
    column_mask = None
    if "column_rects" in masks:
        column_mask = _torch_forbidden_mask_from_rects(grid_w, grid_h, masks["column_rects"], device=dev)

    zones = data.get("zones", {})
    weight_areas: List[Dict[str, Any]] = list(zones.get("weight_areas", [])) if isinstance(zones, dict) else []
    dry_areas: List[Dict[str, Any]] = list(zones.get("dry_areas", [])) if isinstance(zones, dict) else []
    height_areas: List[Dict[str, Any]] = list(zones.get("height_areas", [])) if isinstance(zones, dict) else []

    # NOTE: New schema (no backward-compat): areas are dicts with {"rect": [x0,y0,x1,y1], "value": float}
    for name, areas in [("weight_areas", weight_areas), ("dry_areas", dry_areas), ("height_areas", height_areas)]:
        if not isinstance(areas, list):
            raise ValueError(f"zones.{name} must be a list")
        for i, a in enumerate(areas):
            if not isinstance(a, dict):
                raise ValueError(f"zones.{name}[{i}] must be an object with keys rect/value")
            if "rect" not in a or "value" not in a:
                raise ValueError(f"zones.{name}[{i}] must contain keys 'rect' and 'value'")

    env = FactoryLayoutEnv(
        grid_width=grid_w,
        grid_height=grid_h,
        groups=groups,
        group_flow=flow,
        max_steps=int(env_cfg.get("max_steps")) if env_cfg.get("max_steps") is not None else None,
        reward_scale=float(env_cfg.get("reward_scale", 100.0)),
        penalty_scale=float(env_cfg.get("penalty_scale", 30.0)),
        forbidden_mask=forbidden_mask,
        column_mask=column_mask,
        # Optional defaults (if omitted, behave as "no constraint" by default):
        # - weight/height: +inf => never invalid by (map < facility_value)
        # - dry (reverse): -inf => never invalid by (map > facility_value)
        default_weight=float(env_cfg.get("default_weight", float("inf"))),
        default_height=float(env_cfg.get("default_height", float("inf"))),
        default_dry=float(env_cfg.get("default_dry", -float("inf"))),
        weight_areas=weight_areas,
        height_areas=height_areas,
        dry_areas=dry_areas,
        device=dev,
        log=bool(env_cfg.get("log", False)),
    )

    reset_cfg = data.get("reset", {})
    reset_kwargs: Dict[str, Any] = {}
    if "initial_positions" in reset_cfg and reset_cfg["initial_positions"] is not None:
        ip = {}
        for gid, pose in reset_cfg["initial_positions"].items():
            if not (isinstance(pose, list) and len(pose) == 3):
                raise ValueError(f"initial_positions[{gid}] must be [x, y, rot], got: {pose}")
            # (x,y) are integer bottom-left coordinates of rotated AABB.
            # Convert to ints to match engine semantics.
            try:
                x_bl = int(round(float(pose[0])))
                y_bl = int(round(float(pose[1])))
            except Exception as e:
                raise ValueError(f"initial_positions[{gid}]: x/y must be numbers, got: {pose}") from e
            ip[gid] = (x_bl, y_bl, int(pose[2]))
        reset_kwargs["initial_positions"] = ip
    if "remaining_order" in reset_cfg and reset_cfg["remaining_order"] is not None:
        reset_kwargs["remaining_order"] = list(reset_cfg["remaining_order"])

    return LoadedEnv(env=env, reset_kwargs=reset_kwargs)


@dataclass(frozen=True)
class LoadedEnvOld:
    """Result of loading a legacy candidate env from a JSON spec."""

    env: Any
    reset_kwargs: Dict[str, Any]


def load_env_old(json_path: str) -> LoadedEnvOld:
    """Load a FactoryLayoutEnvOld (legacy) from a JSON spec file."""
    if FactoryLayoutEnvOld is None or FacilityGroupOld is None:
        raise RuntimeError("load_env_old() is unavailable because envs/env_old.py is missing.")
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    grid = data["grid"]
    env_cfg = data["env"]
    groups_cfg: Dict[str, Dict[str, Any]] = data["groups"]

    grid_w = int(grid["width"])
    grid_h = int(grid["height"])
    grid_size = float(grid.get("grid_size", 1.0))

    groups: Dict[GroupId, FacilityGroupOld] = {}
    for gid, g in groups_cfg.items():
        groups[gid] = FacilityGroupOld(
            id=gid,
            width=float(g["width"]),
            height=float(g["height"]),
            movable=bool(g.get("movable", True)),
            rotatable=bool(g.get("rotatable", True)),
            dry_level=float(g.get("dry_level", 0.0)),
            weight=float(g.get("weight", 0.0)),
            type=g.get("type"),
            ent_rel=tuple(g.get("ent_rel", (0.0, 0.0))),  # type: ignore[arg-type]
            exi_rel=tuple(g.get("exi_rel", (0.0, 0.0))),  # type: ignore[arg-type]
        )

    flow_raw = data.get("flow", {})
    if isinstance(flow_raw, list):
        flow = _edges_to_adj(flow_raw)
    elif isinstance(flow_raw, dict):
        flow = {src: {dst: float(w) for dst, w in dsts.items()} for src, dsts in flow_raw.items()}
    else:
        raise ValueError("flow must be a dict adjacency or an edge list")

    masks = data.get("masks", {})
    forbidden_mask = None
    if "forbidden_rects" in masks:
        forbidden_mask = _mask_from_forbidden_rects(grid_w, grid_h, masks["forbidden_rects"])
    column_mask = None
    if "column_rects" in masks:
        column_mask = _mask_from_forbidden_rects(grid_w, grid_h, masks["column_rects"])
    dry_mask = None
    if "dry_forbidden_rects" in masks:
        dry_mask = _mask_from_forbidden_rects(grid_w, grid_h, masks["dry_forbidden_rects"])
    weight_mask = None
    if "weight_forbidden_rects" in masks:
        weight_mask = _mask_from_forbidden_rects(grid_w, grid_h, masks["weight_forbidden_rects"])

    env = FactoryLayoutEnvOld(
        grid_width=grid_w,
        grid_height=grid_h,
        grid_size=grid_size,
        groups=groups,
        group_flow=flow,
        max_candidates=int(env_cfg["max_candidates"]),
        max_steps=int(env_cfg.get("max_steps")) if env_cfg.get("max_steps") is not None else None,
        reward_scale=float(env_cfg.get("reward_scale", 100.0)),
        penalty_scale=float(env_cfg.get("penalty_scale", 30.0)),
        forbidden_mask=forbidden_mask,
        column_mask=column_mask,
        dry_mask=dry_mask,
        weight_mask=weight_mask,
        log=bool(env_cfg.get("log", False)),
        seed=env_cfg.get("seed"),
    )

    reset_cfg = data.get("reset", {})
    reset_kwargs: Dict[str, Any] = {}
    if "initial_positions" in reset_cfg and reset_cfg["initial_positions"] is not None:
        ip = {}
        for gid, pose in reset_cfg["initial_positions"].items():
            if not (isinstance(pose, list) and len(pose) == 3):
                raise ValueError(f"initial_positions[{gid}] must be [x, y, rot], got: {pose}")
            ip[gid] = (float(pose[0]), float(pose[1]), int(pose[2]))
        reset_kwargs["initial_positions"] = ip
    if "remaining_order" in reset_cfg and reset_cfg["remaining_order"] is not None:
        reset_kwargs["remaining_order"] = list(reset_cfg["remaining_order"])

    return LoadedEnvOld(env=env, reset_kwargs=reset_kwargs)

