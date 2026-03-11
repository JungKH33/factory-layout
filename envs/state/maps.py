from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..action import GroupId
from ..placement.static import StaticSpec

RectI = Tuple[int, int, int, int]


class GridMaps:
    """Runtime map state + static constraint maps.

    Runtime tensors are copied on `copy()`.
    Static maps/cache are shared by reference for efficiency when device matches.
    """

    def __init__(
        self,
        *,
        grid_height: int,
        grid_width: int,
        device: torch.device,
        forbidden_areas: List[Dict[str, Any]],
        zone_constraints: Dict[str, Dict[str, Any]],
    ) -> None:
        self._H = int(grid_height)
        self._W = int(grid_width)
        self._device = torch.device(device)

        self._static_invalid = self._build_static_invalid(
            self._H,
            self._W,
            self._device,
            forbidden_areas,
        )
        self._zone_constraints = dict(zone_constraints or {})
        (
            self._constraint_maps,
            self._constraint_defined_masks,
            self._constraint_ops,
            self._constraint_dtypes,
        ) = self._build_constraint_maps(
            self._H,
            self._W,
            self._device,
            self._zone_constraints,
        )

        # Runtime fields.
        self.occ_invalid = torch.zeros((self._H, self._W), dtype=torch.bool, device=self._device)
        self.clear_invalid = torch.zeros((self._H, self._W), dtype=torch.bool, device=self._device)
        self.zone_invalid = torch.zeros((self._H, self._W), dtype=torch.bool, device=self._device)
        self.invalid = self._static_invalid.clone()
        self.has_bbox = False
        self.bbox_min_x = 0.0
        self.bbox_max_x = 0.0
        self.bbox_min_y = 0.0
        self.bbox_max_y = 0.0

        # Private caches.
        self._group_specs: Dict[GroupId, StaticSpec] = {}
        self._zone_by_gid: Dict[GroupId, torch.Tensor] = {}

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self._H), int(self._W)

    @property
    def grid_width(self) -> int:
        return int(self._W)

    @property
    def grid_height(self) -> int:
        return int(self._H)

    @property
    def static_invalid(self) -> torch.Tensor:
        return self._static_invalid

    @property
    def constraint_maps(self) -> Dict[str, torch.Tensor]:
        return self._constraint_maps

    @staticmethod
    def _rect_hits_invalid(rect: RectI, src: torch.Tensor) -> bool:
        x0, y0, x1, y1 = rect
        h, w = src.shape
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return True
        if x0 >= x1 or y0 >= y1:
            return False
        return bool(torch.any(src[y0:y1, x0:x1]).item())

    def is_placeable(
        self,
        *,
        gid: GroupId,
        body_rect: RectI,
        pad_rect: RectI,
    ) -> bool:
        invalid = self._static_invalid | self.occ_invalid | self._zone_for_gid(gid)
        if self._rect_hits_invalid(body_rect, invalid):
            return False
        if self._rect_hits_invalid(body_rect, self.clear_invalid):
            return False
        if self._rect_hits_invalid(pad_rect, invalid):
            return False
        return True

    def is_placeable_map(
        self,
        *,
        gid: GroupId,
        body_w: int,
        body_h: int,
        cL: int,
        cR: int,
        cB: int,
        cT: int,
    ) -> torch.Tensor:
        w = int(body_w)
        h = int(body_h)
        cL = int(cL)
        cR = int(cR)
        cB = int(cB)
        cT = int(cT)
        invalid = self._static_invalid | self.occ_invalid | self._zone_for_gid(gid)
        kw = int(w + cL + cR)
        kh = int(h + cB + cT)

        H, W = invalid.shape
        result = torch.zeros((H, W), dtype=torch.bool, device=self._device)
        if kh > H or kw > W or h <= 0 or w <= 0:
            return result

        inv_f = invalid.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        clear_f = self.clear_invalid.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        body_kernel = torch.ones((1, 1, h, w), device=self._device, dtype=torch.float32)
        pad_kernel = torch.ones((1, 1, kh, kw), device=self._device, dtype=torch.float32)

        body_inv = F.conv2d(inv_f, body_kernel, padding=0).squeeze()
        body_clear = F.conv2d(clear_f, body_kernel, padding=0).squeeze()
        pad_inv = F.conv2d(inv_f, pad_kernel, padding=0).squeeze()

        valid_h = H - kh + 1
        valid_w = W - kw + 1
        if valid_h <= 0 or valid_w <= 0:
            return result
        body_inv_slice = body_inv[cB: cB + valid_h, cL: cL + valid_w]
        body_clear_slice = body_clear[cB: cB + valid_h, cL: cL + valid_w]
        valid_mask = (body_inv_slice == 0) & (body_clear_slice == 0) & (pad_inv == 0)
        result[:valid_h, :valid_w] = valid_mask
        return result

    def bind_group_specs(self, group_specs: Dict[GroupId, StaticSpec]) -> None:
        self._group_specs = group_specs
        self.rebuild_zone_cache()

    def copy(self) -> "GridMaps":
        out = object.__new__(GridMaps)
        out._H = self._H
        out._W = self._W
        out._device = self._device

        out.occ_invalid = self.occ_invalid.clone()
        out.clear_invalid = self.clear_invalid.clone()
        out.zone_invalid = self.zone_invalid.clone()
        out.invalid = self.invalid.clone()
        out.has_bbox = bool(self.has_bbox)
        out.bbox_min_x = float(self.bbox_min_x)
        out.bbox_max_x = float(self.bbox_max_x)
        out.bbox_min_y = float(self.bbox_min_y)
        out.bbox_max_y = float(self.bbox_max_y)

        out._static_invalid = self._static_invalid
        out._zone_constraints = self._zone_constraints
        out._constraint_maps = self._constraint_maps
        out._constraint_defined_masks = self._constraint_defined_masks
        out._constraint_ops = self._constraint_ops
        out._constraint_dtypes = self._constraint_dtypes
        out._group_specs = self._group_specs
        out._zone_by_gid = self._zone_by_gid
        return out

    def restore(self, src: "GridMaps") -> None:
        """In-place restore of runtime map fields."""
        if not isinstance(src, GridMaps):
            raise TypeError(f"src must be GridMaps, got {type(src).__name__}")
        if int(self._H) != int(src._H) or int(self._W) != int(src._W):
            raise ValueError(
                f"grid shape mismatch: source=({int(src._H)},{int(src._W)}), "
                f"target=({int(self._H)},{int(self._W)})"
            )
        self.occ_invalid.copy_(src.occ_invalid.to(device=self._device, dtype=torch.bool))
        self.clear_invalid.copy_(src.clear_invalid.to(device=self._device, dtype=torch.bool))
        self.zone_invalid.copy_(src.zone_invalid.to(device=self._device, dtype=torch.bool))
        self.invalid.copy_(src.invalid.to(device=self._device, dtype=torch.bool))
        self.has_bbox = bool(src.has_bbox)
        self.bbox_min_x = float(src.bbox_min_x)
        self.bbox_max_x = float(src.bbox_max_x)
        self.bbox_min_y = float(src.bbox_min_y)
        self.bbox_max_y = float(src.bbox_max_y)

    def reset_runtime(self) -> None:
        self.occ_invalid.zero_()
        self.clear_invalid.zero_()
        self.zone_invalid.zero_()
        self.has_bbox = False
        self.bbox_min_x = 0.0
        self.bbox_max_x = 0.0
        self.bbox_min_y = 0.0
        self.bbox_max_y = 0.0
        self.recompute_invalid()

    def recompute_invalid(self) -> None:
        self.invalid.copy_(self._static_invalid)
        self.invalid.logical_or_(self.occ_invalid)
        self.invalid.logical_or_(self.zone_invalid)

    def paint_rects(
        self,
        *,
        bbox_min_x: float,
        bbox_max_x: float,
        bbox_min_y: float,
        bbox_max_y: float,
        body_rect: RectI,
        clear_rect: RectI,
    ) -> None:
        min_x = float(bbox_min_x)
        min_y = float(bbox_min_y)
        max_x = float(bbox_max_x)
        max_y = float(bbox_max_y)
        if not self.has_bbox:
            self.has_bbox = True
            self.bbox_min_x = float(min_x)
            self.bbox_max_x = float(max_x)
            self.bbox_min_y = float(min_y)
            self.bbox_max_y = float(max_y)
        else:
            self.bbox_min_x = min(float(self.bbox_min_x), float(min_x))
            self.bbox_max_x = max(float(self.bbox_max_x), float(max_x))
            self.bbox_min_y = min(float(self.bbox_min_y), float(min_y))
            self.bbox_max_y = max(float(self.bbox_max_y), float(max_y))

        x0, y0, x1, y1 = body_rect
        bx0 = max(0, min(self._W, int(x0)))
        bx1 = max(0, min(self._W, int(x1)))
        by0 = max(0, min(self._H, int(y0)))
        by1 = max(0, min(self._H, int(y1)))
        if bx0 < bx1 and by0 < by1:
            self.occ_invalid[by0:by1, bx0:bx1] = True

        px0, py0, px1, py1 = clear_rect
        cx0 = max(0, min(self._W, int(px0)))
        cx1 = max(0, min(self._W, int(px1)))
        cy0 = max(0, min(self._H, int(py0)))
        cy1 = max(0, min(self._H, int(py1)))
        if cx0 < cx1 and cy0 < cy1:
            self.clear_invalid[cy0:cy1, cx0:cx1] = True
        self.recompute_invalid()

    def placed_bbox(self) -> Tuple[float, float, float, float]:
        if not self.has_bbox:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(self.bbox_min_x),
            float(self.bbox_max_x),
            float(self.bbox_min_y),
            float(self.bbox_max_y),
        )

    def apply_zone_for_gid(self, gid: Optional[GroupId]) -> None:
        if gid is None:
            self.zone_invalid.zero_()
            self.recompute_invalid()
            return

        zone = self._zone_for_gid(gid)
        self.zone_invalid.copy_(zone)
        self.recompute_invalid()

    def rebuild_zone_cache(self) -> None:
        self._zone_by_gid = {}
        for gid, spec in self._group_specs.items():
            self._zone_by_gid[gid] = self._zone_for_spec(spec).clone()

    def _zone_for_gid(self, gid: GroupId) -> torch.Tensor:
        if gid not in self._group_specs:
            raise KeyError(f"unknown gid={gid!r}")
        z = self._zone_by_gid.get(gid, None)
        if isinstance(z, torch.Tensor):
            return z
        z2 = self._zone_for_spec(self._group_specs[gid]).clone()
        self._zone_by_gid[gid] = z2
        return z2

    def _zone_for_spec(self, spec: StaticSpec) -> torch.Tensor:
        z = torch.zeros((self._H, self._W), dtype=torch.bool, device=self._device)
        zone_values = dict(getattr(spec, "zone_values", {}) or {})
        for cname, cmap in self._constraint_maps.items():
            if cname not in zone_values:
                # Group에 값이 없으면 해당 제약은 무시.
                continue
            op = self._constraint_ops[cname]
            dtype = self._constraint_dtypes[cname]
            gval = self._coerce_group_value(zone_values[cname], dtype=dtype, cname=cname)
            pass_mask = self._compare_constraint(cmap, gval, op=op)
            defined = self._constraint_defined_masks[cname]
            if op == "==":
                # == 는 정의된 영역만 통과 가능 (예: placeable zone-only)
                pass_mask = pass_mask & defined
            else:
                # 그 외 연산은 미정의 영역을 통과로 본다.
                pass_mask = pass_mask | (~defined)
            z |= (~pass_mask)
        return z

    @staticmethod
    def _build_static_invalid(
        H: int,
        W: int,
        device: torch.device,
        forbidden_areas: List[Dict[str, Any]],
    ) -> torch.Tensor:
        inv = torch.zeros((H, W), dtype=torch.bool, device=device)
        for area in forbidden_areas:
            if not isinstance(area, dict) or "rect" not in area:
                continue
            rect = area["rect"]
            x0 = max(0, min(W, int(rect[0])))
            x1 = max(0, min(W, int(rect[2])))
            y0 = max(0, min(H, int(rect[1])))
            y1 = max(0, min(H, int(rect[3])))
            if x1 > x0 and y1 > y0:
                inv[y0:y1, x0:x1] = True
        return inv

    def _build_constraint_maps(
        self,
        H: int,
        W: int,
        device: torch.device,
        constraints: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, str], Dict[str, str]]:
        maps: Dict[str, torch.Tensor] = {}
        defined_masks: Dict[str, torch.Tensor] = {}
        ops: Dict[str, str] = {}
        dtypes: Dict[str, str] = {}

        for cname, cfg in constraints.items():
            if not isinstance(cfg, dict):
                continue
            dtype = str(cfg.get("dtype", "float")).lower()
            op = str(cfg.get("op", "=="))
            areas = cfg.get("areas", [])
            if dtype not in {"float", "int", "bool"}:
                raise ValueError(f"invalid dtype for constraint {cname!r}: {dtype!r}")
            if op not in {"<", "<=", ">", ">=", "==", "!="}:
                raise ValueError(f"invalid op for constraint {cname!r}: {op!r}")
            if dtype == "bool" and op not in {"==", "!="}:
                raise ValueError(f"bool constraint {cname!r} supports only == or !=")
            if not isinstance(areas, list):
                raise ValueError(f"constraint {cname!r}.areas must be a list")

            if dtype == "float":
                m = torch.zeros((H, W), dtype=torch.float32, device=device)
            elif dtype == "int":
                m = torch.zeros((H, W), dtype=torch.int64, device=device)
            else:
                m = torch.zeros((H, W), dtype=torch.bool, device=device)
            defined = torch.zeros((H, W), dtype=torch.bool, device=device)

            for area in areas:
                if not isinstance(area, dict):
                    continue
                rect = area.get("rect", None)
                value = area.get("value", None)
                if rect is None or value is None:
                    continue
                x0, y0, x1, y1 = rect
                x0 = max(0, min(W, int(x0)))
                x1 = max(0, min(W, int(x1)))
                y0 = max(0, min(H, int(y0)))
                y1 = max(0, min(H, int(y1)))
                if x1 <= x0 or y1 <= y0:
                    continue
                if dtype == "float":
                    v = float(value)
                elif dtype == "int":
                    vf = float(value)
                    rv = round(vf)
                    if abs(vf - rv) > 1e-6:
                        raise ValueError(
                            f"constraint {cname!r} expects int values, got {value!r}"
                        )
                    v = int(rv)
                else:
                    v = bool(value)
                m[y0:y1, x0:x1] = v
                defined[y0:y1, x0:x1] = True

            name = str(cname)
            maps[name] = m
            defined_masks[name] = defined
            ops[name] = op
            dtypes[name] = dtype

        return maps, defined_masks, ops, dtypes

    def _coerce_group_value(self, v: Any, *, dtype: str, cname: str) -> Any:
        if dtype == "float":
            return float(v)
        if dtype == "int":
            vf = float(v)
            rv = round(vf)
            if abs(vf - rv) > 1e-6:
                raise ValueError(
                    f"group zone value for {cname!r} must be int-like, got {v!r}"
                )
            return int(rv)
        if dtype == "bool":
            return bool(v)
        raise ValueError(f"unknown dtype {dtype!r} for {cname!r}")

    @staticmethod
    def _compare_constraint(src: torch.Tensor, v: Any, *, op: str) -> torch.Tensor:
        if op == "<":
            return src < v
        if op == "<=":
            return src <= v
        if op == ">":
            return src > v
        if op == ">=":
            return src >= v
        if op == "==":
            return src == v
        if op == "!=":
            return src != v
        raise ValueError(f"unsupported op: {op!r}")
