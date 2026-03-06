from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Tuple

import torch

from ..action import GroupId
from .flow import FlowGraph
from .maps import GridMaps


@dataclass
class EnvState:
    """Runtime engine state.

    Runtime copy policy:
    - copied: placements/placed/remaining/step_count, maps(runtime), flow(runtime)
    - shared: maps static tensors/cache, flow graph edges
    """

    _COPY_PUBLIC_FIELDS: ClassVar[Tuple[str, ...]] = (
        "placements",
        "placed",
        "remaining",
        "step_count",
        "current_gid",
        "placed_nodes_order",
        "maps",
        "flow",
    )

    placements: Dict[GroupId, object]
    placed: set[GroupId]
    remaining: List[GroupId]
    step_count: int
    current_gid: Optional[GroupId]
    placed_nodes_order: List[GroupId]
    maps: GridMaps
    flow: FlowGraph
    _state_sig: Tuple[int, int, Tuple[str, ...]]

    @staticmethod
    def _make_state_sig(*, grid_height: int, grid_width: int, gids: List[GroupId]) -> Tuple[int, int, Tuple[str, ...]]:
        key = tuple(sorted(str(gid) for gid in gids))
        return int(grid_height), int(grid_width), key

    @classmethod
    def empty(
        cls,
        *,
        maps: GridMaps,
        group_specs: Dict[GroupId, object],
        flow: FlowGraph,
    ) -> "EnvState":
        maps.bind_group_specs(group_specs)
        maps.reset_runtime()
        flow.reset_runtime()
        h, w = maps.shape
        return cls(
            placements={},
            placed=set(),
            remaining=[],
            step_count=0,
            current_gid=None,
            placed_nodes_order=[],
            maps=maps,
            flow=flow,
            _state_sig=cls._make_state_sig(grid_height=h, grid_width=w, gids=list(group_specs.keys())),
        )

    def copy(self) -> "EnvState":
        """Return a runtime-safe copy.

        Placement objects inside ``placements`` are shared by reference
        (shallow dict copy).  This is intentional – placements are treated
        as **immutable** after creation (``build_placement``).  Do NOT
        mutate a placement object in-place once it has been stored here.
        """
        return EnvState(
            placements=dict(self.placements),
            placed=set(self.placed),
            remaining=list(self.remaining),
            step_count=int(self.step_count),
            current_gid=self.current_gid,
            placed_nodes_order=list(self.placed_nodes_order),
            maps=self.maps.copy(),
            flow=self.flow.copy(),
            _state_sig=self._state_sig,
        )

    def restore(self, src: "EnvState") -> None:
        """In-place restore from another state with the same signature.

        Placement objects are shared by reference (same contract as ``copy``).
        """
        if not isinstance(src, EnvState):
            raise TypeError(f"src must be EnvState, got {type(src).__name__}")
        if self._state_sig and src._state_sig and self._state_sig != src._state_sig:
            raise ValueError(f"state signature mismatch: source={src._state_sig}, target={self._state_sig}")
        self.placements.clear()
        self.placements.update(src.placements)
        self.placed.clear()
        self.placed.update(src.placed)
        self.remaining[:] = list(src.remaining)
        self.step_count = int(src.step_count)
        self.current_gid = src.current_gid
        self.placed_nodes_order[:] = list(src.placed_nodes_order)
        self.maps.restore(src.maps)
        self.flow.restore(src.flow)

    def set_current_gid(self, gid: Optional[GroupId]) -> None:
        if self.current_gid == gid:
            return
        self.current_gid = gid
        self.maps.apply_zone_for_gid(gid)

    def reset_runtime(self, *, remaining: List[GroupId]) -> None:
        self.placements = {}
        self.placed = set()
        self.remaining = list(remaining)
        self.step_count = 0
        self.current_gid = None
        self.placed_nodes_order = []
        self.maps.reset_runtime()
        self.flow.reset_runtime()
        self.set_current_gid(self.remaining[0] if self.remaining else None)

    def set_remaining(self, remaining: List[GroupId], *, update_current_gid: bool = True) -> None:
        self.remaining = list(remaining)
        if update_current_gid:
            self.set_current_gid(self.remaining[0] if self.remaining else None)

    def place(self, *, gid: GroupId, placement: object) -> None:
        is_new = gid not in self.placed
        self.placements[gid] = placement
        self.placed.add(gid)
        if is_new:
            self.placed_nodes_order.append(gid)
            self.flow.invalidate_on_nodes_changed()
        else:
            self.flow.clear_flow_port_pairs()
        if gid in self.remaining:
            self.remaining.remove(gid)
        self.maps.paint_placement(placement)  # type: ignore[arg-type]
        self.flow.upsert_io(
            gid=gid,
            placement=placement,
            nodes=self.placed_nodes_order,
        )
        self.set_current_gid(self.remaining[0] if self.remaining else None)

    def step(
        self,
        *,
        apply: bool,
        gid: Optional[GroupId] = None,
        placement: Optional[object] = None,
    ) -> None:
        self.step_count += 1
        if not bool(apply):
            return
        if gid is None or placement is None:
            raise ValueError("step(apply=True) requires gid and placement")
        self.place(
            gid=gid,
            placement=placement,
        )

    def placed_nodes(self) -> List[GroupId]:
        return list(self.placed_nodes_order)

    def io_tensors(self) -> Tuple[List[GroupId], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.flow.io_tensors(self.placed_nodes_order)

    def build_flow_w(self) -> torch.Tensor:
        return self.flow.build_flow_w(self.placed_nodes_order)

    def build_delta_flow_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.flow.build_delta_flow_weights(self.current_gid, self.placed_nodes_order)

    def build_delta_flow_weights_for(self, gid: Optional[GroupId]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.flow.build_delta_flow_weights(gid, self.placed_nodes_order)

    def clear_flow_port_pairs(self) -> None:
        self.flow.clear_flow_port_pairs()

    def set_flow_port_pairs(self, pairs: Dict[Tuple[GroupId, GroupId], Tuple[Tuple[float, float], Tuple[float, float]]]) -> None:
        self.flow.set_flow_port_pairs(pairs)

    @property
    def flow_port_pairs(self) -> Dict[Tuple[GroupId, GroupId], Tuple[Tuple[float, float], Tuple[float, float]]]:
        return self.flow.flow_port_pairs

    @property
    def invalid_map(self) -> torch.Tensor:
        return self.maps.invalid

    @property
    def clear_invalid_map(self) -> torch.Tensor:
        return self.maps.clear_invalid

    def get_placement(self, gid: GroupId) -> object:
        p = self.placements.get(gid, None)
        if p is None:
            raise KeyError(f"placement missing for gid={gid!r}")
        return p

    def placed_bbox(self) -> Tuple[float, float, float, float]:
        return self.maps.placed_bbox()
