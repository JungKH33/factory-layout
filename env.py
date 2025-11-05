# Factory Layout Placement Environment
"""
FactoryLayoutEnv는 직사각형 공장 공간에 장비를 배치하는 Gymnasium 호환 환경입니다.

주요 특징
----------
1. **장비 사양 & 배치 계획** – ``equipment_defs`` 로 클래스별 폭/높이/이격/이름을 정의하고,
   ``to_place`` 에서 클래스 인덱스와 배치 수량을 별도로 지정합니다.
2. **사전 배치(placed)** – ``{class_index: [(x, y), ...]}`` 형태로 초기 좌표를 지정하면, 이후 에이전트는
   해당 영역을 침범할 수 없습니다.
3. **이격(클리어런스) 반영** – 장비 배치 시 본체뿐 아니라 이격이 요구하는 영역까지 사전에 차단하여
   안전 거리를 보장합니다.
4. **보상 구조** – 면적 활용, 동일 장비 클러스터링, 정렬 보상 등을 조합한 다중 목표 보상을 제공합니다.
5. **시각화** – 장비 수에 맞춰 자동으로 색상을 지정하며, 이격으로 인한 배치 불가 영역과 사전 배치된
   장비를 시각적으로 구분할 수 있습니다.

Dependencies
------------
- gymnasium
- numpy
- matplotlib
- Pillow
"""
from __future__ import annotations

import io
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from PIL import Image


@dataclass(frozen=True)
class EquipmentSpec:
    """장비 한 종류의 기본 사양."""

    name: str
    class_index: int
    width: int
    height: int
    clearance_left: int
    clearance_right: int
    clearance_top: int
    clearance_bottom: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EquipmentSpec":
        required_keys = {"name", "width", "height", "class_index"}
        missing = required_keys - data.keys()
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Equipment spec missing required keys: {missing_str}")

        class_index = int(data["class_index"])
        if class_index <= 0:
            raise ValueError("Equipment class_index must be a positive integer")

        width = int(data["width"])
        height = int(data["height"])
        if width <= 0 or height <= 0:
            raise ValueError("Equipment width and height must be positive")

        clearance_left = int(data.get("clearance_left", 0))
        clearance_right = int(data.get("clearance_right", 0))
        clearance_top = int(data.get("clearance_top", 0))
        clearance_bottom = int(data.get("clearance_bottom", 0))
        for value in (clearance_left, clearance_right, clearance_top, clearance_bottom):
            if value < 0:
                raise ValueError("Equipment clearance values must be non-negative")

        return cls(
            name=str(data["name"]),
            class_index=class_index,
            width=width,
            height=height,
            clearance_left=clearance_left,
            clearance_right=clearance_right,
            clearance_top=clearance_top,
            clearance_bottom=clearance_bottom,
        )

    @property
    def area(self) -> int:
        return self.width * self.height


class FactoryLayoutEnv(gym.Env):
    """Factory layout 환경.

    격자는 1x1 셀 단위로 구성됩니다. 각 스텝마다 하나의 장비를 선택된 셀에 배치하며,
    장비는 회전 없이 지정된 폭/높이를 그대로 사용합니다. 장비별 이격(좌/우/상/하)은
    배치 시 주변 금지 영역으로 반영되어 다른 장비와의 최소 거리를 보장합니다.

    Observation
    -----------
    납작하게(flatten) 펼친 ``(height * width,)`` 크기의 실수 배열.
        -1  – 이격(클리어런스) 영역
        0   – 비어 있는 배치 가능 셀
        >0  – 해당 클래스 인덱스를 차지한 장비

    Action
    ------
    Discrete ``height * width``.  액션 ``a``는 ``(row = a // width, col = a % width)``로 매핑.

    Reward
    ------
    ``reward = R_area + R_cluster + R_align``
        * *R_area*:  +area_weight * (장비 면적)
        * *R_cluster*: -cluster_weight * (같은 index 장비 간 평균 맨해튼 거리)
        * *R_align*: +align_weight (같은 index 장비가 같은 행/열에 있으면)
    유효하지 않은 배치는 ``-invalid_penalty``.

    Termination
    -----------
    모든 장비를 배치하거나, ``attempt_limit``(장비 총 수 * 2) 시도를 초과하면 종료합니다.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        width: int,
        height: int,
        equipment_defs: Iterable[Dict[str, Any]],
        to_place: Mapping[int, int] | Sequence[Dict[str, Any]],
        placed: Optional[Mapping[int, Sequence[Tuple[int, int]]]] = None,
        reward_cfg: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        if width <= 0 or height <= 0:
            raise ValueError("Factory dimensions must be positive")

        self.W = width
        self.H = height

        self.equipment_defs = equipment_defs
        self.template_by_class_index: Dict[int, EquipmentSpec] = {}
        for spec_data in equipment_defs:
            spec = EquipmentSpec.from_dict(spec_data)
            if spec.class_index in self.template_by_class_index:
                raise ValueError(
                    f"Duplicate equipment definition for class_index {spec.class_index}"
                )
            self.template_by_class_index[spec.class_index] = spec

        if not self.template_by_class_index:
            raise ValueError("At least one equipment definition is required")

        self.equipment_templates = list(self.template_by_class_index.values())
        self.max_class_index = max(self.template_by_class_index.keys())

        self.placed_positions = self._expand_placed(placed)

        self._base_to_place, self.initial_to_place_counts = self._expand_to_place(to_place)
        self.total_items = len(self._base_to_place)
        if self.total_items <= 0:
            raise ValueError("Total equipment to place must be positive")

        # Reward hyper-parameters
        default_reward_cfg = {
            "area_weight": 1.0,
            "cluster_weight": 0.1,
            "align_weight": 0.2,
            "invalid_penalty": 1.0,
        }
        self.reward_cfg = {**default_reward_cfg, **(reward_cfg or {})}

        # Spaces
        self.observation_space = spaces.Box(
            low=-1.0,
            high=float(max(self.max_class_index, 1)),
            shape=(self.H * self.W,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.H * self.W)

        # 동적 상태는 reset에서 초기화
        self.grid: np.ndarray
        self.blocked_mask: np.ndarray
        self.remaining_items: Deque[EquipmentSpec]
        self.placed_items: List[Dict[str, Any]] = []
        self.attempt_limit: int

        self.reset()

    # ---------------------------------------------------------------------
    # Environment API implementation
    # ---------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, **kwargs):  # noqa: D401
        super().reset(seed=seed)

        self.grid = np.zeros((self.H, self.W), dtype=np.int16)
        self.blocked_mask = np.zeros((self.H, self.W), dtype=bool)
        self.placed_items = []

        self._place_pre_placed_equipment()

        expanded_items = list(self._base_to_place)
        self.np_random.shuffle(expanded_items)
        self.remaining_items = deque(expanded_items)

        self.attempts = 0
        self.attempt_limit = len(expanded_items) * 2 if expanded_items else 0

        observation = self._get_obs()
        info: Dict[str, Any] = {}
        return observation, info

    def step(self, action: int):  # noqa: D401
        if not self.remaining_items:
            raise RuntimeError("No remaining items; call reset() first.")

        row, col = divmod(action, self.W)
        current_template = self.remaining_items[0]
        w, h = current_template.width, current_template.height

        self.attempts += 1
        reward = 0.0
        terminated = False
        truncated = False

        # 배치 가능 여부 확인
        if self._can_place(row, col, current_template):
            self._place(row, col, current_template)
            self.remaining_items.popleft()
            reward += self.reward_cfg["area_weight"] * current_template.area
            reward += self.reward_cfg["cluster_weight"] * (
                -self._mean_pairwise_dist(current_template.class_index)
            )
            if self._shares_row_or_col(row, col, current_template):
                reward += self.reward_cfg["align_weight"]
        else:
            reward -= self.reward_cfg["invalid_penalty"]

        if not self.remaining_items:
            terminated = True
        if self.attempt_limit and self.attempts >= self.attempt_limit:
            truncated = True

        observation = self._get_obs()
        info: Dict[str, Any] = {}
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        return self.grid.astype(np.float32, copy=True).reshape(-1)

    def _place_pre_placed_equipment(self) -> None:
        for equipment, x, y in self.placed_positions:
            row = y
            col = x

            if not self._can_place(row, col, equipment):
                raise ValueError(
                    f"Pre-placed equipment '{equipment.name}' cannot be placed at"
                    f" (row={row}, col={col}) due to overlap or bounds"
                )

            self._place(row, col, equipment, is_preplaced=True)

    def _can_place(self, r: int, c: int, template: EquipmentSpec) -> bool:
        w, h = template.width, template.height
        r0 = r - template.clearance_top
        c0 = c - template.clearance_left
        r1 = r + h + template.clearance_bottom
        c1 = c + w + template.clearance_right

        if r0 < 0 or c0 < 0 or r1 > self.H or c1 > self.W:
            return False

        body_x0, body_y0 = c, r
        body_x1, body_y1 = c + w, r + h
        clearance_x0, clearance_y0 = c0, r0
        clearance_x1, clearance_y1 = c1, r1

        for existing in self.placed_items:
            ex_x0 = int(existing["x"])
            ex_y0 = int(existing["y"])
            ex_x1 = ex_x0 + int(existing["width"])
            ex_y1 = ex_y0 + int(existing["height"])

            ex_clearance_x0 = ex_x0 - int(existing["clearance_left"])
            ex_clearance_y0 = ex_y0 - int(existing["clearance_top"])
            ex_clearance_x1 = ex_x1 + int(existing["clearance_right"])
            ex_clearance_y1 = ex_y1 + int(existing["clearance_bottom"])

            if self._rects_overlap(
                body_x0,
                body_y0,
                body_x1,
                body_y1,
                ex_clearance_x0,
                ex_clearance_y0,
                ex_clearance_x1,
                ex_clearance_y1,
            ):
                return False

            if self._rects_overlap(
                ex_x0,
                ex_y0,
                ex_x1,
                ex_y1,
                clearance_x0,
                clearance_y0,
                clearance_x1,
                clearance_y1,
            ):
                return False

        return True

    def _place(
        self,
        r: int,
        c: int,
        template: EquipmentSpec,
        *,
        is_preplaced: bool = False,
    ) -> None:
        w, h = template.width, template.height
        idx = template.class_index
        r0, r1, c0, c1 = self._expanded_bounds(r, c, template)
        self._mark_clearance(r0, r1, c0, c1)
        self.grid[r : r + h, c : c + w] = idx
        self.blocked_mask[r0:r1, c0:c1] = True
        self.placed_items.append(
            {
                "name": template.name,
                "class_index": idx,
                "x": c,
                "y": r,
                "width": w,
                "height": h,
                "is_preplaced": is_preplaced,
                "clearance_left": template.clearance_left,
                "clearance_right": template.clearance_right,
                "clearance_top": template.clearance_top,
                "clearance_bottom": template.clearance_bottom,
            }
        )

    def _mean_pairwise_dist(self, cls: int) -> float:
        ys, xs = np.where(self.grid == cls)
        if len(xs) <= 1:
            return 0.0
        coords = np.column_stack((ys, xs))
        # Compute pairwise Manhattan distances efficiently
        diffs = np.abs(coords[:, None, :] - coords[None, :, :])
        dists = diffs.sum(axis=-1)
        return dists[np.triu_indices(len(coords), k=1)].mean()

    def _shares_row_or_col(self, r: int, c: int, template: EquipmentSpec) -> bool:
        w, h = template.width, template.height
        cls = template.class_index
        ys, xs = np.where(self.grid == cls)
        if len(xs) == 0:
            return False
        # New block occupies rows [r, r+h) & cols [c, c+w)
        rows_new = set(range(r, r + h))
        cols_new = set(range(c, c + w))
        return any(y in rows_new for y in ys) or any(x in cols_new for x in xs)

    def _expanded_bounds(
        self, r: int, c: int, template: EquipmentSpec
    ) -> tuple[int, int, int, int]:
        r0 = r - template.clearance_top
        r1 = r + template.height + template.clearance_bottom
        c0 = c - template.clearance_left
        c1 = c + template.width + template.clearance_right
        return r0, r1, c0, c1

    @staticmethod
    def _rects_overlap(
        ax0: int,
        ay0: int,
        ax1: int,
        ay1: int,
        bx0: int,
        by0: int,
        bx1: int,
        by1: int,
    ) -> bool:
        return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1

    def _mark_clearance(self, r0: int, r1: int, c0: int, c1: int) -> None:
        subgrid = self.grid[r0:r1, c0:c1]
        subgrid[(subgrid == 0)] = -1

    def _expand_to_place(
        self, to_place: Mapping[int, int] | Sequence[Dict[str, Any]]
    ) -> Tuple[List[EquipmentSpec], Dict[int, int]]:
        expanded: List[EquipmentSpec] = []
        counts: Dict[int, int] = {}

        if isinstance(to_place, Mapping):
            items = ((int(k), int(v)) for k, v in to_place.items())
        else:
            items = []
            for entry in to_place:
                if not isinstance(entry, Mapping):
                    raise TypeError(
                        "Sequence-based to_place entries must be mappings with "
                        "'class_index' and 'count'"
                    )
                if "class_index" not in entry:
                    raise ValueError("to_place entry missing 'class_index'")
                class_index = int(entry["class_index"])
                count = int(entry.get("count", 1))
                items.append((class_index, count))

        for class_index, count in items:
            cls_idx = int(class_index)
            if cls_idx not in self.template_by_class_index:
                raise ValueError(
                    f"to_place references undefined class_index {cls_idx}"
                )
            if count < 0:
                raise ValueError("to_place count must be non-negative")
            expanded.extend([self.template_by_class_index[cls_idx]] * count)
            counts[cls_idx] = counts.get(cls_idx, 0) + count

        return expanded, counts

    def _expand_placed(
        self, placed: Optional[Mapping[int, Sequence[Tuple[int, int]]]]
    ) -> List[Tuple[EquipmentSpec, int, int]]:
        if not placed:
            return []

        expanded: List[Tuple[EquipmentSpec, int, int]] = []
        for class_index, positions in placed.items():
            cls_idx = int(class_index)
            if cls_idx not in self.template_by_class_index:
                raise ValueError(
                    f"pre_placed references undefined class_index {cls_idx}"
                )

            template = self.template_by_class_index[cls_idx]

            raw_positions = positions
            if raw_positions is None:
                continue

            if isinstance(raw_positions, (tuple, list)) and raw_positions and not isinstance(
                raw_positions[0], (tuple, list)
            ):
                iterable = [raw_positions]
            else:
                iterable = raw_positions

            for pos in iterable:
                if not isinstance(pos, (tuple, list)) or len(pos) != 2:
                    raise ValueError(
                        "Each pre_placed position must be a tuple/list of (x, y)"
                    )
                x, y = int(pos[0]), int(pos[1])
                expanded.append((template, x, y))

        return expanded

    def render(
        self,
        mode: str = "human",
        *,
        show_forbidden: bool = False,
        cmap_name: str = "tab20",
    ):
        unique_ids = sorted(int(v) for v in np.unique(self.grid) if v > 0)
        id_to_color_index: Dict[int, int] = {}

        if unique_ids:
            base_cmap = plt.get_cmap(cmap_name, len(unique_ids))
            colors: List[Any] = ["#ffffff"]
            for offset, equipment_id in enumerate(unique_ids):
                colors.append(base_cmap(offset))
                id_to_color_index[equipment_id] = offset + 1
        else:
            colors = ["#ffffff"]

        cmap = ListedColormap(colors)

        display_grid = np.zeros((self.H, self.W), dtype=int)
        for equipment_id, color_index in id_to_color_index.items():
            display_grid[self.grid == equipment_id] = color_index

        for item in self.placed_items:
            if not item.get("is_preplaced"):
                continue
            y0 = int(item["y"])
            y1 = y0 + int(item["height"])
            x0 = int(item["x"])
            x1 = x0 + int(item["width"])
            display_grid[y0:y1, x0:x1] = 0

        fig, ax = plt.subplots(figsize=(max(self.W / 2, 4), max(self.H / 2, 4)))
        ax.imshow(display_grid, cmap=cmap, origin="upper", interpolation="none")
        ax.set_xticks(np.arange(-0.5, self.W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.H, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Factory Layout — step {self.attempts}")

        for item in self.placed_items:
            if not item.get("is_preplaced"):
                continue
            cls_idx = item.get("class_index")
            color_index = id_to_color_index.get(cls_idx)
            base_color = cmap.colors[color_index] if color_index is not None else "#555555"
            rect = Rectangle(
                (item["x"] - 0.5, item["y"] - 0.5),
                item["width"],
                item["height"],
                facecolor="none",
                edgecolor=base_color,
                linewidth=0.0,
                hatch="///",
                zorder=5,
            )
            ax.add_patch(rect)

        if show_forbidden:
            forbidden_mask = self.blocked_mask & (self.grid == 0)
            if np.any(forbidden_mask):
                overlay = np.zeros((self.H, self.W), dtype=int)
                overlay[forbidden_mask] = 1
                overlay_cmap = ListedColormap(((0, 0, 0, 0), (1, 0, 0, 0.35)))
                ax.imshow(
                    overlay,
                    cmap=overlay_cmap,
                    origin="upper",
                    interpolation="none",
                )

        plt.tight_layout()

        if mode == "human":
            plt.show()
            plt.close(fig)
        elif mode == "rgb_array":
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return np.asarray(Image.open(buf))
        else:
            plt.close(fig)
            raise NotImplementedError(f"Unsupported render mode: {mode}")

    def export_layout(self) -> Dict[str, Any]:
        placed: Dict[int, List[List[int]]] = {}
        for item in self.placed_items:
            cls_idx = int(item["class_index"])
            placed.setdefault(cls_idx, []).append([int(item["x"]), int(item["y"])])

        return {
            "equipments": self.equipment_defs,
            "to_place": {
                int(cls_idx): int(count)
                for cls_idx, count in sorted(self.initial_to_place_counts.items())
            },
            "placed": placed,
        }


# ---------------------------------------------------------------------------
# Quick sanity check (run as module)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    equipments = [
        {
            "name": "Press-200",
            "class_index": 1,
            "width": 2,
            "height": 2,
            "clearance_left": 1,
            "clearance_right": 1,
            "clearance_top": 1,
            "clearance_bottom": 1,
        },
        {
            "name": "CNC-450",
            "class_index": 2,
            "width": 3,
            "height": 1,
            "clearance_left": 0,
            "clearance_right": 2,
            "clearance_top": 1,
            "clearance_bottom": 1,
        },
        {
            "name": "Inspection",
            "class_index": 3,
            "width": 2,
            "height": 2,
            "clearance_left": 1,
            "clearance_right": 1,
            "clearance_top": 1,
            "clearance_bottom": 1,
        },
    ]

    to_place = {1: 6, 2: 4, 3: 12}
    placed = {1: [(2, 2), (6, 2)]}

    env = FactoryLayoutEnv(
        width=20,
        height=10,
        equipment_defs=equipments,
        to_place=to_place,
        placed=placed,
    )
    print("Action Space:", env.action_space)
    print("Observation Space:", env.observation_space)
    obs, _ = env.reset()
    frames: List[Image.Image] = []
    frame = env.render(mode="rgb_array", show_forbidden=True)
    frames.append(Image.fromarray(frame))

    while env.remaining_items:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render(mode="rgb_array", show_forbidden=True)
        frames.append(Image.fromarray(frame))
        if terminated or truncated:
            break

    if frames:
        output_path = "random_layout.gif"
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=300,
            loop=0,
        )
        print(f"Random placement GIF saved to {output_path}")

    layout_path = "random_layout.json"
    with open(layout_path, "w", encoding="utf-8") as fp:
        json.dump(env.export_layout(), fp, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Layout exported to {layout_path}")
