"""Dynamic group generation using block-based placement.

블록 단위로 배치 가능 위치를 검사하고, cost가 낮은 곳부터 채워서
동적 그룹을 생성합니다.

사용법:
    generator = DynamicGroupGenerator(env)
    result = generator.generate(
        unit_w=8,
        unit_h=20,
        clearance_w=2,
        clearance_h=3,
        target_area=500,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import torch
import torch.nn.functional as F


@dataclass
class DynamicGroup:
    """동적 그룹 생성 결과."""
    cells: Set[Tuple[int, int]]  # 전체 영역 셀 집합 (unit + clearance)
    unit_cells: Set[Tuple[int, int]]  # unit 본체 셀 집합
    unit_positions: List[Tuple[int, int, int]]  # 각 unit의 (x, y, rotation)
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    area: int  # 전체 면적 (셀 개수)
    unit_w: int
    unit_h: int
    clearance_w: int
    clearance_h: int
    num_units: int  # 배치된 unit 수
    
    @property
    def block_w(self) -> int:
        return self.unit_w + 2 * self.clearance_w
    
    @property
    def block_h(self) -> int:
        return self.unit_h + 2 * self.clearance_h
    
    def to_mask(self, grid_width: int, grid_height: int) -> torch.Tensor:
        """전체 영역을 bool mask로 변환."""
        mask = torch.zeros((grid_height, grid_width), dtype=torch.bool)
        for x, y in self.cells:
            if 0 <= x < grid_width and 0 <= y < grid_height:
                mask[y, x] = True
        return mask
    
    def to_unit_mask(self, grid_width: int, grid_height: int) -> torch.Tensor:
        """unit 본체만 bool mask로 변환."""
        mask = torch.zeros((grid_height, grid_width), dtype=torch.bool)
        for x, y in self.unit_cells:
            if 0 <= x < grid_width and 0 <= y < grid_height:
                mask[y, x] = True
        return mask


class DynamicGroupGenerator:
    """블록 기반 동적 그룹 생성기.
    
    사용법:
        generator = DynamicGroupGenerator(env)
        result = generator.generate(block_w=10, block_h=10, target_area=500)
    """
    
    def __init__(self, env):
        """
        Args:
            env: FactoryLayoutEnv 인스턴스
        """
        self.env = env
        self.grid_width = env.grid_width
        self.grid_height = env.grid_height
        self.device = env.device
        
        # 유효 영역 (배치 가능한 셀)
        self._valid = ~(env._occ_invalid | env._static_invalid)
    
    def generate(
        self,
        unit_w: int,
        unit_h: int,
        clearance_w: int,
        clearance_h: int,
        target_area: int,
        allow_rotation: bool = False,
        rotation_mode: str = 'mixed',
        gid: Optional[str] = None,
    ) -> Optional[DynamicGroup]:
        """블록 단위 배치로 동적 그룹 생성.
        
        stride_x = unit_w (가로로 unit만큼 이동, clearance 겹침)
        stride_y = unit_h + clearance_h (세로로 unit + clearance만큼 이동)
        
        Args:
            unit_w: unit 본체 가로 크기
            unit_h: unit 본체 세로 크기
            clearance_w: 좌우 여백 (한쪽 기준)
            clearance_h: 상하 여백 (한쪽 기준)
            target_area: 목표 면적 (셀 개수)
            allow_rotation: 90도 회전 허용 여부
            rotation_mode: 'mixed' (cost 순 혼합) 또는 'sequential' (원본 먼저, 회전 나중)
            gid: (선택) cost 계산용 그룹 ID
        
        Returns:
            DynamicGroup 또는 None (실패 시)
        """
        H, W = self.grid_height, self.grid_width
        
        # ===== orientation 설정 =====
        # orientation 0: 원본 (unit_w x unit_h)
        # orientation 1: 90도 회전 (unit_h x unit_w)
        # stride는 회전과 무관하게 원본 기준으로 동일 유지
        stride_x = unit_w
        stride_y = unit_h + clearance_h
        
        orientations = [
            {
                'rot': 0,
                'unit_w': unit_w, 'unit_h': unit_h,
                'clearance_w': clearance_w, 'clearance_h': clearance_h,
                'block_w': unit_w + 2 * clearance_w,
                'block_h': unit_h + 2 * clearance_h,
            }
        ]
        
        if allow_rotation and unit_w != unit_h:
            orientations.append({
                'rot': 90,
                'unit_w': unit_h, 'unit_h': unit_w,  # 뒤집힘
                'clearance_w': clearance_h, 'clearance_h': clearance_w,  # 뒤집힘
                'block_w': unit_h + 2 * clearance_h,
                'block_h': unit_w + 2 * clearance_w,
            })
        
        # ===== 모든 orientation에서 가능한 위치 수집 =====
        all_candidates = []  # (cost, bx, by, orientation_idx)
        
        valid_float = self._valid.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        
        for ori_idx, ori in enumerate(orientations):
            block_w = ori['block_w']
            block_h = ori['block_h']
            # stride는 위에서 정의한 원본 기준 값 사용
            
            if block_h > H or block_w > W:
                continue
            
            # conv2d로 배치 가능 위치 검사
            kernel = torch.ones((1, 1, block_h, block_w), device=self.device)
            valid_conv = F.conv2d(
                valid_float, kernel, padding=0, stride=(stride_y, stride_x)
            ).squeeze()
            
            if valid_conv.dim() == 0:
                valid_conv = valid_conv.unsqueeze(0).unsqueeze(0)
            elif valid_conv.dim() == 1:
                valid_conv = valid_conv.unsqueeze(0)
            
            placeable = (valid_conv == block_w * block_h)
            
            if not placeable.any():
                continue
            
            placeable_indices = placeable.nonzero(as_tuple=False)
            if placeable_indices.numel() == 0:
                continue
            
            grid_y = placeable_indices[:, 0]
            grid_x = placeable_indices[:, 1]
            
            # offset 계산 (회전 시 그리드 정렬용)
            # 원본 unit 본체 시작: (clearance_w, clearance_h)
            # 회전 unit 본체 시작: (ori['clearance_w'], ori['clearance_h'])
            # offset = 원본 clearance - 현재 orientation clearance
            offset_x = clearance_w - ori['clearance_w']
            offset_y = clearance_h - ori['clearance_h']
            
            pos_x = grid_x * stride_x + offset_x
            pos_y = grid_y * stride_y + offset_y
            
            # offset 적용 후 유효성 재검사 (범위 및 _valid 체크)
            valid_mask = (pos_x >= 0) & (pos_y >= 0) & \
                         (pos_x + block_w <= W) & (pos_y + block_h <= H)
            
            # 각 위치에서 블록 전체가 _valid인지 체크
            valid_positions = []
            for i in range(len(pos_x)):
                if not valid_mask[i]:
                    continue
                bx, by = int(pos_x[i].item()), int(pos_y[i].item())
                # 블록 영역이 모두 valid인지 확인
                if self._valid[by:by+block_h, bx:bx+block_w].all():
                    valid_positions.append(i)
            
            if not valid_positions:
                continue
            
            # 유효한 위치만 추출
            valid_indices = torch.tensor(valid_positions, device=self.device)
            pos_x = pos_x[valid_indices]
            pos_y = pos_y[valid_indices]
            
            # cost 계산
            if gid is not None and gid in self.env.groups:
                center_x = pos_x.float() + block_w / 2.0
                center_y = pos_y.float() + block_h / 2.0
                rot_tensor = torch.full_like(pos_x, ori['rot'])
                
                costs = self.env.estimate_delta_obj(
                    gid=gid,
                    x=center_x,
                    y=center_y,
                    rot=rot_tensor,
                )
            else:
                cx, cy = W / 2.0, H / 2.0
                center_x = pos_x.float() + block_w / 2.0
                center_y = pos_y.float() + block_h / 2.0
                costs = torch.abs(center_x - cx) + torch.abs(center_y - cy)
            
            # 후보 추가
            for i in range(len(pos_x)):
                all_candidates.append((
                    costs[i].item(),
                    pos_x[i].item(),
                    pos_y[i].item(),
                    ori_idx
                ))
        
        if not all_candidates:
            return None
        
        # 정렬
        if rotation_mode == 'sequential':
            # 원본(ori_idx=0) 먼저, 그 다음 회전(ori_idx=1), 각각 cost 순
            all_candidates.sort(key=lambda x: (x[3], x[0]))
        else:
            # mixed: cost 순으로 혼합 배치
            all_candidates.sort(key=lambda x: x[0])
        
        # ===== 블록 배치 =====
        cells: Set[Tuple[int, int]] = set()
        unit_cells: Set[Tuple[int, int]] = set()
        unit_positions: List[Tuple[int, int, int]] = []  # (x, y, rotation)
        unit_used = torch.zeros((H, W), dtype=torch.bool, device=self.device)
        num_units = 0
        
        for cost, bx, by, ori_idx in all_candidates:
            if len(cells) >= target_area:
                break
            
            ori = orientations[ori_idx]
            uw = ori['unit_w']
            uh = ori['unit_h']
            cw = ori['clearance_w']
            ch = ori['clearance_h']
            bw = ori['block_w']
            bh = ori['block_h']
            rot = ori['rot']
            
            bx, by = int(bx), int(by)
            
            # unit 본체 영역
            ux = bx + cw
            uy = by + ch
            
            # 범위 확인
            if ux + uw > W or uy + uh > H:
                continue
            
            # unit 영역 겹침 확인
            unit_region = unit_used[uy:uy+uh, ux:ux+uw]
            if unit_region.any():
                continue
            
            # 블록 배치 (전체 영역)
            for dy in range(bh):
                for dx in range(bw):
                    cx, cy = bx + dx, by + dy
                    if 0 <= cx < W and 0 <= cy < H:
                        cells.add((cx, cy))
            
            # unit 본체 배치 및 사용 표시
            for dy in range(uh):
                for dx in range(uw):
                    cx, cy = ux + dx, uy + dy
                    if 0 <= cx < W and 0 <= cy < H:
                        unit_cells.add((cx, cy))
                        unit_used[cy, cx] = True
            
            # unit 위치 저장 (회전 정보 포함)
            unit_positions.append((ux, uy, rot))
            num_units += 1
        
        if not cells:
            return None
        
        # bbox 계산
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        return DynamicGroup(
            cells=cells,
            unit_cells=unit_cells,
            unit_positions=unit_positions,
            bbox=(x_min, y_min, x_max + 1, y_max + 1),
            area=len(cells),
            unit_w=unit_w,
            unit_h=unit_h,
            clearance_w=clearance_w,
            clearance_h=clearance_h,
            num_units=num_units,
        )
    
    def generate_multiple(
        self,
        count: int,
        unit_w: int,
        unit_h: int,
        clearance_w: int,
        clearance_h: int,
        target_area: int,
        allow_rotation: bool = False,
        rotation_mode: str = 'mixed',
        **kwargs,
    ) -> List[DynamicGroup]:
        """여러 개의 동적 그룹 생성 (서로 겹치지 않게).
        
        Args:
            count: 생성할 그룹 수
            unit_w: unit 본체 가로 크기
            unit_h: unit 본체 세로 크기
            clearance_w: 좌우 여백
            clearance_h: 상하 여백
            target_area: 각 그룹의 목표 면적
            allow_rotation: 90도 회전 허용 여부
            rotation_mode: 'mixed' 또는 'sequential'
            **kwargs: generate()에 전달할 추가 인자
        
        Returns:
            DynamicGroup 리스트
        """
        groups = []
        original_valid = self._valid.clone()
        
        for _ in range(count):
            group = self.generate(
                unit_w=unit_w,
                unit_h=unit_h,
                clearance_w=clearance_w,
                clearance_h=clearance_h,
                target_area=target_area,
                allow_rotation=allow_rotation,
                rotation_mode=rotation_mode,
                **kwargs,
            )
            if group is None:
                break
            
            groups.append(group)
            
            # 생성된 영역을 유효 영역에서 제외
            for x, y in group.cells:
                if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                    self._valid[y, x] = False
        
        # 원래 상태 복원
        self._valid = original_valid
        
        return groups


if __name__ == "__main__":
    """사용 예시 (시각화 포함)."""
    import sys
    sys.path.insert(0, str(__file__).rsplit("postprocess", 1)[0])
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from envs.json_loader import load_env
    
    # 1. env 로드
    config_path = "env_configs/clearance_03.json"
    print(f"[1] Loading: {config_path}")
    env = load_env(config_path).env
    env.reset()
    
    # 2. 동적 그룹 생성
    print("\n[2] Generating dynamic groups...")
    generator = DynamicGroupGenerator(env)
    
    # 파라미터 설정
    unit_w, unit_h = 8, 20
    clearance_w, clearance_h = 2, 3
    allow_rotation = True
    rotation_mode = 'sequential'  # 'mixed' 또는 'sequential'
    
    groups = generator.generate_multiple(
        count=2,
        unit_w=unit_w,
        unit_h=unit_h,
        clearance_w=clearance_w,
        clearance_h=clearance_h,
        target_area=1000,
        allow_rotation=allow_rotation,
        rotation_mode=rotation_mode,
    )
    
    print(f"    Generated {len(groups)} groups")
    print(f"    unit_size={unit_w}x{unit_h}, clearance=({clearance_w}, {clearance_h})")
    print(f"    allow_rotation={allow_rotation}, rotation_mode={rotation_mode}")
    for i, g in enumerate(groups):
        rot_counts = {}
        for _, _, rot in g.unit_positions:
            rot_counts[rot] = rot_counts.get(rot, 0) + 1
        print(f"    [Group {i+1}] area={g.area}, units={g.num_units}, rotations={rot_counts}")
    
    # 3. 시각화
    print("\n[3] Visualizing...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    H, W = generator.grid_height, generator.grid_width
    
    # --- 유효 영역 ---
    ax1 = axes[0]
    valid_img = generator._valid.cpu().numpy().astype(float)
    ax1.imshow(valid_img, cmap='Greens', origin='lower', aspect='equal')
    ax1.set_title('Valid Area (Green = Placeable)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # --- 생성된 그룹 (unit/clearance 구분) ---
    ax2 = axes[1]
    
    # 배경
    bg_img = np.zeros((H, W, 3))
    valid_np = generator._valid.cpu().numpy()
    bg_img[valid_np] = [0.9, 0.9, 0.9]  # 유효 = 밝은 회색
    bg_img[~valid_np] = [0.3, 0.3, 0.3]  # 무효 = 어두운 회색
    
    # 그룹별 색상 (clearance=연한색, unit=진한색)
    group_colors = [
        ([0.6, 0.9, 0.6], [0.4, 0.4, 0.4]),  # 연두 / 회색
        ([0.6, 0.7, 0.9], [0.3, 0.3, 0.6]),  # 연파랑 / 파랑
        ([0.9, 0.8, 0.6], [0.6, 0.4, 0.2]),  # 연주황 / 주황
    ]
    
    for i, group in enumerate(groups):
        clearance_color, unit_color = group_colors[i % len(group_colors)]
        
        # clearance 영역 (전체 - unit)
        clearance_cells = group.cells - group.unit_cells
        for x, y in clearance_cells:
            if 0 <= x < W and 0 <= y < H:
                bg_img[y, x] = clearance_color
        
        # unit 본체
        for x, y in group.unit_cells:
            if 0 <= x < W and 0 <= y < H:
                bg_img[y, x] = unit_color
    
    ax2.imshow(bg_img, origin='lower', aspect='equal')
    ax2.set_title(f'Generated Groups ({len(groups)}) - Unit(dark) / Clearance(light)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    # 셀 경계 그리기 (unit 본체 경계)
    for i, group in enumerate(groups):
        for ux, uy, rot in group.unit_positions:
            # 회전에 따라 unit 크기 결정
            if rot == 0:
                uw, uh = group.unit_w, group.unit_h
            else:  # 90도 회전
                uw, uh = group.unit_h, group.unit_w
            
            # unit 본체 경계 (검은 테두리)
            rect = plt.Rectangle(
                (ux - 0.5, uy - 0.5), 
                uw, uh,
                linewidth=1.5, edgecolor='black', facecolor='none'
            )
            ax2.add_patch(rect)
    
    # 범례
    legend_patches = []
    for i, g in enumerate(groups):
        clearance_color, unit_color = group_colors[i % len(group_colors)]
        legend_patches.append(
            mpatches.Patch(color=unit_color, label=f'G{i+1} unit ({g.num_units} units)')
        )
        legend_patches.append(
            mpatches.Patch(color=clearance_color, label=f'G{i+1} clearance')
        )
    if legend_patches:
        ax2.legend(handles=legend_patches, loc='upper right', fontsize=7)
    
    plt.tight_layout()
    plt.savefig('dynamic_group_result.png', dpi=150)
    print("    Saved: dynamic_group_result.png")
    plt.show()
