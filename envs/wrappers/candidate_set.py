from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass(frozen=True)
class CandidateSet:
    """Per-state action candidates (wrapper-native).

    Conventions:
    - `mask`: 1D torch.BoolTensor[N], where True means VALID/selectable.
    - `xyrot`: torch.Tensor[N,3] of (x_bl, y_bl, rot) in bottom-left integer coordinates.
    - `gid`: optional next group id (needed for BL->center conversion in visualization).
    """

    xyrot: torch.Tensor  # [N,3]
    mask: torch.Tensor  # bool [N]
    gid: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.xyrot, torch.Tensor) or not isinstance(self.mask, torch.Tensor):
            raise TypeError("CandidateSet.xyrot and CandidateSet.mask must be torch.Tensor")
        if self.xyrot.ndim != 2 or int(self.xyrot.shape[-1]) != 3:
            raise ValueError(f"CandidateSet.xyrot must have shape [N,3], got {tuple(self.xyrot.shape)}")
        if self.mask.ndim != 1 or int(self.mask.shape[0]) != int(self.xyrot.shape[0]):
            raise ValueError(
                f"CandidateSet.mask must have shape [N], got {tuple(self.mask.shape)} for N={int(self.xyrot.shape[0])}"
            )
        if self.mask.dtype != torch.bool:
            raise TypeError(f"CandidateSet.mask must be torch.bool, got {self.mask.dtype}")

