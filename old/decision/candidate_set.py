from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass(frozen=True)
class CandidateSet:
    """Per-state action candidates.

    Conventions (aligned with `FactoryLayoutEnv.step_masked` and existing wrappers):
    - `mask`: 1D torch.BoolTensor[N], where True means VALID/selectable.
    - `xyrot`: torch.FloatTensor[N,3] of raw (x, y, rot).
    """

    xyrot: torch.Tensor  # float32 [N,3]
    mask: torch.Tensor  # bool [N]
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


if __name__ == "__main__":
    import time

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    N = 5
    xyrot = torch.tensor([[10.0, 20.0, 0.0]] * N, device=dev)
    mask = torch.tensor([True, True, False, True, False], dtype=torch.bool, device=dev)
    t0 = time.perf_counter()
    cs = CandidateSet(xyrot=xyrot, mask=mask, meta={"demo": True})
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print("[CandidateSet demo]")
    print(" input.xyrot.shape=", tuple(xyrot.shape), "dtype=", xyrot.dtype, "device=", xyrot.device)
    print(" input.mask.shape =", tuple(mask.shape), "dtype=", mask.dtype, "valid=", int(mask.sum().item()))
    print(" output.meta     =", cs.meta)
    print(f" elapsed_ms={dt_ms:.3f}")

