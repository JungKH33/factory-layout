from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

GroupId = Union[int, str]


@dataclass(frozen=True)
class EnvAction:
    """Normalized placement action payload.

    - gid=None means "place current env.get_state().remaining[0]".
    - gid is optional override; ordering agent typically controls `env.get_state().remaining`.
    - x/y are bottom-left integer grid coordinates.
    - rot is in degrees (multiples of 90 expected by env).
    """

    x: int
    y: int
    rot: int
    gid: Optional[GroupId] = None
