"""Backward-compatibility shim.

``CandidateSet`` is now an alias for ``envs.action_space.ActionSpace``.
New code should import ``ActionSpace`` directly from ``envs.action_space``.
"""
from __future__ import annotations

from envs.action_space import ActionSpace

# backward-compat alias
CandidateSet = ActionSpace

__all__ = ["CandidateSet", "ActionSpace"]
