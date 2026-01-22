# coding=utf-8
# Copyright 2026.
#
# Fully-connected policy/value networks.
"""FC policy/value networks with mask application."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def _mask_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
  if mask is None:
    return logits
  if mask.dim() < logits.dim():
    mask = mask.unsqueeze(-2)
  almost_neg_inf = torch.ones_like(logits) * (-(2.0**32) + 1)
  return torch.where(mask.eq(1), logits, almost_neg_inf)


class PolicyNetwork(nn.Module):
  """FC policy network."""

  def __init__(self, action_dim: int, hidden_units: Optional[list[int]] = None):
    super().__init__()
    if hidden_units is None:
      hidden_units = [64, 64, 64, 64]

    layers = [nn.LazyLinear(hidden_units[0]), nn.ReLU()]
    for in_dim, out_dim in zip(hidden_units[:-1], hidden_units[1:]):
      layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_units[-1], action_dim))
    self._mlp = nn.Sequential(*layers)

  def forward(self, graph_embedding: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if graph_embedding.dim() > 2:
      graph_embedding = graph_embedding.flatten(start_dim=1)
    logits = self._mlp(graph_embedding)
    return _mask_logits(logits, mask)


class ValueNetwork(nn.Module):
  """FC value network."""

  def __init__(self, hidden_units: Optional[list[int]] = None):
    super().__init__()
    if hidden_units is None:
      hidden_units = [64, 64, 64, 64]

    layers = [nn.LazyLinear(hidden_units[0]), nn.ReLU()]
    for in_dim, out_dim in zip(hidden_units[:-1], hidden_units[1:]):
      layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_units[-1], 1))
    self._mlp = nn.Sequential(*layers)

  def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
    if graph_embedding.dim() > 2:
      graph_embedding = graph_embedding.flatten(start_dim=1)
    value = self._mlp(graph_embedding)
    return value.squeeze(-1)


if __name__ == "__main__":
  torch.manual_seed(0)
  action_dim = 32 * 32
  policy = PolicyNetwork(action_dim)
  value = ValueNetwork()

  graph_embedding = torch.rand(2, 64)
  mask = torch.ones(2, action_dim, dtype=torch.int32)
  logits = policy(graph_embedding, mask)
  value_out = value(graph_embedding)
  print("FC policy logits shape:", tuple(logits.shape))
  print("FC value shape:", tuple(value_out.shape))
