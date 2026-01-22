from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
from torch_geometric.data import Data

from agents.alphachip.model import AlphaChip
from envs.env import FactoryLayoutEnv
from .candidate_set import CandidateSet


class Agent(Protocol):
    """Evaluate candidates for the given state."""

    def priors(self, *, env: FactoryLayoutEnv, obs: dict, candidates: CandidateSet) -> torch.Tensor:
        """Return float32 [N] non-negative priors (not necessarily normalized)."""

    def select_action(self, *, env: FactoryLayoutEnv, obs: dict, candidates: CandidateSet) -> int:
        """Return an action index in [0, N). Default: argmax(priors) over valid actions."""


def _obs_to_pyg_data(obs: dict) -> Data:
    return Data(
        x=obs["x"],
        edge_index=obs["edge_index"],
        edge_attr=obs["edge_attr"],
        netlist_metadata=obs["netlist_metadata"].view(1, -1),
        current_node=obs["current_node"].view(-1),
    )


@dataclass(frozen=True)
class GreedyAgent:
    """Greedy agent (parity with `agents/greedy.py` logic).

    - Select action: argmin(delta_obj) among valid candidates.
    - Priors: softmax(-delta_obj / prior_temperature) over valid candidates.
    """

    prior_temperature: float = 1.0

    def priors(self, *, env: FactoryLayoutEnv, obs: dict, candidates: CandidateSet) -> torch.Tensor:
        device = env.device
        N = int(candidates.xyrot.shape[0])
        gid = env.remaining[0] if env.remaining else None
        if gid is None:
            return torch.zeros((N,), dtype=torch.float32, device=device)

        priors = torch.zeros((N,), dtype=torch.float32, device=device)
        valid = candidates.mask

        valid_indices = [i for i in range(N) if bool(valid[i].item())]
        if not valid_indices:
            return priors

        # Collect delta objective scores (lower is better).
        scores = torch.empty((len(valid_indices),), dtype=torch.float32, device=device)
        for k, i in enumerate(valid_indices):
            x = float(candidates.xyrot[i, 0].item())
            y = float(candidates.xyrot[i, 1].item())
            rot = int(round(float(candidates.xyrot[i, 2].item())))
            scores[k] = float(env.estimate_delta_obj(gid=gid, x=x, y=y, rot=rot))

        temp = float(self.prior_temperature) if float(self.prior_temperature) > 0.0 else 1.0
        logits = -scores / temp
        logits = logits - torch.max(logits)
        probs = torch.exp(logits)
        probs_sum = float(probs.sum().item())
        if probs_sum <= 0.0:
            probs = torch.full_like(probs, 1.0 / float(len(valid_indices)))
        else:
            probs = probs / probs_sum

        for idx, p in zip(valid_indices, probs):
            priors[int(idx)] = float(p.item())
        return priors

    def select_action(self, *, env: FactoryLayoutEnv, obs: dict, candidates: CandidateSet) -> int:
        device = env.device
        N = int(candidates.xyrot.shape[0])
        gid = env.remaining[0] if env.remaining else None
        if gid is None or N <= 0:
            return 0

        valid = candidates.mask
        valid_indices = [i for i in range(N) if bool(valid[i].item())]
        if not valid_indices:
            return 0

        best_i = valid_indices[0]
        best_score = float("inf")
        for i in valid_indices:
            x = float(candidates.xyrot[i, 0].item())
            y = float(candidates.xyrot[i, 1].item())
            rot = int(round(float(candidates.xyrot[i, 2].item())))
            d = float(env.estimate_delta_obj(gid=gid, x=x, y=y, rot=rot))
            if d < best_score:
                best_score = d
                best_i = int(i)
        return int(best_i)


class AlphaChipCoarseAgent:
    """AlphaChip-based agent that produces priors for arbitrary candidates by coarse-cell lookup.

    AlphaChip's native action space is G*G (coarse grid). For a candidate (x,y),
    we map it to the nearest coarse cell index and use that cell's logit as prior.
    """

    def __init__(self, *, model: AlphaChip, coarse_grid: int):
        self.model = model
        self.coarse_grid = int(coarse_grid)

    def _xy_to_cell_index(self, *, env: FactoryLayoutEnv, x: float, y: float) -> int:
        g = int(self.coarse_grid)
        cell_w = float(env.grid_width) / float(g)
        cell_h = float(env.grid_height) / float(g)
        j = int(max(0, min(g - 1, math.floor(x / cell_w))))
        i = int(max(0, min(g - 1, math.floor(y / cell_h))))
        return i * g + j

    @torch.no_grad()
    def _coarse_logits(self, *, obs: dict, coarse_mask_valid: torch.Tensor) -> torch.Tensor:
        data = _obs_to_pyg_data(obs)
        mask_flat = coarse_mask_valid.view(1, -1).to(dtype=torch.int32)
        logits_flat, _value = self.model(data, mask_flat=mask_flat, is_eval=True)  # [1, G*G]
        return logits_flat[0]

    def priors(self, *, env: FactoryLayoutEnv, obs: dict, candidates: CandidateSet) -> torch.Tensor:
        device = env.device
        N = int(candidates.xyrot.shape[0])
        g = int(self.coarse_grid)
        # For now we allow all coarse cells (all-ones mask). CandidateSet.mask still gates invalid candidates.
        # (If you want tighter priors, we can plug CoarseSelector's conv-mask here.)
        coarse_mask_valid = torch.ones((g * g,), dtype=torch.bool, device=device)
        logits = self._coarse_logits(obs=obs, coarse_mask_valid=coarse_mask_valid)

        out = torch.full((N,), 0.0, dtype=torch.float32, device=device)
        for i in range(N):
            if not bool(candidates.mask[i].item()):
                out[i] = 0.0
                continue
            x = float(candidates.xyrot[i, 0].item())
            y = float(candidates.xyrot[i, 1].item())
            idx = self._xy_to_cell_index(env=env, x=x, y=y)
            out[i] = float(logits[idx].item())
        return out

    def select_action(self, *, env: FactoryLayoutEnv, obs: dict, candidates: CandidateSet) -> int:
        pri = self.priors(env=env, obs=obs, candidates=candidates)
        pri = pri.masked_fill(~candidates.mask, float("-inf"))
        return int(torch.argmax(pri).item()) if pri.numel() > 0 else 0


if __name__ == "__main__":
    import time

    from envs.json_loader import load_env
    from .selectors import TopKSelector  # type: ignore

    ENV_JSON = "env_configs/basic_01.json"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    env = loaded.env
    env.log = False
    obs, _ = env.reset(options=loaded.reset_kwargs)

    selector = TopKSelector(k=50, scan_step=10.0, quant_step=10.0, random_seed=0)
    candidates = selector.build(env)

    agent = GreedyAgent(prior_temperature=1.0)
    t0 = time.perf_counter()
    pri = agent.priors(env=env, obs=obs, candidates=candidates)
    a = agent.select_action(env=env, obs=obs, candidates=candidates)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print("[agents demo]")
    print(" env=", ENV_JSON, "next_gid=", (env.remaining[0] if env.remaining else None))
    print(" candidates.N=", int(candidates.mask.shape[0]), "valid=", int(candidates.mask.sum().item()))
    print(" pri.shape=", tuple(pri.shape), "chosen_action=", a, "elapsed_ms=", dt_ms)
    if int(candidates.mask.sum().item()) > 0:
        print(" chosen_xyrot=", candidates.xyrot[a].tolist(), "prior=", float(pri[a].item()))

