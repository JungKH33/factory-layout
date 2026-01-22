from __future__ import annotations

from dataclasses import dataclass

import torch

from envs.env import FactoryLayoutEnv
from actionspace.candidate_set import CandidateSet
from .base import Agent


@dataclass(frozen=True)
class GreedyAgent:
    """Greedy agent (parity with legacy `agents/greedy.py` logic).

    - Select action: argmin(delta_obj) among valid candidates.
    - Priors: softmax(-delta_obj / prior_temperature) over valid candidates.
    """

    prior_temperature: float = 1.0

    def policy(self, *, env: FactoryLayoutEnv, obs: dict, candidates: CandidateSet) -> torch.Tensor:
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

        scores = torch.empty((len(valid_indices),), dtype=torch.float32, device=device)
        for k, i in enumerate(valid_indices):
            x = float(candidates.xyrot[i, 0].item())
            y = float(candidates.xyrot[i, 1].item())
            rot = int(candidates.xyrot[i, 2].item())
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
            rot = int(candidates.xyrot[i, 2].item())
            d = float(env.estimate_delta_obj(gid=gid, x=x, y=y, rot=rot))
            if d < best_score:
                best_score = d
                best_i = int(i)
        return int(best_i)


if __name__ == "__main__":
    import time

    from envs.json_loader import load_env
    from actionspace.topk import TopKSelector

    ENV_JSON = "env_configs/basic_01.json"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    env = loaded.env
    env.log = False
    obs, _info = env.reset(options=loaded.reset_kwargs)

    selector = TopKSelector(k=50, scan_step=10.0, quant_step=10.0, random_seed=0)
    candidates = selector.build(env)
    agent = GreedyAgent(prior_temperature=1.0)

    t0 = time.perf_counter()
    pri = agent.policy(env=env, obs=obs, candidates=candidates)
    a = agent.select_action(env=env, obs=obs, candidates=candidates)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    valid_n = int(candidates.mask.sum().item())
    xyrot = candidates.xyrot[a].tolist() if int(candidates.xyrot.shape[0]) > 0 else [0, 0, 0]

    print("[agents.greedy demo]")
    print(" env=", ENV_JSON, "device=", device, "next_gid=", (env.remaining[0] if env.remaining else None))
    print(" action=", a, "valid_candidates=", valid_n, "xyrot=", xyrot, "prior=", (float(pri[a].item()) if pri.numel() > 0 else 0.0))
    print(f" elapsed_ms={dt_ms:.3f}")
