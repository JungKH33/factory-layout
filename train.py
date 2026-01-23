# coding=utf-8
# Copyright 2026.
#
# PPO training (Tianshou) wired to envs/env.py + agents/alphachip/model.py (AlphaChip).
"""Train AlphaChip with Tianshou PPO using FactoryLayoutEnv observations.

Notes:
- env returns torch.Tensors; we convert them to numpy for Tianshou collectors.
- The AlphaChip model expects a PyG Batch (Data/Batch) + action_mask (passed as `mask_flat` into the model API).
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import uuid
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from tianshou.algorithm.modelfree.ppo import PPO  # type: ignore
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy  # type: ignore
from tianshou.algorithm.optim import AdamOptimizerFactory  # type: ignore
from tianshou.data.collector import Collector  # type: ignore
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnPolicyTrainer, OnPolicyTrainerParams  # type: ignore
from tianshou.utils.net.common import AbstractDiscreteActor, ModuleWithVectorOutput  # type: ignore

from torch_geometric.data import Batch, Data

from agents.alphachip.model import AlphaChip
from envs.wrappers.alphachip import AlphaChipWrapperEnv
from envs.env import FacilityGroup, FactoryLayoutEnv


def _as_torch(x: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
  if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _slice_batched_obs(obs: Dict[str, Any], b: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            out[k] = v[b]
        else:
            # scalar / object / already-sliced
            out[k] = v
    return out


def obs_to_pyg_and_mask(obs: Dict[str, Any], *, device: torch.device) -> Tuple[Batch, torch.Tensor]:
    """Convert env_new observation(s) into a PyG Batch + action_mask (int32).

    env_new keys used:
      - x: [N,F] or [B,N,F]
      - edge_index: [2,E] or [B,2,E]
      - edge_attr: [E,1] or [B,E,1]
      - current_node: [1] or [B,1]
      - netlist_metadata: [12] or [B,12]
      - action_mask: [A] or [B,A]
  """
    x = obs["x"]
    batched = isinstance(x, np.ndarray) and x.ndim == 3

    if batched:
        B = int(x.shape[0])
        data_list = []
        for b in range(B):
            o = _slice_batched_obs(obs, b)
            data_list.append(
                Data(
                    x=_as_torch(o["x"], device=device, dtype=torch.float32),
                    edge_index=_as_torch(o["edge_index"], device=device, dtype=torch.long),
                    edge_attr=_as_torch(o["edge_attr"], device=device, dtype=torch.float32),
                    netlist_metadata=_as_torch(o["netlist_metadata"], device=device, dtype=torch.float32).view(1, -1),
                    current_node=_as_torch(o["current_node"], device=device, dtype=torch.long).view(-1),
                )
            )
        batch = Batch.from_data_list(data_list)
        mask_flat = _as_torch(obs["action_mask"], device=device, dtype=torch.int32)
        if mask_flat.dim() == 1:
            mask_flat = mask_flat.view(B, -1)
        return batch, mask_flat

    data = Data(
        x=_as_torch(obs["x"], device=device, dtype=torch.float32),
        edge_index=_as_torch(obs["edge_index"], device=device, dtype=torch.long),
        edge_attr=_as_torch(obs["edge_attr"], device=device, dtype=torch.float32),
        netlist_metadata=_as_torch(obs["netlist_metadata"], device=device, dtype=torch.float32).view(1, -1),
        current_node=_as_torch(obs["current_node"], device=device, dtype=torch.long).view(-1),
    )
    batch = Batch.from_data_list([data])
    mask_flat = _as_torch(obs["action_mask"], device=device, dtype=torch.int32).view(1, -1)
    return batch, mask_flat


class _NumpyObsWrapper(gym.Wrapper):
    """Gymnasium wrapper: convert env_new torch obs -> numpy obs for Tianshou.

    Tianshou's VectorEnv stack expects `env.unwrapped` to exist (Gym API).
    """

    def __init__(self, env: FactoryLayoutEnv):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_np = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in obs.items()}
        return obs_np, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        obs_np = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in obs.items()}
        return obs_np, float(reward), bool(terminated), bool(truncated), info


class _IdentityPreprocess(ModuleWithVectorOutput):
    """Dummy preprocess net required by tianshou's Actor base classes.

    We bypass preprocess and directly consume structured dict observations.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x


class Actor(AbstractDiscreteActor):
    """Discrete actor that maps env_new obs(dict) -> flat logits [B, A]."""

    def __init__(self, model: AlphaChip, *, action_dim: int):
        super().__init__(output_dim=int(action_dim))
        self.model = model
        self._preprocess = _IdentityPreprocess(output_dim=int(action_dim))

    def get_preprocess_net(self) -> ModuleWithVectorOutput:  # type: ignore[override]
        return self._preprocess

    def forward(self, obs, state=None, info=None):  # type: ignore[override]
        data, mask_flat = obs_to_pyg_and_mask(obs, device=self.model.device)
        logits_flat, _value = self.model(data, mask_flat=mask_flat, is_eval=not self.training)
        return logits_flat, state


class Critic(nn.Module):
    def __init__(self, model: AlphaChip):
        super().__init__()
        self.model = model

    def forward(self, obs, state=None, info=None):  
        data, mask_flat = obs_to_pyg_and_mask(obs, device=self.model.device)
        _logits, value = self.model(data, mask_flat=mask_flat, is_eval=True)
        return value


def _build_env(*, grid_w: int, grid_h: int, coarse_grid: int, max_steps: int) -> _NumpyObsWrapper:
    dev_env = torch.device("cpu")
    groups = {
        "A": FacilityGroup(id="A", width=20, height=10, rotatable=True),
        "B": FacilityGroup(id="B", width=15, height=15, rotatable=True),
        "C": FacilityGroup(id="C", width=18, height=12, rotatable=True),
    }
    group_flow = {"A": {"B": 1.0}}

    forbidden = torch.zeros((grid_h, grid_w), dtype=torch.bool, device=dev_env)
    forbidden[0 : max(1, grid_h // 4), 0 : max(1, grid_w // 4)] = True

    engine = FactoryLayoutEnv(
        grid_width=grid_w,
        grid_height=grid_h,
        groups=groups,
        group_flow=group_flow,
        forbidden_mask=forbidden,
        device=dev_env,
        max_steps=max_steps,
    )
    env = AlphaChipWrapperEnv(engine=engine, coarse_grid=coarse_grid, rot=0)
    return _NumpyObsWrapper(env)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--train-env-num", type=int, default=4)
  parser.add_argument("--test-env-num", type=int, default=2)
  parser.add_argument("--epoch", type=int, default=10)
  parser.add_argument("--step-per-epoch", type=int, default=2000)
  parser.add_argument("--step-per-collect", type=int, default=200)
  parser.add_argument("--repeat-per-collect", type=int, default=4)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--gamma", type=float, default=0.99)
  parser.add_argument("--gae-lambda", type=float, default=0.95)
  parser.add_argument("--clip-ratio", type=float, default=0.2)
  parser.add_argument("--vf-coef", type=float, default=0.5)
  parser.add_argument("--ent-coef", type=float, default=0.0)
  parser.add_argument("--max-grad-norm", type=float, default=0.5)

    # Model/env dims
  parser.add_argument("--metadata-dim", type=int, default=12)
  parser.add_argument("--node-feature-dim", type=int, default=8)
    parser.add_argument("--max-grid-size", type=int, default=32)  # must be divisible by 16
    parser.add_argument("--grid-w", type=int, default=120)
    parser.add_argument("--grid-h", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=200)
  args = parser.parse_args()

    if args.max_grid_size % 16 != 0:
        raise ValueError("--max-grid-size must be divisible by 16 (deconv policy head constraint).")

    # ---- checkpoints (latest + best only) ----
    run_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{uuid.uuid4().hex[:6]}"
    ckpt_dir = (Path("results") / "checkpoints" / run_id).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_envs = DummyVectorEnv(
        [
            lambda: _build_env(
                grid_w=args.grid_w, grid_h=args.grid_h, coarse_grid=args.max_grid_size, max_steps=args.max_steps
            )
            for _ in range(args.train_env_num)
        ]
    )
    test_envs = DummyVectorEnv(
        [
            lambda: _build_env(
                grid_w=args.grid_w, grid_h=args.grid_h, coarse_grid=args.max_grid_size, max_steps=args.max_steps
            )
            for _ in range(args.test_env_num)
        ]
    )

    # NOTE: In tianshou 2.0.0, VectorEnv may expose action_space as a list (per-env).
    action_space_any = train_envs.action_space
    action_space = action_space_any[0] if isinstance(action_space_any, (list, tuple)) else action_space_any

    model = AlphaChip(
      metadata_dim=args.metadata_dim,
      node_feature_dim=args.node_feature_dim,
      max_grid_size=args.max_grid_size,
  )
    action_dim = int(action_space.n)
    actor = Actor(model, action_dim=action_dim)
  critic = Critic(model)

    policy = ProbabilisticActorPolicy(
      actor=actor,
        dist_fn=lambda logits: Categorical(logits=logits),
        deterministic_eval=False,
        action_space=action_space,
        observation_space=None,
        action_scaling=False,
    )

    algo = PPO(
        policy=policy,
      critic=critic,
        optim=AdamOptimizerFactory(lr=args.lr),
        eps_clip=args.clip_ratio,
      vf_coef=args.vf_coef,
      ent_coef=args.ent_coef,
      max_grad_norm=args.max_grad_norm,
        gae_lambda=args.gae_lambda,
        gamma=args.gamma,
        max_batchsize=args.batch_size,
  )

    train_collector = Collector(algo, train_envs)
    test_collector = Collector(algo, test_envs)

    def _save_ckpt(path: Path) -> None:
        payload = {
            "model": model.state_dict(),
            "meta": {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "grid_width": int(args.grid_w),
                "grid_height": int(args.grid_h),
                "coarse_grid": int(args.max_grid_size),
                "metadata_dim": int(args.metadata_dim),
                "node_feature_dim": int(args.node_feature_dim),
            },
        }
        torch.save(payload, str(path))

    def _compute_score_fn(collect_stats) -> float:
        # tianshou 2.0.0: CollectStats.returns_stat.mean after refresh
        if getattr(collect_stats, "returns_stat", None) is None:
            try:
                collect_stats.refresh_return_stats()
            except Exception:
                return float("-inf")
        rs = getattr(collect_stats, "returns_stat", None)
        return float(getattr(rs, "mean", float("-inf")))

    def _save_best_fn(_algorithm) -> None:
        # Called by trainer when best score improves.
        _save_ckpt(ckpt_dir / "best.ckpt")

    def _save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        _save_ckpt(ckpt_dir / "latest.ckpt")
        return str((ckpt_dir / "latest.ckpt").resolve())

    params = OnPolicyTrainerParams(
        training_collector=train_collector,
      test_collector=test_collector,
        max_epochs=args.epoch,
        epoch_num_steps=args.step_per_epoch,
        collection_step_num_env_steps=args.step_per_collect,
        update_step_num_repetitions=args.repeat_per_collect,
      batch_size=args.batch_size,
        test_step_num_episodes=1,
        compute_score_fn=_compute_score_fn,
        save_best_fn=_save_best_fn,
        save_checkpoint_fn=_save_checkpoint_fn,
        verbose=True,
        show_progress=True,
    )
    trainer = OnPolicyTrainer(algorithm=algo, params=params)
    stats = trainer.run()
    print("Training finished:", stats)
    print(f"checkpoint_dir={ckpt_dir}")


if __name__ == "__main__":
  main()
