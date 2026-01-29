from __future__ import annotations

"""Legacy RLlib training entrypoint (env_old-based).

This file is a preservation copy of the former project-root `main.py`.
"""

from datetime import datetime
import os
import uuid
from pathlib import Path
from typing import Dict, Tuple, Union
import shutil

import numpy as np
import torch

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

from envs.env_old import FactoryLayoutEnvOld, FacilityGroup
from envs.visualizer_old import plot_layout
from old.policies.topk_selector import TopKConfig, TopKSelector
from agents.ppo import CandidateRLModule


GroupId = Union[int, str]


class CandidateTrainingEnv(FactoryLayoutEnvOld):
    """FactoryLayoutEnvOld with internal Top-K candidate refresh."""

    def __init__(self, config: Dict):
        groups = config["groups"]
        group_flow = config["group_flow"]
        max_candidates = config["max_candidates"]
        grid_width = config["grid_width"]
        grid_height = config["grid_height"]
        grid_size = config.get("grid_size", 1.0)

        super().__init__(
            grid_width=grid_width,
            grid_height=grid_height,
            grid_size=grid_size,
            groups=groups,
            group_flow=group_flow,
            max_candidates=max_candidates,
            forbidden_mask=config.get("forbidden_mask"),
            column_mask=config.get("column_mask"),
            dry_mask=config.get("dry_mask"),
            weight_mask=config.get("weight_mask"),
            exclude_compact_types=config.get("exclude_compact_types"),
            seed=config.get("seed"),
            max_steps=config.get("max_steps"),
            reward_scale=config.get("reward_scale", 100.0),
        )

        self._selector = TopKSelector(config["topk_config"])
        self._current_gid: GroupId | None = None

    def _next_gid(self) -> GroupId | None:
        return self.remaining[0] if self.remaining else None

    def _refresh_candidates(self) -> None:
        self._current_gid = self._next_gid()
        if self._current_gid is None:
            self.set_candidates([])
            return
        candidates, _ = self._selector.generate(self, self._current_gid)
        self.set_candidates(candidates)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._refresh_candidates()
        obs = self._build_observation()
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        if not terminated and not truncated:
            self._refresh_candidates()
            obs = self._build_observation()
        return obs, reward, terminated, truncated, info


def _build_demo_groups() -> Tuple[Dict[GroupId, FacilityGroup], Dict[GroupId, Dict[GroupId, float]]]:
    groups = {
        "A": FacilityGroup(id="A", width=80, height=40),
        "B": FacilityGroup(id="B", width=60, height=60, rotatable=False),
        "C": FacilityGroup(id="C", width=50, height=30),
        "D": FacilityGroup(id="D", width=40, height=40),
        "E": FacilityGroup(id="E", width=70, height=50),
        "F": FacilityGroup(id="F", width=90, height=30, rotatable=False),
        "G": FacilityGroup(id="G", width=35, height=35),
        "H": FacilityGroup(id="H", width=120, height=50),
        "I": FacilityGroup(id="I", width=55, height=45),
    }
    flow = {
        "A": {"B": 1.0, "D": 0.6},
        "B": {"C": 1.0, "E": 0.4},
        "C": {"F": 0.7},
        "D": {"E": 0.5, "G": 0.3},
        "E": {"H": 0.6},
        "F": {"I": 0.4},
    }
    return groups, flow


def _safe_float(x, default=float("-inf")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def main() -> None:
    groups, flow = _build_demo_groups()

    # Training params
    num_iters = 5
    checkpoint_every = 1
    num_env_runners = 0  # (= old num_rollout_workers)
    train_batch_size = 128
    minibatch_size = 128
    num_epochs = 5
    lr = 5e-4
    results_root = Path("results")

    topk_config = TopKConfig(
        k=70,
        scan_step=5.0,
        quant_step=10.0,
        p_high=0.1,
        p_near=0.9,
        p_coarse=0.0,
        oversample_factor=4,
        diversity_ratio=0.0,
        min_diversity=10,
        random_seed=7,
    )

    env_config = {
        "grid_width": 500,
        "grid_height": 500,
        "grid_size": 1.0,
        "groups": groups,
        "group_flow": flow,
        "max_candidates": 70,
        "max_steps": 200,
        "topk_config": topk_config,
    }

    register_env("candidate_env", lambda cfg: CandidateTrainingEnv(cfg))

    run_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{uuid.uuid4().hex[:6]}"
    checkpoint_root = (results_root / "checkpoints" / run_id).resolve()
    tb_logs_root = (results_root / "tb_logs").resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    tb_logs_root.mkdir(parents=True, exist_ok=True)

    os.environ["RAY_RESULTS_DIR"] = str(tb_logs_root)

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    algo = (
        PPOConfig()
        .framework("torch")
        .environment(env="candidate_env", env_config=env_config)
        .env_runners(num_env_runners=num_env_runners)
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=CandidateRLModule,
                model_config={"cand_feat_dim": 11, "state_dim": 4, "hidden_size": 128},
            )
        )
        .training(
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            lr=lr,
            use_gae=True,
            use_critic=True,
        )
        .build_algo()
    )

    latest_dirs: list[Path] = []
    best_score = float("-inf")
    best_dir = (checkpoint_root / "best").resolve()

    for _ in range(num_iters):
        result = algo.train()

        it = int(result.get("training_iteration", 0))
        env_stats = result.get("env_runners", {})

        ret_mean = env_stats.get("episode_return_mean")
        ret_min = env_stats.get("episode_return_min")
        ret_max = env_stats.get("episode_return_max")
        len_mean = env_stats.get("episode_len_mean")

        print(
            f"iter={it} return_mean={_safe_float(ret_mean, 0.0):.3f} "
            f"return[min,max]=({_safe_float(ret_min, 0.0):.1f},{_safe_float(ret_max, 0.0):.1f}) "
            f"len_mean={_safe_float(len_mean, 0.0):.2f}"
        )

        if checkpoint_every > 0 and it % checkpoint_every == 0:
            tag = f"iter_{it:04d}"
            save_dir = (checkpoint_root / tag).resolve()
            save_dir.mkdir(parents=True, exist_ok=True)
            algo.save(str(save_dir))
            latest_dirs.append(save_dir)
            print("checkpoint_dir:", save_dir)

            while len(latest_dirs) > 3:
                old = latest_dirs.pop(0)
                if old.exists():
                    shutil.rmtree(old, ignore_errors=True)

            score = _safe_float(env_stats.get("episode_return_mean"), float("-inf"))
            if score > best_score:
                best_score = score
                if best_dir.exists():
                    shutil.rmtree(best_dir, ignore_errors=True)
                shutil.copytree(save_dir, best_dir)

                eval_env = CandidateTrainingEnv(env_config)
                obs, _ = eval_env.reset()
                terminated = truncated = False
                module = algo.get_module()
                while not (terminated or truncated):
                    obs_batch = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
                    fwd_out = module.forward_inference({"obs": obs_batch})
                    logits = fwd_out["action_dist_inputs"]
                    act = torch.argmax(logits, dim=-1).item()
                    obs, _, terminated, truncated, _ = eval_env.step(int(act))

                plot_layout(
                    eval_env,
                    show_masks=True,
                    candidates=None,
                    save_path=str((save_dir / f"layout_iter_{it:04d}.png").resolve()),
                    show=False,
                )

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()

