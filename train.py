"""RLlib 학습 실행 스크립트 (기본 설정)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from env import FactoryLayoutEnv


# ---------------------------------------------------------------------------
# 환경 및 학습 기본 설정
# ---------------------------------------------------------------------------

FACTORY_ENV_KWARGS: Dict[str, Any] = {
    "width": 20,
    "height": 10,
    "equipment_defs": [
        {
            "name": "Press-200",
            "class_index": 1,
            "width": 2,
            "height": 2,
            "clearance_left": 1,
            "clearance_right": 1,
            "clearance_top": 1,
            "clearance_bottom": 1,
        },
        {
            "name": "CNC-450",
            "class_index": 2,
            "width": 3,
            "height": 1,
            "clearance_left": 0,
            "clearance_right": 2,
            "clearance_top": 1,
            "clearance_bottom": 1,
        },
        {
            "name": "Inspection",
            "class_index": 3,
            "width": 2,
            "height": 2,
            "clearance_left": 1,
            "clearance_right": 1,
            "clearance_top": 1,
            "clearance_bottom": 1,
        },
    ],
    "to_place": {1: 6, 2: 4, 3: 12},
    "pre_placed": {1: [(1, 1)], 2: [(10, 5)]},
}

# 학습 루프 관련 상수
NUM_ITERATIONS: int = 10
DEFAULT_CHECKPOINT_DIR = Path("./checkpoints")
EXPORT_JSON_PATH = Path("trained_layout.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FactoryLayoutEnv with RLlib PPO")
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="체크포인트를 저장할 경로 (기본값: ./checkpoints)",
    )
    return parser.parse_args()


def _env_creator(env_config: Dict[str, Any]) -> FactoryLayoutEnv:
    env_kwargs = env_config.get("env_kwargs", FACTORY_ENV_KWARGS)
    return FactoryLayoutEnv(**env_kwargs)


def build_algorithm() -> Algorithm:
    register_env("factory_layout_env", _env_creator)

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    config = (
        PPOConfig()
        .environment(
            env="factory_layout_env",
            env_config={"env_kwargs": FACTORY_ENV_KWARGS},
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .training(
            gamma=0.99,
            lr=5e-4,
            train_batch_size=4000,
            model={"fcnet_hiddens": [256, 256]},
        )
        .framework("torch")
    )

    return config.build_algo()


def evaluate_layout(algo: Algorithm) -> Dict[str, Any]:
    """학습된 정책으로 1회 시뮬레이션을 수행하고 결과를 반환합니다."""

    env = FactoryLayoutEnv(**FACTORY_ENV_KWARGS)
    obs, _ = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = algo.compute_single_action(obs, explore=False)
        obs, _, terminated, truncated, _ = env.step(int(action))

    return env.export_layout()


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    export_json_path = EXPORT_JSON_PATH.resolve()

    algorithm: Optional[Algorithm] = None
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        algorithm = build_algorithm()

        for iteration in range(1, NUM_ITERATIONS + 1):
            result = algorithm.train()
            reward_mean = result["env_runners"]["episode_return_mean"]
            timesteps = result["num_env_steps_sampled_lifetime"]
            print(
                f"Iteration {iteration:02d} | "
                f"episode_return_mean={reward_mean} | "
                f"env_steps_sampled={timesteps}"
            )

        checkpoint_path = algorithm.save(str(checkpoint_dir))
        print(f"Checkpoint saved to {checkpoint_path}")

        layout = evaluate_layout(algorithm)
        with export_json_path.open("w", encoding="utf-8") as fp:
            json.dump(layout, fp, indent=2, ensure_ascii=False, sort_keys=True)
        print(f"Policy rollout saved to {export_json_path}")

    finally:
        if algorithm is not None:
            algorithm.stop()
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()

