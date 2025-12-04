"""
Ray entry point to train a REBEL policy within OpenRLHF.

This mirrors the role of ``train_ppo_ray.py`` for PPO: it sets up Ray,
parses CLI arguments, and launches a REBEL training run inside a Ray
worker process. The actual training loop is implemented in
``openrlhf.trainer.rebel_trainer``.

Example:

    python -m openrlhf.cli.train_rebel_ray \\
        --base_model meta-llama/Meta-Llama-3-8B-Instruct \\
        --reward_model openbmb/Eurus-RM-7b
"""

import argparse

import ray

from openrlhf.trainer.rebel_trainer import Args as RebelArgs, REBELTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train REBEL with Ray in OpenRLHF")

    # High-level experiment settings
    parser.add_argument("--exp_name", type=str, default="ultrafeedback_rebel")
    parser.add_argument("--seed", type=int, default=555134)

    # Models
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base policy model to finetune.",
    )
    parser.add_argument(
        "--reward_model",
        type=str,
        default="openbmb/Eurus-RM-7b",
        help="Reward model used for REBEL updates.",
    )

    # Data
    parser.add_argument(
        "--query_dataset",
        type=str,
        default="GitBag/ultrafeedback_llama3_eurus",
        help="Ultrafeedback-style dataset to use.",
    )

    # Output / logging
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/rebel_ultrafeedback",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="ultrafeedback",
        help="Wandb project name (if tracking is enabled).",
    )
    parser.add_argument(
        "--no_track",
        action="store_true",
        help="Disable Weights & Biases tracking.",
    )

    return parser.parse_args()


@ray.remote
def rebel_worker(cfg: RebelArgs):
    trainer = REBELTrainer(args=cfg)
    trainer.fit()


def train():
    cli_args = parse_args()

    # Build REBEL config dataclass
    cfg = RebelArgs()
    cfg.exp_name = cli_args.exp_name
    cfg.seed = cli_args.seed
    cfg.base_model = cli_args.base_model
    cfg.reward_model = cli_args.reward_model
    cfg.output_dir = cli_args.output_dir
    cfg.wandb_project_name = cli_args.wandb_project_name
    cfg.track = not cli_args.no_track
    cfg.task.query_dataset = cli_args.query_dataset

    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true"}})

    # Launch a single REBEL training worker
    ray.get(rebel_worker.remote(cfg))


if __name__ == "__main__":
    train()

