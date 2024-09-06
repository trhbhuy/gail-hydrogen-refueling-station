# src/train_gail.py

import os
import argparse
import numpy as np
import random
import pathlib
import logging

import torch
import torch.nn as nn

import stable_baselines3 as sb3

from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.policies.serialize import save_stable_model
# from imitation.scripts.train_adversarial import save
from imitation.data import serialize
from imitation.util import logger

from solver.platform.hrs_env import HRSEnv

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_option():
    """
    Parse command-line arguments for the training script.
    """
    parser = argparse.ArgumentParser(description='Train a GAIL model.')

    # Adversarial hyperparameters
    parser.add_argument('--adversarial_algo', type=str, choices=['gail', 'airl'], default='gail')
    parser.add_argument('--demo_batch_size', type=int, default=1024)
    parser.add_argument('--gen_replay_buffer_capacity', type=int, default=512)
    parser.add_argument('--n_disc_updates_per_round', type=int, default=8)
    parser.add_argument('--total_timesteps', type=int, default=50_000)

    # Deep RL hyperparameters
    parser.add_argument('--gen_algo', type=str, choices=['ppo', 'ddpg', 'trpo'], default='ppo')
    parser.add_argument('--policy', type=str, choices=['MlpPolicy', 'CnnPolicy', 'MultiInputPolicy'], default='MlpPolicy')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--learning_rate', type=float, default=0.00015)
    parser.add_argument('--ent_coef', type=float, default=0.02)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--gae_lambda', type=float, default=0.98)
    parser.add_argument('--max_grad_norm', type=float, default=0.7)
    parser.add_argument('--vf_coef', type=float, default=0.75)
    parser.add_argument('--net_arch_type', type=str, choices=['tiny', 'small', 'medium'], default="medium")
    parser.add_argument('--ortho_init', type=bool, default=True)
    parser.add_argument('--activation_fn_name', type=str, choices=['tanh', 'relu'], default='relu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rid', type=str, default='')

    opt = parser.parse_args()

    if opt.batch_size > opt.n_steps:
        opt.batch_size = opt.n_steps

    # Network architecture settings
    net_arch_map = {
        "tiny": dict(pi=[64], vf=[64]),
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }
    activation_fn_map = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU
    }

    opt.hyperparams = {
        "policy": opt.policy,
        "batch_size": opt.batch_size,
        "n_steps": opt.n_steps,
        "gamma": opt.gamma,
        "learning_rate": opt.learning_rate,
        "ent_coef": opt.ent_coef,
        "n_epochs": opt.n_epochs,
        "gae_lambda": opt.gae_lambda,
        "max_grad_norm": opt.max_grad_norm,
        "vf_coef": opt.vf_coef,
        "policy_kwargs": dict(
            net_arch=net_arch_map[opt.net_arch_type],
            activation_fn=activation_fn_map[opt.activation_fn_name],
            ortho_init=opt.ortho_init,
        ),
    }

    # Set up the logging directory
    log_dir = BASE_DIR / "logs"

    # Create a descriptive model name based on hyperparameters
    opt.model_name = f"{opt.adversarial_algo}_{opt.gen_algo}_lr{opt.learning_rate}_bs{opt.batch_size}_{opt.n_steps}n_steps"

    # Define the save folder path using pathlib
    opt.save_folder = log_dir / opt.model_name / "checkpoints" / "final"

    # Define the folder path of the expert trajectories
    opt.trajectories_path = BASE_DIR / 'data' / 'trajectories'

    opt.save_folder.mkdir(parents=True, exist_ok=True)

    return opt

def set_rl_algo(opt, env):
    """
    Initialize the learner.

    Args:
        opt (Namespace): Parsed command-line arguments.
        env: The environment to be used by the RL algorithm.

    Returns:
        The initialized RL algorithm model.
    """
    algo_cls = getattr(sb3, opt.gen_algo.upper(), None)

    if algo_cls is None:
        raise ValueError(f"Algorithm {opt.gen_algo} is not supported.")

    model = algo_cls(env=env, seed=opt.seed, **opt.hyperparams)
    logging.info(f"Initialized {opt.gen_algo.upper()} algorithm with provided hyperparameters.")
    return model

def save_model(trainer, save_path: pathlib.Path):
    """Save discriminator and generator."""
    logging.info('Saving model...')
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.reward_train, save_path / "reward_train.pt")
    torch.save(trainer.reward_test, save_path / "reward_test.pt")
    save_stable_model(
        save_path / "gen_policy",
        trainer.gen_algo,
        )
    logging.info(f'Model saved successfully at {save_path}')
    del trainer  # Explicitly delete to free up memory

def main():
    """
    Main function to train the model.
    """
    opt = parse_option()

    # Seed all random number generators for reproducibility
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)  # If using CUDA

    # Set up the environment
    env = HRSEnv(is_train = True)
    venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])  # Wrap a single environment -- only useful for simple testing like this

    # Load the expert trajectories
    expert_trajs = serialize.load(opt.trajectories_path)

    # Set up the RL learner
    learner = set_rl_algo(opt, venv)

    # Set up the logger
    # tmpdir = "Logger"
    # tmpdir = None  # No directory will be created
    # custom_logger = logger.configure(tmpdir, format_strs=["stdout", "log"])
    custom_logger = logger.configure(format_strs=[])

    # Set up the reward network
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
        )

    # Set up the GAIL trainer
    gail_trainer = GAIL(
        demonstrations=expert_trajs,
        demo_batch_size=opt.demo_batch_size,
        gen_replay_buffer_capacity=opt.gen_replay_buffer_capacity,
        n_disc_updates_per_round=opt.n_disc_updates_per_round,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        custom_logger=custom_logger
        )

    # Train the model
    gail_trainer.train(opt.total_timesteps)

    # Save the last model
    save_model(gail_trainer, opt.save_folder)

# python3 src/train_gail.py --adversarial_algo gail --demo_batch_size 1024 --gen_replay_buffer_capacity 512 --n_disc_updates_per_round 8 --gen_algo ppo --policy MlpPolicy --batch_size 64 --ent_coef 0.0 --learning_rate 0.0004 --gamma 0.95 --n_epochs 5
# Entry point
if __name__ == '__main__':
    main()