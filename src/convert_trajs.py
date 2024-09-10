import os
import argparse
import numpy as np
import logging
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import serialize

from solver.platform.trajs_env import HydrogenEnv

# Define base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    """
    Parse command-line arguments for the generating trajectories script.
    """
    parser = argparse.ArgumentParser(description='Arguments for testing the trained model.')
    parser.add_argument('--base_dir', type=str, default=BASE_DIR, help='Base directory for the project')
    parser.add_argument('--data_dir', type=str, default='data', help='Relative path to the data directory')
    parser.add_argument('--trajs_dir', type=str, default='trajectories', help='Subdirectory for trajectories')
    parser.add_argument('--num_train_scenarios', type=int, default=730, help='Number of test scenarios')

    return parser.parse_args()

def set_paths(args):
    """Set up paths for trajectories."""
    args.data_path = os.path.join(args.base_dir, args.data_dir, args.trajs_dir)
    os.makedirs(args.data_path, exist_ok=True)

def my_policy_loader(env, venv, sample_until, rng):
    """Generate and return trajectories by simulating the environment."""
    # Collect rollout tuples.
    trajectories = []

    # accumulator for incomplete trajectories
    trajectories_accum = rollout.TrajectoryAccumulator()
    obs = venv.reset()
    for env_idx, ob in enumerate(obs):
        # Seed with first obs only. Inside loop, we'll only add second obs from
        # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
        # get all observations, but they're not duplicated into "next obs" and
        # "previous obs" (this matters for, e.g., Atari, where observations are
        # really big).
        trajectories_accum.add_step(dict(obs=ob), env_idx)

    # Now, we sample until `sample_until(trajectories)` is true.
    # If we just stopped then this would introduce a bias towards shorter episodes,
    # since longer episodes are more likely to still be active, i.e. in the process
    # of being sampled from. To avoid this, we continue sampling until all epsiodes
    # are complete.
    #
    # To start with, all environments are active.
    active = np.ones(venv.num_envs, dtype=bool)
    if not isinstance(obs, np.ndarray):
        raise ValueError(
            "Dict/tuple observations are not supported."
            "Currently only np.ndarray observations are supported.",
        )

    state = None
    dones = np.zeros(venv.num_envs, dtype=bool)
    while np.any(active):
        acts = env.get_action(obs)
        # acts, state = get_actions(obs, state, dones)
        obs, rews, dones, infos = venv.step(acts)
        assert isinstance(obs, np.ndarray)

        # If an environment is inactive, i.e. the episode completed for that
        # environment after `sample_until(trajectories)` was true, then we do
        # *not* want to add any subsequent trajectories from it. We avoid this
        # by just making it never done.
        dones &= active

        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            acts,
            obs,
            rews,
            dones,
            infos,
        )
        trajectories.extend(new_trajs)

        if sample_until(trajectories):
            # Termination condition has been reached. Mark as inactive any
            # environments where a trajectory was completed this timestep.
            active &= ~dones

    # Note that we just drop partial trajectories. This is not ideal for some
    # algos; e.g. BC can probably benefit from partial trajectories, too.

    # Each trajectory is sampled i.i.d.; however, shorter episodes are added to
    # `trajectories` sooner. Shuffle to avoid bias in order. This is important
    # when callees end up truncating the number of trajectories or transitions.
    # It is also cheap, since we're just shuffling pointers.
    rng.shuffle(trajectories)  # type: ignore[arg-type]

    # Sanity checks.
    for trajectory in trajectories:
        n_steps = len(trajectory.acts)
        # extra 1 for the end
        exp_obs = (n_steps + 1,) + venv.observation_space.shape
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        exp_act = (n_steps,) + venv.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    return trajectories

def main(args):
    """Main function point for generating trajectories."""
    # Set paths
    set_paths(args)

    # Initialize environment and vectorized wrapper
    env = HydrogenEnv()
    venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])  # Wrap a single environment -- only useful for simple testing like this
    rng = np.random.default_rng(0)

    # Define the sampling condition based on time steps and episodes
    sample_until = rollout.make_sample_until(
        min_timesteps=int(args.num_train_scenarios*24), 
        min_episodes=args.num_train_scenarios
        )

    # Generate and save trajectories
    logging.info("Generating trajectories...")
    trajectories = my_policy_loader(env, venv, sample_until, rng)
    logging.info(f"Generated {len(trajectories)} trajectories.")

    serialize.save(args.data_path, trajectories)
    logging.info(f"Trajectories saved at {args.data_path}.")

#python3 src/convert_trajs.py
if __name__ == '__main__':
    args = parse_args()
    main(args)