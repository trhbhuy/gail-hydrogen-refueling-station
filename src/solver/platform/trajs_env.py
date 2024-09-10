import os
import logging
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

from .. import config as cfg
from ..methods.data_loader import load_data
from .util import scaler_loader
from utils.preprocessing_util import data_loader

# Define base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATASET_DIR = os.path.join(BASE_DIR, 'data', 'generated')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HydrogenEnv(gym.Env):
    def __init__(self, is_train: bool = True):
        """Initialize the microgrid environment."""
        self.is_train = is_train
        self.T_num = cfg.T_NUM

        # Load the simulation data
        self.data = load_data(is_train=self.is_train)
        self.num_scenarios = len(self.data['p_pv_max']) // self.T_num
        logging.info(f"Number of scenarios: {self.num_scenarios}")

        self.unused_scenarios = list(range(self.num_scenarios))
        random.shuffle(self.unused_scenarios)

        # Load the dataset
        self.dataset = data_loader(os.path.join(DATASET_DIR, 'dataset.pkl'))

        # Load state and action scalers
        self.state_scaler, self.action_scaler = scaler_loader()

        # Define observation space (normalized to [0, 1])
        observation_dim = 5
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(observation_dim,), dtype=np.float32)

        # Define action space (normalized to [-1, 1])
        action_dim = 3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # Initialize variables for scenarios
        self._reset_unused_scenarios()

    def _reset_unused_scenarios(self):
        """Shuffle and reset the unused scenario list."""
        self.unused_scenarios = list(range(self.num_scenarios))
        random.shuffle(self.unused_scenarios)

    def reset(self, seed, options=None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)

        if not self.unused_scenarios:
            self._reset_unused_scenarios()

        # Pop a day from the list for use as the scenario seed
        self.scenario_seed = self.unused_scenarios.pop()
        self.time_step = 0

        # logging.info(f"Scenario seed selected: {self.scenario_seed}, unused scenarios left: {len(self.unused_scenarios)}")

        # Initialize state
        index = self._get_index(self.scenario_seed, self.time_step)
        initial_state = self.dataset['data_seq'][index]
        self.state = self.state_scaler.transform([initial_state])[0].astype(np.float32)

        return self.state, {}

    def step(self, action):
        """Take an action and return the next state, reward, and termination status."""
        current_state = self.state_scaler.inverse_transform([self.state])[0].astype(np.float32)
        time_step = int(np.round(current_state[0]))

        # Fetch the data for the current time step
        base_idx = self._get_index(self.scenario_seed, time_step)
        reward = self.dataset['reward'][base_idx]
        next_state, terminated = self._get_obs(time_step)

        # Update the state for the next step
        self.state = self.state_scaler.transform([next_state])[0].astype(np.float32)
        
        return self.state, reward, terminated, False, {}

    def _get_obs(self, time_step):
        """Prepare the next state and determine if the episode has terminated."""
        # Increment the time step.
        time_step += 1

        # Determine if the episode has terminated.
        terminated = time_step >= self.T_num

        # Prepare the next state if the episode is ongoing.
        if not terminated:
            base_idx = self._get_index(self.scenario_seed, time_step)
            next_state = self.dataset['data_seq'][base_idx]
        else:
            next_state = np.array([time_step, 0, 0, 0, 0], dtype=np.float32)

        return next_state, terminated

    def get_action(self, state):
        """Return a normalized action for the current state."""
        current_state = self.state_scaler.inverse_transform(state)[0].astype(np.float32)
        time_step = int(np.round(current_state[0]))

        # Fetch state values for the current scenario and hour
        base_idx = self._get_index(self.scenario_seed, time_step)
        action = self.dataset['label'][base_idx]
                
        # Normalize the action
        action_scaled = self.action_scaler.transform([action]).astype(np.float32)

        return action_scaled

    def _get_index(self, scenario: int, time_step: int) -> int:
        """Get index for scenario data based on time step and scenario seed."""
        return scenario * self.T_num + time_step