import logging
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from .. import config as cfg
from ..methods.data_loader import load_data
from .util import scaler_loader, check_boundary_constraint, check_setpoint

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HydrogenEnv(gym.Env):
    def __init__(self, is_train: bool = True):
        """Initialize the microgrid environment."""
        self.is_train = is_train
        self._init_params()
    
        # Load the simulation data
        self.data = load_data(is_train=self.is_train)
        self.num_scenarios = len(self.data['p_pv_max']) // self.T_num
        logging.info(f"Number of scenarios: {self.num_scenarios}")

        # Load state and action scalers
        self.state_scaler, self.action_scaler = scaler_loader()

        # Define observation space (normalized to [0, 1])
        observation_dim = 5
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(observation_dim,), dtype=np.float32)

        # Define action space (normalized to [-1, 1])
        action_dim = 3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    def reset(self, seed, options=None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)

        if self.is_train:
            self.scenario_seed = self.np_random.choice(self.num_scenarios)
        else:
            self.scenario_seed = int(seed)

        self.time_step = 0

        # Get the index for the scenario data
        index = self._get_index(self.scenario_seed, self.time_step)

        # Initialize state
        initial_state = np.array([
            self.time_step,  # time_step
            self.data['rtp'][index],  # rtp
            self.data['p_pv_max'][index],  # PV
            self.data['g_fcev_demand'][index],  # FCEV demand
            self.sop_hss_max  # initial ESS level
        ], dtype=np.float32)
        
        # Normalize the initial state
        self.state = self.state_scaler.transform([initial_state])[0].astype(np.float32)

        return self.state, {}

    def step(self, action):
        """Take an action and return the next state, reward, and termination status."""
        # Inverse transform the state from normalized form
        current_state = self.state_scaler.inverse_transform([self.state])[0].astype(np.float32)

        # Decompose the state into individual variables
        time_step, rtp, p_pv_max, g_fcev_demand, sop_hss_tempt = current_state
        time_step = int(np.round(time_step))

        # Fetch the data for the current time step
        base_idx = self._get_index(self.scenario_seed, time_step)

        # Inverse transform the action using the scaler
        action_pred = self.action_scaler.inverse_transform(action.reshape(1, -1))[0]
        
        # Clip actions to be within their respective limits
        action_ez, action_fc, action_fcev = np.clip(action_pred[:3],
                                                    [0, 0, 0],
                                                    [self.g_ez_max, self.g_fc_max, g_fcev_demand])
        
        g_ez, g_fc, g_fcev, sop_hss = self._update_hss(time_step, action_ez, action_fc, action_fcev, g_fcev_demand, sop_hss_tempt)

        # Solve the MILP optimization problem
        p_grid_pur, p_grid_exp, u_grid_pur, u_grid_exp, p_pv, p_ez, u_ez, p_fc, u_fc, p_fcev, u_fcev, reward = self._optim(rtp, p_pv_max, g_ez, g_fc, g_fcev)

        # Calculate the reward and penalties
        cumulative_penalty = self._get_penalty(time_step, p_grid_pur, p_grid_exp, sop_hss)
        reward += cumulative_penalty * self.penalty_coefficient

        # Prepare next state
        next_state, terminated = self._get_obs(time_step, sop_hss)

        # Update the state for the next step
        self.state = self.state_scaler.transform([next_state])[0].astype(np.float32)

        return self.state, reward, terminated, False, {
            "p_grid_pur": p_grid_pur,
            "u_grid_pur": u_grid_pur,
            "p_grid_exp": p_grid_exp,
            "u_grid_exp": u_grid_exp,
            "p_pv": p_pv,
            "p_ez": p_ez,
            "g_ez": g_ez,
            "u_ez": u_ez,
            "p_fc": p_fc,
            "g_fc": g_fc,
            "u_fc": u_fc,
            "p_fcev": p_fcev,
            "g_fcev": g_fcev,
            "u_fcev": u_fcev,
            "sop_hss": sop_hss
        }

    def _get_obs(self, time_step, sop_hss):
        """Prepare the next state and determine if the episode has terminated."""
        # Increment the time step.
        time_step += 1

        # Determine if the episode has terminated.
        terminated = time_step >= self.T_num

        # Prepare the next state if the episode is ongoing.
        if not terminated:
            base_idx = self.scenario_seed * self.T_num + time_step
            next_state = np.array([
                time_step,
                self.data['rtp'][base_idx],
                self.data['p_pv_max'][base_idx],
                self.data['g_fcev_demand'][base_idx],
                sop_hss
            ], dtype=np.float32)
        else:
            next_state = np.array([time_step, 0, 0, 0, 0], dtype=np.float32)

        return next_state, terminated

    def _update_hss(self, time_step, action_ez, action_fc, action_fcev, g_fcev_demand, sop_hss_tempt):
        """Update the HSS state based on the action taken."""
        # Initialize charge and discharge power
        g_ez, g_fc = 0.0, 0.0
        g_fcev = action_fcev

        if time_step == 0:
            g_ez = g_fcev + (self.dt_hss * sop_hss_tempt) / (self.delta_t * self.Y_hss)
            pass
        elif time_step == self.T_set[-1]:
            # Limit the charge power to not exceed sop_hss_max
            g_ez, g_fc, g_fcev = self._postprocess_hss_setpoint(g_ez, g_fc, g_fcev, g_fcev_demand, sop_hss_tempt)

        else:
            # Determine g_ez, g_fc, and g_fcev based on actions
            g_ez = max(action_ez - action_fc, 0)
            g_fc = max(action_fc - action_ez, 0)

            # Calculate potential SOP of HSS after applying action
            sop_hss = sop_hss_tempt + self.delta_t * self.Y_hss * (g_ez - g_fc - g_fcev) - self.dt_hss * sop_hss_tempt

            # Postprocess action to meet all SOP of HSS bound constraints
            g_ez, g_fc, g_fcev = self._postprocess_hss_bound(time_step, g_ez, g_fc, g_fcev, g_fcev_demand, sop_hss, sop_hss_tempt)

        # Update SOP of HSS
        sop_hss = sop_hss_tempt + self.delta_t * self.Y_hss * (g_ez - g_fc - g_fcev) - self.dt_hss * sop_hss_tempt

        return g_ez, g_fc, g_fcev, sop_hss

    def _postprocess_hss_bound(self, time_step, g_ez, g_fc, g_fcev, g_fcev_demand, sop_hss, sop_hss_tempt):
        """Adjust HSS charging and discharging powers based on SOP constraints."""
        # Calculate common terms for charging and discharging power limits
        g_hss_ch_max = (self.sop_hss_max - sop_hss_tempt + self.dt_hss * sop_hss_tempt) / (self.delta_t * self.Y_hss)
        
        # Determine the minimum SOP threshold and max discharging power
        sop_hss_min = self.sop_hss_threshold if time_step == self.T_set[-2] else self.sop_hss_min
        g_hss_dch_max = (sop_hss_min - sop_hss_tempt + self.dt_hss * sop_hss_tempt) / (self.delta_t * self.Y_hss)

        # Adjust powers based on SOP constraints
        if sop_hss > self.sop_hss_max:
            g_ez = min(g_hss_ch_max + g_fcev, self.g_ez_max)
            g_fc = 0
        elif sop_hss < sop_hss_min:
            if g_fc == 0:
                g_fcev = max(min(g_ez - g_hss_dch_max, g_fcev_demand), 0)
            else:  # g_fc > 0
                g_ez = 0
                g_fc = max(-g_hss_dch_max - g_fcev, 0)

        return g_ez, g_fc, g_fcev

    def _postprocess_hss_setpoint(self, g_ez, g_fc, g_fcev, g_fcev_demand, sop_hss_tempt):
        """Adjust the charge power to match the HSS setpoint."""
        # Calculate maximum possible HSS charging power
        g_hss_ch_max = (self.sop_hss_max - sop_hss_tempt + self.dt_hss * sop_hss_tempt) / (self.delta_t * self.Y_hss)
        
        # Adjust g_ez and g_fc to ensure it does not exceed maximum allowable power
        g_ez = min(g_hss_ch_max + g_fcev, self.g_ez_max)
        g_fc = 0

        # Update SOP of HSS
        sop_hss = sop_hss_tempt + self.delta_t * self.Y_hss * (g_ez - g_fc - g_fcev) - self.dt_hss * sop_hss_tempt

        # If the calculated SOP HSS deviates from the setpoint, adjust g_fcev
        if sop_hss != self.sop_hss_setpoint:
            g_fcev = max(min(g_ez - g_hss_ch_max, g_fcev_demand), 0)

        return g_ez, g_fc, g_fcev

    def _get_hss_power(self, g_ez, g_fc, g_fcev):
        """Calculate EZ, FC, and FCEV power."""
        p_ez = g_ez * self.LHV / self.n_ez
        p_fc = g_fc * self.LHV * self.n_fc
        p_fcev = g_fcev * self.z_comp / (self.delta_t * self.n_comp)

        return p_ez, p_fc, p_fcev

    def _get_hss_mode(self, g_ez, g_fc, g_fcev):
        """Calculate EZ, FC, and FCEV modes."""
        # Constants for threshold comparison
        THRESHOLD = 1e-5

        # Determine EZ, FC, FCEV modes
        u_ez = int(g_ez > THRESHOLD)
        u_fc = int(g_fc >= THRESHOLD)
        u_fcev = int(g_fcev >= THRESHOLD)

        return u_ez, u_fc, u_fcev

    def _get_hss_cost(self, p_ez, p_fc, g_fcev, u_ez, u_fc, u_fcev):
        """Calculate EZ, FC, and FCEV costs."""
        F_ez = self.delta_t * (self.m_ez * p_ez + u_ez * self.k_ez * self.p_ez_max / self.T_ez)
        F_fc = self.delta_t * (self.m_fc * p_fc + u_fc * self.k_fc * self.p_fc_max / self.T_fc)
        F_fcev = g_fcev * self.phi_fcev

        return F_ez, F_fc, F_fcev

    def _get_penalty(self, time_step, p_grid_pur, p_grid_exp, sop_hss):
        """Calculate penalties for boundary and ramp rate violations for the generator and ESS."""
        # Grid penalties
        grid_penalty = 0
        grid_penalty += check_boundary_constraint(p_grid_pur, 0, self.p_grid_pur_max)
        grid_penalty += check_boundary_constraint(p_grid_exp, 0, self.p_grid_exp_max)

        # ESS penalties
        hss_penalty = 0
        hss_penalty += check_boundary_constraint(sop_hss, self.sop_hss_min, self.sop_hss_max)

        if time_step == 0 or time_step == self.T_set[-1]:
            hss_penalty += check_setpoint(sop_hss, self.sop_hss_setpoint)

        return grid_penalty + hss_penalty

    def _optim(self, rtp, p_pv_max, g_ez, g_fc, g_fcev):
        """Optimization method for the microgrid."""
        # Create a new model
        model = gp.Model()
        model.ModelSense = GRB.MAXIMIZE
        model.Params.LogToConsole = 0

        ## Utility grid modeling
        p_grid_pur = model.addVar(vtype=GRB.CONTINUOUS, name="p_grid_pur")
        u_grid_pur = model.addVar(vtype=GRB.BINARY, name="u_grid_pur")
        p_grid_exp = model.addVar(vtype=GRB.CONTINUOUS, name="p_grid_exp")
        u_grid_exp = model.addVar(vtype=GRB.BINARY, name="u_grid_exp")

        # Grid constraints
        model.addConstr(p_grid_pur <= self.p_grid_pur_max * u_grid_pur)
        model.addConstr(p_grid_exp <= self.p_grid_exp_max * u_grid_exp)
        model.addConstr(u_grid_pur + u_grid_exp >= 0)
        model.addConstr(u_grid_pur + u_grid_exp <= 1)

        ## Renewable energies modeling
        p_pv = model.addVar(lb=0, ub=p_pv_max, vtype=GRB.CONTINUOUS, name="p_pv")

        # Hydrogen chain
        p_ez, p_fc, p_fcev = self._get_hss_power(g_ez, g_fc, g_fcev)
        u_ez, u_fc, u_fcev = self._get_hss_mode(g_ez, g_fc, g_fcev)
        F_ez, F_fc, F_fcev = self._get_hss_cost(p_ez, p_fc, g_fcev, u_ez, u_fc, u_fcev)

        # Energy balance
        model.addConstr(p_grid_pur + p_pv + p_fc == p_grid_exp + p_ez + p_fcev)

        # Cost exchange with utility grid
        F_grid = self.delta_t * (p_grid_pur * rtp - p_grid_exp * rtp * self.phi_rtp)

        # Define problem and solve
        model.setObjective(F_fcev - F_grid - F_ez - F_fc)
        model.optimize()
            
        return p_grid_pur.x, p_grid_exp.x, u_grid_exp.x, u_grid_pur.x, p_pv.x, p_ez, u_ez, p_fc, u_fc, p_fcev, u_fcev, model.objVal

    def _init_params(self):
        """Initialize constants and parameters from the scenario configuration."""
        # General parameters
        self.T_num = cfg.T_NUM
        self.T_set = cfg.T_SET
        self.delta_t = cfg.DELTA_T
        self.penalty_coefficient = cfg.PENALTY_COEFFICIENT

        # Grid exchange parameters
        self.p_grid_pur_max = cfg.P_GRID_PUR_MAX
        self.r_grid_pur = cfg.R_GRID_PUR
        self.p_grid_exp_max = cfg.P_GRID_EXP_MAX
        self.r_grid_exp = cfg.R_GRID_EXP
        self.phi_rtp = cfg.PHI_RTP

        # Parameters for Generation Units
        self.p_pv_rate = cfg.P_PV_RATE
        self.n_pv = cfg.N_PV
        self.phi_pv = cfg.PHI_PV
        
        # Electrolyzer (EZ) Parameters
        self.LHV = cfg.LHV
        self.p_ez_max = cfg.P_EZ_MAX
        self.p_ez_min = cfg.P_EZ_MIN
        self.r_ez = cfg.R_EZ
        self.n_ez = cfg.N_EZ
        self.k_ez = cfg.K_EZ
        self.T_ez = cfg.T_EZ
        self.m_ez = cfg.M_EZ
        self.q_ez = cfg.Q_EZ
        self.g_ez_max = cfg.G_EZ_MAX

        # Fuel Cell (FC) Parameters
        self.p_fc_max = cfg.P_FC_MAX
        self.p_fc_min = cfg.P_FC_MIN
        self.r_fc = cfg.R_FC
        self.n_fc = cfg.N_FC
        self.k_fc = cfg.K_FC
        self.T_fc = cfg.T_FC
        self.m_fc = cfg.M_FC
        self.q_fc = cfg.Q_FC
        self.g_fc_max = cfg.G_FC_MAX

        # FCEV Refueling Station Parameters
        self.n_comp = cfg.N_COMP
        self.z_comp = cfg.Z_COMP
        self.phi_fcev = cfg.PHI_FCEV

        # Hydrogen Storage System (HSS) Parameters
        self.sop_hss_max = cfg.SOP_HSS_MAX
        self.sop_hss_min = cfg.SOP_HSS_MIN
        self.dt_hss = cfg.DT_HSS
        self.sop_hss_setpoint = cfg.SOP_HSS_SETPOINT
        self.Y_hss = cfg.Y_HSS
        self.sop_hss_threshold = cfg.SOP_HSS_THRESHOLD

    def _get_index(self, scenario: int, time_step: int) -> int:
        """Get index for scenario data based on time step and scenario seed."""
        return scenario * self.T_num + time_step
