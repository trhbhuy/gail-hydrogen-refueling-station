# src/solver/platform/hydrogen_station.py

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, Optional
from .. import config as cfg

from .components.utility_grid import Grid
from .components.renewables import PV
from .components.electrolyzer import EZ
from .components.fuel_cell import FC
from .components.compressor import FCEV
from .components.hydrogen_storage import HSS

class HydrogenStation:
    def __init__(self):
        """Initialize constants and parameters from the configuration."""
        # Time Horizon
        self.T_num = cfg.T_NUM
        self.T_set = cfg.T_SET
        self.delta_t = cfg.DELTA_T

        # Initialize the components of the microgrid
        self.grid = Grid(cfg.P_GRID_PUR_MAX, cfg.R_GRID_PUR, cfg.P_GRID_EXP_MAX, cfg.R_GRID_EXP, cfg.PHI_RTP, cfg.T_NUM, cfg.T_SET, cfg.DELTA_T)
        self.pv = PV(cfg.P_PV_RATE, cfg.N_PV, cfg.PHI_PV, cfg.T_NUM, cfg.T_SET)
        self.ez = EZ(cfg.P_EZ_MAX, cfg.P_EZ_MIN, cfg.R_EZ, cfg.N_EZ, cfg.K_EZ, cfg.T_EZ, cfg.M_EZ, cfg.Q_EZ, cfg.LHV, cfg.T_NUM, cfg.T_SET, cfg.DELTA_T)     
        self.fc = FC(cfg.P_FC_MAX, cfg.P_FC_MIN, cfg.R_FC, cfg.N_FC, cfg.K_FC, cfg.T_FC, cfg.M_FC, cfg.Q_FC, cfg.LHV, cfg.T_NUM, cfg.T_SET, cfg.DELTA_T)
        self.fcev = FCEV(cfg.N_COMP, cfg.Z_COMP, cfg.PHI_FCEV, cfg.T_NUM, cfg.T_SET, cfg.DELTA_T)
        self.hss = HSS(cfg.SOP_HSS_MAX, cfg.SOP_HSS_MIN, cfg.DT_HSS, cfg.SOP_HSS_SETPOINT, cfg.Y_HSS, cfg.T_NUM, cfg.T_SET, cfg.DELTA_T)

    def optim(self, rtp: np.ndarray, p_pv_max: np.ndarray, g_fcev_demand: np.ndarray) -> Dict[str, np.ndarray]:
        """Optimization method for the hydrogen refueling station."""
        # Create a new model
        model = gp.Model()
        model.ModelSense = GRB.MAXIMIZE
        model.Params.LogToConsole = 0

        # Initialize variables for each component
        p_grid_pur, u_grid_pur, p_grid_exp, u_grid_exp = self.grid.add_variables(model)
        p_pv = self.pv.add_variables(model, p_pv_max)
        p_ez, g_ez, u_ez, F_ez = self.ez.add_variables(model)
        p_fc, g_fc, u_fc, F_fc = self.fc.add_variables(model)
        p_fcev, g_fcev, u_fcev = self.fcev.add_variables(model)
        sop_hss = self.hss.add_variables(model)

        # Add constraints for each component
        self.grid.add_constraints(model, p_grid_pur, u_grid_pur, p_grid_exp, u_grid_exp)
        self.ez.add_constraints(model, p_ez, g_ez, u_ez, F_ez)
        self.fc.add_constraints(model, p_fc, g_fc, u_fc, F_fc)
        self.fcev.add_constraints(model, p_fcev, g_fcev, u_fcev, g_fcev_demand)
        self.hss.add_constraints(model, sop_hss, g_ez, g_fc, g_fcev, u_ez, u_fc)

        # Energy balance
        for i in self.T_set:
            model.addConstr(p_grid_pur[i] + p_pv[i] + p_fc[i] == p_grid_exp[i] + p_ez[i] + p_fcev[i])

        # Profit cost of FCEV refueling station
        F_fcev = self.fcev.get_revenue(g_fcev)

        # Cost exchange with utility grid
        F_grid = self.grid.get_cost(rtp, p_grid_pur, p_grid_exp)

        # Define problem and solve
        model.setObjective(F_fcev - F_grid - F_ez.sum() - F_fc.sum())
        model.optimize()

        if model.status == GRB.OPTIMAL:
            # Return the results
            results = {
                'ObjVal': model.ObjVal,
                'p_grid_pur': p_grid_pur.X,
                'u_grid_pur': u_grid_pur.X,
                'p_grid_exp': p_grid_exp.X,
                'u_grid_exp': u_grid_exp.X,
                'p_pv': p_pv.X,
                'p_ez': p_ez.X,
                'g_ez': g_ez.X,
                'u_ez': u_ez.X,
                'p_fc': p_fc.X,
                'g_fc': g_fc.X,
                'u_fc': u_fc.X,
                'p_fcev': p_fcev.X,
                'g_fcev': g_fcev.X,
                'u_fcev': u_fcev.X,
                'sop_hss': sop_hss.X,
            }
        else:
            raise RuntimeError(f"Optimization was unsuccessful. Model status: {model.status}")

        return results