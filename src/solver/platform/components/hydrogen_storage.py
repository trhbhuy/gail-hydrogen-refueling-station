import gurobipy as gp
from gurobipy import GRB

class HSS:
    def __init__(self, T_set, delta_t, sop_hss_max, sop_hss_min, dt_hss, sop_hss_setpoint, Y_hss):
        """Initialize parameters."""
        self.T_set = T_set
        self.delta_t = delta_t

        self.sop_hss_max = sop_hss_max
        self.sop_hss_min = sop_hss_min
        self.dt_hss = dt_hss
        self.sop_hss_setpoint = sop_hss_setpoint
        self.Y_hss = Y_hss

    def add_variables(self, model):
        """Add variables to the model."""
        sop_hss = model.addVars(self.T_set, lb=self.sop_hss_min, ub=self.sop_hss_max, vtype=GRB.CONTINUOUS, name="sop_hss")
        return sop_hss

    def add_constraints(self, model, sop_hss, g_ez, g_fc, g_fcev, u_ez, u_fc):
        """Add constraints to the model."""
        for t in self.T_set:
            # Commitment of electrolyzer and fuel cell
            model.addConstr(u_ez[t] + u_fc[t] >= 0)
            model.addConstr(u_ez[t] + u_fc[t] <= 1)

            # State of pressure of hydrogen storage system
            if t == 0:
                model.addConstr(sop_hss[t] == self.sop_hss_setpoint + self.delta_t * self.Y_hss * (g_ez[t] - g_fc[t] - g_fcev[t]) - self.dt_hss * self.sop_hss_setpoint)
            else:
                model.addConstr(sop_hss[t] == sop_hss[t - 1] + self.delta_t * self.Y_hss * (g_ez[t] - g_fc[t] - g_fcev[t]) - self.dt_hss * sop_hss[t - 1])

        model.addConstr(sop_hss[0] == self.sop_hss_setpoint)
        model.addConstr(sop_hss[self.T_set[-1]] == self.sop_hss_setpoint)
