import gurobipy as gp
from gurobipy import GRB

class HSS:
    def __init__(self, sop_hss_max, sop_hss_min, dt_hss, sop_hss_setpoint, Y_hss, T_num, T_set, delta_t):
        self.sop_hss_max = sop_hss_max
        self.sop_hss_min = sop_hss_min
        self.dt_hss = dt_hss
        self.sop_hss_setpoint = sop_hss_setpoint
        self.Y_hss = Y_hss

        self.T_num = T_num
        self.T_set = T_set
        self.delta_t = delta_t

    def add_variables(self, model):
        sop_hss = model.addMVar(self.T_num, lb=self.sop_hss_min, ub=self.sop_hss_max, vtype=GRB.CONTINUOUS, name="sop_hss")
        return sop_hss

    def add_constraints(self, model, sop_hss, g_ez, g_fc, g_fcev, u_ez, u_fc):
        for i in self.T_set:
            # Committment of electrolyzer and fuel cell
            model.addConstr(u_ez[i] + u_fc[i] >= 0)
            model.addConstr(u_ez[i] + u_fc[i] <= 1)

            # State of pressure of hydrogen storage system
            if i == 0:
                model.addConstr(sop_hss[i] == self.sop_hss_setpoint + self.delta_t * self.Y_hss * (g_ez[i] - g_fc[i] - g_fcev[i]) - self.dt_hss * self.sop_hss_setpoint)
            else:
                model.addConstr(sop_hss[i] == sop_hss[i - 1] + self.delta_t * self.Y_hss * (g_ez[i] - g_fc[i] - g_fcev[i]) - self.dt_hss * sop_hss[i - 1])

        model.addConstr(sop_hss[0] == self.sop_hss_setpoint)
        model.addConstr(sop_hss[self.T_set[-1]] == self.sop_hss_setpoint)
