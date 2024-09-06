import gurobipy as gp
from gurobipy import GRB

class FCEV:
    def __init__(self, n_comp, z_comp, phi_fcev, T_num, T_set, delta_t):
        self.n_comp = n_comp
        self.z_comp = z_comp
        self.phi_fcev = phi_fcev

        self.T_num = T_num
        self.T_set = T_set
        self.delta_t = delta_t

    def add_variables(self, model):
        p_fcev = model.addMVar(self.T_num, vtype=GRB.CONTINUOUS, name="p_fcev")
        g_fcev = model.addMVar(self.T_num, vtype=GRB.CONTINUOUS, name="g_fcev")
        u_fcev = model.addMVar(self.T_num, vtype=GRB.BINARY, name="u_fcev")

        return p_fcev, g_fcev, u_fcev

    def add_constraints(self, model, p_fcev, g_fcev, u_fcev, g_fcev_demand):
        for i in self.T_set:
            model.addConstr(p_fcev[i] == g_fcev[i] * self.z_comp / (self.delta_t * self.n_comp))
            model.addConstr(g_fcev[i] <= g_fcev_demand[i] * u_fcev[i])

    def get_revenue(self, g_fcev):
        # Profit cost of FCEV refueling station
        F_fcev = gp.quicksum(g_fcev[i] * self.phi_fcev for i in self.T_set)

        return F_fcev

