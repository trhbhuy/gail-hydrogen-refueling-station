import gurobipy as gp
from gurobipy import GRB

class EZ:
    def __init__(self, p_ez_max, p_ez_min, r_ez, n_ez, k_ez, T_ez, m_ez, q_ez, LHV, T_num, T_set, delta_t):
        self.p_ez_max = p_ez_max
        self.p_ez_min = p_ez_min
        self.r_ez = r_ez
        self.n_ez = n_ez
        self.k_ez = k_ez
        self.T_ez = T_ez
        self.m_ez = m_ez
        self.q_ez = q_ez
        self.LHV = LHV

        self.T_num = T_num
        self.T_set = T_set
        self.delta_t = delta_t

    def add_variables(self, model):
        p_ez = model.addMVar(self.T_num, vtype=GRB.CONTINUOUS, name="p_ez")
        g_ez = model.addMVar(self.T_num, vtype=GRB.CONTINUOUS, name="g_ez")
        u_ez = model.addMVar(self.T_num, vtype=GRB.BINARY, name="u_ez")
        F_ez = model.addMVar(self.T_num, vtype=GRB.CONTINUOUS, name="F_ez")

        return p_ez, g_ez, u_ez, F_ez

    def add_constraints(self, model, p_ez, g_ez, u_ez, F_ez):
        for i in self.T_set:
            model.addConstr(p_ez[i] <= self.p_ez_max * u_ez[i])
            model.addConstr(g_ez[i] == p_ez[i] * self.n_ez / self.LHV)
            model.addConstr(F_ez[i] == self.delta_t * (self.m_ez * p_ez[i] + u_ez[i] * self.k_ez * self.p_ez_max / self.T_ez))
