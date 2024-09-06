import gurobipy as gp
from gurobipy import GRB

class FC:
    def __init__(self, p_fc_max, p_fc_min, r_fc, n_fc, k_fc, T_fc, m_fc, q_fc, LHV, T_num, T_set, delta_t):
        self.p_fc_max = p_fc_max
        self.p_fc_min = p_fc_min
        self.r_fc = r_fc
        self.n_fc = n_fc
        self.k_fc = k_fc
        self.T_fc = T_fc
        self.m_fc = m_fc
        self.q_fc = q_fc
        self.LHV = LHV

        self.T_num = T_num
        self.T_set = T_set
        self.delta_t = delta_t

    def add_variables(self, model):
        p_fc = model.addMVar(self.T_num, vtype=GRB.CONTINUOUS, name="p_fc")
        g_fc = model.addMVar(self.T_num, vtype=GRB.CONTINUOUS, name="g_fc")
        u_fc = model.addMVar(self.T_num, vtype=GRB.BINARY, name="u_fc")
        F_fc = model.addMVar(self.T_num, vtype=GRB.CONTINUOUS, name="F_fc")

        return p_fc, g_fc, u_fc, F_fc

    def add_constraints(self, model, p_fc, g_fc, u_fc, F_fc):
        for i in self.T_set:
            model.addConstr(p_fc[i] <= self.p_fc_max * u_fc[i])
            model.addConstr(g_fc[i] == p_fc[i] / (self.n_fc * self.LHV))    
            model.addConstr(F_fc[i] == self.delta_t * (self.m_fc * p_fc[i] + u_fc[i] * self.k_fc * self.p_fc_max / self.T_fc))