import gurobipy as gp
from gurobipy import GRB

class FC:
    def __init__(self, T_set, delta_t, p_fc_max, p_fc_min, r_fc, n_fc, k_fc, T_fc, m_fc, q_fc, LHV):
        """Initialize parameters."""
        self.T_set = T_set
        self.delta_t = delta_t
 
        self.p_fc_max = p_fc_max
        self.p_fc_min = p_fc_min
        self.r_fc = r_fc
        self.n_fc = n_fc
        self.k_fc = k_fc
        self.T_fc = T_fc
        self.m_fc = m_fc
        self.q_fc = q_fc
        self.LHV = LHV

    def add_variables(self, model):
        """Add variables to the model."""
        p_fc = model.addVars(self.T_set, vtype=GRB.CONTINUOUS, name="p_fc")
        g_fc = model.addVars(self.T_set, vtype=GRB.CONTINUOUS, name="g_fc")
        u_fc = model.addVars(self.T_set, vtype=GRB.BINARY, name="u_fc")
        F_fc = model.addVars(self.T_set, vtype=GRB.CONTINUOUS, name="F_fc")

        return p_fc, g_fc, u_fc, F_fc

    def add_constraints(self, model, p_fc, g_fc, u_fc, F_fc):
        """Add constraints to the model."""
        for t in self.T_set:
            model.addConstr(p_fc[t] <= self.p_fc_max * u_fc[t])
            model.addConstr(g_fc[t] == p_fc[t] / (self.n_fc * self.LHV))    
            model.addConstr(F_fc[t] == self.delta_t * (self.m_fc * p_fc[t] + u_fc[t] * self.k_fc * self.p_fc_max / self.T_fc))