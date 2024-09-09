import gurobipy as gp
from gurobipy import GRB

class FCEV:
    def __init__(self, T_set, delta_t, n_comp, z_comp, phi_fcev):
        """Initialize parameters."""
        self.T_set = T_set
        self.delta_t = delta_t

        self.n_comp = n_comp
        self.z_comp = z_comp
        self.phi_fcev = phi_fcev

    def add_variables(self, model):
        """Add variables to the model."""
        p_fcev = model.addVars(self.T_set, vtype=GRB.CONTINUOUS, name="p_fcev")
        g_fcev = model.addVars(self.T_set, vtype=GRB.CONTINUOUS, name="g_fcev")
        u_fcev = model.addVars(self.T_set, vtype=GRB.BINARY, name="u_fcev")

        return p_fcev, g_fcev, u_fcev

    def add_constraints(self, model, p_fcev, g_fcev, u_fcev, g_fcev_demand):
        """Add constraints to the model."""
        for t in self.T_set:
            model.addConstr(p_fcev[t] == g_fcev[t] * self.z_comp / (self.delta_t * self.n_comp))
            model.addConstr(g_fcev[t] <= g_fcev_demand[t] * u_fcev[t])

    def get_revenue(self, g_fcev):
        """Calculate profit cost of FCEV refueling station."""
        F_fcev = gp.quicksum(g_fcev[t] * self.phi_fcev for t in self.T_set)
        return F_fcev

