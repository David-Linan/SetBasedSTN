import math
from hexaly.optimizer import HexalyOptimizer, HxInterval, HxParam,HxStatistics
from tabulate import tabulate
import pandas as pd
from pathlib import Path
import numpy as np
from base_formulations import base_mip,base_minp_1,base_minlip_1,base_minp_2,base_minlip_2
from hexaly_benchmark import flush_table_to_txt,write_to_excel
# === Problem data ===

class Data:
    def __init__(self, eta_f=18.0, delta_f=1.0, acc_level=1):
        # Scheduling horizon and time discretization parameters
        self.eta_f = eta_f                  # Total scheduling horizon (e.g., in hours)
        self.delta_f = delta_f              # Base time step
        self.acc_level = acc_level          # Accuracy level multiplier

        self.firstT = 0                     # Start time index (always 0)
        self.delta = acc_level * delta_f    # Actual time step used in the model
        self.lastT = math.floor(eta_f / self.delta)  # Last time index based on horizon and step
        self.eta = self.lastT * self.delta  # Effective horizon covered by discretization

        # Sets of units, tasks, and states
        self.J = ['C1', 'C2', 'C3']    # Units
        self.I = ['P1', 'P2', 'P3', 'P4', 'P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15']  # Tasks
        self.K = ['S1', 'S2']  # States

        # Time indices
        self.T = list(range(self.firstT, self.lastT + 1))  # Discrete time steps
        self.Tp = [t * self.delta for t in self.T]         # Physical time points


        # State-to-task consumption mapping
        self.I_i_k_minus = {
            ('P1','S1'):1,
            ('P2','S1'):1,
            ('P3','S1'):1,
            ('P4','S1'):1,
            ('P5','S1'):1,
            ('P6','S1'):1,
            ('P7','S1'):1,
            ('P8','S1'):1,
            ('P9','S1'):1,
            ('P10','S1'):1,
            ('P11','S1'):1,
            ('P12','S1'):1,
            ('P13','S1'):1,
            ('P14','S1'):1,
            ('P15','S1'):1,
        }

        # Task-to-state production mapping
        self.I_i_k_plus = {
            ('P1','S2'):1,
            ('P2','S2'):1,
            ('P3','S2'):1,
            ('P4','S2'):1,
            ('P5','S2'):1,
            ('P6','S2'):1,
            ('P7','S2'):1,
            ('P8','S2'):1,
            ('P9','S2'):1,
            ('P10','S2'):1,
            ('P11','S2'):1,
            ('P12','S2'):1,
            ('P13','S2'):1,
            ('P14','S2'):1,
            ('P15','S2'):1,
        }

        # Consumption coefficients
        self.rho_minus = {
            ('P1','S1'):1,
            ('P2','S1'):1,
            ('P3','S1'):1,
            ('P4','S1'):1,
            ('P5','S1'):1,
            ('P6','S1'):1,
            ('P7','S1'):1,
            ('P8','S1'):1,
            ('P9','S1'):1,
            ('P10','S1'):1,
            ('P11','S1'):1,
            ('P12','S1'):1,
            ('P13','S1'):1,
            ('P14','S1'):1,
            ('P15','S1'):1,
        }

        # Production coefficients
        self.rho_plus = {
            ('P1','S2'):1,
            ('P2','S2'):1,
            ('P3','S2'):1,
            ('P4','S2'):1,
            ('P5','S2'):1,
            ('P6','S2'):1,
            ('P7','S2'):1,
            ('P8','S2'):1,
            ('P9','S2'):1,
            ('P10','S2'):1,
            ('P11','S2'):1,
            ('P12','S2'):1,
            ('P13','S2'):1,
            ('P14','S2'):1,
            ('P15','S2'):1,
        }

        # Task-unit assignment
        I_i_j_prod_partial = {
            ('P1','C1'):1,
            ('P2','C1'):1,
            ('P3','C1'):1,
            ('P4','C1'):1,
            ('P5','C1'):1,
            ('P6','C1'):1,
            ('P7','C1'):1,
            ('P8','C1'):1,
            ('P9','C1'):1,
            ('P10','C1'):1,
            ('P11','C1'):1,
            ('P12','C1'):1,
            ('P13','C1'):1,
            ('P14','C1'):1,
            ('P15','C1'):1,

            ('P1','C2'):1,
            ('P2','C2'):1,
            ('P3','C2'):1,
            ('P4','C2'):1,
            ('P5','C2'):1,
            ('P6','C2'):1,
            ('P7','C2'):1,
            ('P8','C2'):1,
            ('P9','C2'):1,
            ('P10','C2'):1,
            ('P11','C2'):1,
            ('P12','C2'):1,
            ('P13','C2'):1,
            ('P14','C2'):1,
            ('P15','C2'):1,

            ('P1','C3'):1,
            ('P2','C3'):1,
            ('P3','C3'):1,
            ('P4','C3'):1,
            ('P5','C3'):1,
            ('P6','C3'):1,
            ('P7','C3'):1,
            ('P8','C3'):1,
            ('P9','C3'):1,
            ('P10','C3'):1,
            ('P11','C3'):1,
            ('P12','C3'):1,
            ('P13','C3'):1,
            ('P14','C3'):1,
            ('P15','C3'):1
        }

        # Processing times
        times={
            'P1':3,
            'P2':2,
            'P3':4,
            'P4':3,
            'P5':2,
            'P6':5,
            'P7':3,
            'P8':4,
            'P9':2,
            'P10':5,
            'P11':3,
            'P12':4,
            'P13':3,
            'P14':3,
            'P15':2
        }

        # Project sizes
        size={
            'P1':10,
            'P2':15,
            'P3':20,
            'P4':12,
            'P5':18,
            'P6':25,
            'P7':14,
            'P8':22,
            'P9':9,
            'P10':29,
            'P11':16,
            'P12':27,
            'P13':13,
            'P14':21,
            'P15':19
        }

        # chamber capacities
        ch_capacity_lo ={'C1':5, 'C2':5, 'C3':5}
        ch_capacity_up ={'C1':15, 'C2':25, 'C3':30}
        

        self.tau_p={}
        self.I_i_j_prod={}
        self.beta_min = {}
        self.beta_max = {}

        for key in I_i_j_prod_partial.keys():
            for key_times in times.keys():
                if key[0]==key_times and (size[key[0]]>=ch_capacity_lo[key[1]] and size[key[0]]<=ch_capacity_up[key[1]]):
                    self.tau_p[key]=times[key_times]
                    self.I_i_j_prod[key]=1
                    self.beta_min[key]=size[key_times]
                    self.beta_max[key]=size[key_times]
        # Processing times in time steps (rounded up)
        self.tau = {k: math.ceil(self.tau_p[k] / self.delta) for k in self.tau_p}


        # Inventory bounds

        self.upper_s = {
            'S1':sum(size[i] for i in size.keys()), 'S2':sum(size[i] for i in size.keys())
        }

        self.lower_s = {k: 0 for k in self.K}  # All states have zero lower bound

        # Time-indexed demand and replenishment (default: zero)
        self.demand = {(k,t): 0 for k in self.K for t in self.T}
        self.replenishment = {(k,t): 0 for k in self.K for t in self.T}

        # Initial inventory levels
        self.S0 = {'S1': sum(size[i] for i in size.keys()), 'S2': 0}

        # State revenues
        self.revenue = {
            'P1':45,
            'P2':10,
            'P3':39,
            'P4':67,
            'P5':91,
            'P6':10,
            'P7':24,
            'P8':17,
            'P9':75,
            'P10':61,
            'P11':34,
            'P12':43,
            'P13':44,
            'P14':26,
            'P15':37
        }


        # Execution bounds per task-unit pair
        self.upper_n = {
            (i,j): 1
            for (i,j) in self.I_i_j_prod
        }

        self.lower_n = {
            (i,j): 0
            for (i,j) in self.I_i_j_prod
        }

        # Realization index bounds (used for interval-based models)
        self.lower_q = {
            (i,j): 1 if self.lower_n[(i,j)] == 0 else self.lower_n[(i,j)]
            for (i,j) in self.I_i_j_prod
        }

        self.upper_q = {
            (i,j): self.upper_n[(i,j)]
            for (i,j) in self.I_i_j_prod
        }

        # Realization index ranges
        self.Q = {
            (i,j): range(self.lower_q[(i,j)], self.upper_q[(i,j)] + 1)
            for (i,j) in self.I_i_j_prod
        }     


# === General STN formualtions (with optional tasks) ===
        
def mip(optimizer,data):

    m, x, s, b=base_mip(optimizer,data)


    # Constraint: task execution constraint: each task must be assigned to at most one unit 
    for i in data.I:
        m.constraint(m.sum(m.sum(x[i,j,t] for t in data.T)  for j in data.J if (i,j) in data.I_i_j_prod    ) <= 1)

    # Constraint: time horizon constraint: projects must complete within the scheduling period
    for (i,j) in data.I_i_j_prod:
        for t in (tt for tt in data.T if tt> data.lastT-data.tau[(i,j)]+1):
            m.constraint(x[i,j,t]==0)

    # Objective
    # Maximize revenue
    revenue = m.sum(data.revenue[i]*m.sum(x[i,j,t] for t in data.T) for (i, j) in data.I_i_j_prod)
    
    m.minimize(-revenue)

    return m, x, s, b

def minp_1(optimizer,data):

    m, interv, s, b=base_minp_1(optimizer,data)

    # Constraint: task execution constraint: each task must be assigned to at most one unit 
    for i in data.I:
        m.constraint(m.sum(m.sum(m.gt(m.length(interv[i,j,q]),0) for q in data.Q[(i,j)])  for j in data.J if (i,j) in data.I_i_j_prod    ) <= 1)   

    # Constraint: time horizon constraint: projects must complete within the scheduling period
    #NOTE: Not needed. Already implicity in interval definition

    # Objective
    # Maximize revenue
    revenue=m.sum(data.revenue[i]*m.sum(m.gt(m.length(interv[i,j,q]),0) for q in data.Q[(i,j)]) for (i, j) in data.I_i_j_prod)

    m.minimize(-revenue)

    return m, interv, s, b

def minlip_1(optimizer,data):

    m, interv, s, b =base_minlip_1(optimizer,data)

    # Constraint: task execution constraint: each task must be assigned to at most one unit 
    for i in data.I:
        m.constraint(m.sum(m.sum(m.gt(m.length(interv[i,j,q]),0) for q in data.Q[(i,j)])  for j in data.J if (i,j) in data.I_i_j_prod    ) <= 1)   

  
    # Constraint: time horizon constraint: projects must complete within the scheduling period
    #NOTE: Not needed. Already implicity in interval definition

    # Objective
    # Maximize revenue
    revenue=m.sum(data.revenue[i]*m.sum(m.gt(m.length(interv[i,j,q]),0) for q in data.Q[(i,j)]) for (i, j) in data.I_i_j_prod)

    m.minimize(-revenue)

    return m, interv, s, b

def minp_2(optimizer,data):

    m, interv, s, b=base_minp_2(optimizer,data)

    # Constraint: task execution constraint: each task must be assigned to at most one unit 
    for i in data.I:
        m.constraint(m.sum(m.sum(m.gt(m.length(interv[i,j,q]),0) for q in data.Q[(i,j)])  for j in data.J if (i,j) in data.I_i_j_prod    ) <= 1)   

    # Constraint: time horizon constraint: projects must complete within the scheduling period
    #NOTE: Not needed. Already implicity in interval definition

    # Objective
    # Maximize revenue
    revenue=m.sum(data.revenue[i]*m.sum(m.gt(m.length(interv[i,j,q]),0) for q in data.Q[(i,j)]) for (i, j) in data.I_i_j_prod)

    m.minimize(-revenue)

    return m, interv, s, b

def minlip_2(optimizer,data):

    m, interv, s, b=base_minlip_2(optimizer,data)

    # Constraint: task execution constraint: each task must be assigned to at most one unit 
    for i in data.I:
        m.constraint(m.sum(m.sum(m.gt(m.length(interv[i,j,q]),0) for q in data.Q[(i,j)])  for j in data.J if (i,j) in data.I_i_j_prod    ) <= 1)   

    # Constraint: time horizon constraint: projects must complete within the scheduling period
    #NOTE: Not needed. Already implicity in interval definition

    # Objective
    # Maximize revenue
    revenue=m.sum(data.revenue[i]*m.sum(m.gt(m.length(interv[i,j,q]),0) for q in data.Q[(i,j)]) for (i, j) in data.I_i_j_prod)

    m.minimize(-revenue)

    return m, interv, s, b





if __name__ == '__main__':
    # ------------------------
    # Benchmarking Parameters
    # ------------------------
    delta_f = 1         # Base time step (in time units)
    time_limit = 300    # Time limit for optimization (in seconds)
    seed = 1            # Random seed for reproducibility

    # ------------------------
    # Formulation Dictionaries
    # ------------------------
    # Each key corresponds to a formulation variant
    unknown_n_formulations = {
        1: mip,
        2: minp_1,
        3: minlip_1,
        4: minp_2,
        5: minlip_2
    }

    # ------------------------
    # Result Containers
    # ------------------------
    original_results = {}     # Stores raw results for unknown-n formulations
    original_table = {}       # Tabular format for unknown-n results

    # ------------------------
    # Benchmarking Loop
    # ------------------------
    time_horizons=range(6,18+1)

    for eta_f in time_horizons:
        # Run all unknown-n formulations
        for key, formulation in unknown_n_formulations.items():
            data = Data(eta_f=eta_f, delta_f=delta_f, acc_level=1)
            with HexalyOptimizer() as optimizer:
                # Build and solve model
                m, x, s, b = formulation(optimizer, data)
                m.close()
                optimizer.param.time_limit = time_limit
                optimizer.param.seed = seed
                optimizer.solve()

                # Extract solution metrics
                objective = optimizer.solution.get_value(m.objectives[0])
                objective_bound = optimizer.solution.get_objective_bound(0)
                objective_gap = optimizer.solution.get_objective_gap(0) * 100
                comp_time = optimizer.statistics.get_running_time()
                status = str(optimizer.solution.status)

                # Store results
                original_results.setdefault(key, []).append([
                    eta_f, objective, objective_bound, objective_gap, comp_time, status
                ])
                original_table.setdefault(eta_f, {})[key] = [
                    objective, objective_bound, objective_gap, comp_time, status
                ]

        # ------------------------
        # Save Intermediate Tables
        # ------------------------
        flush_table_to_txt("original_2.txt", original_table, list(unknown_n_formulations.keys()))

    # ------------------------
    # Save Final Results
    # ------------------------
    write_to_excel("original_2.xlsx", original_results)
