import math
from hexaly.optimizer import HexalyOptimizer, HxInterval, HxParam,HxStatistics
from tabulate import tabulate
import pandas as pd
from pathlib import Path
from hexaly_benchmark import minlip_1

# === Problem data ===

class Data:
    def __init__(self, eta_f=6.0, delta_f=1.0, acc_level=1):
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
            ('T2','S3'):1, ('T2','S2'):1,
            ('T3','S4'):1, ('T3','S5'):1,
            ('T4','S6'):1, ('T4','S3'):1,
            ('T5','S7'):1
        }

        # Task-to-state production mapping
        self.I_i_k_plus = {
            ('T1','S4'):1,
            ('T2','S5'):1,
            ('T3','S6'):1, ('T3','S8'):1,
            ('T4','S7'):1,
            ('T5','S6'):1, ('T5','S9'):1
        }

        # Consumption coefficients
        self.rho_minus = {
            ('T1','S1'):1,
            ('T2','S3'):0.5, ('T2','S2'):0.5,
            ('T3','S4'):0.4, ('T3','S5'):0.6,
            ('T4','S6'):0.8, ('T4','S3'):0.2,
            ('T5','S7'):1
        }

        # Production coefficients
        self.rho_plus = {
            ('T1','S4'):1,
            ('T2','S5'):1,
            ('T3','S6'):0.6, ('T3','S8'):0.4,
            ('T4','S7'):1,
            ('T5','S6'):0.1, ('T5','S9'):0.9
        }

        # Task-unit assignment
        self.I_i_j_prod = {
            ('T1','U1'):1,
            ('T2','U2'):1, ('T2','U3'):1,
            ('T3','U2'):1, ('T3','U3'):1,
            ('T4','U2'):1, ('T4','U3'):1,
            ('T5','U4'):1
        }

        # Processing times (in hours)
        self.tau_p = {
            ('T1','U1'):0.5,
            ('T2','U2'):0.5, ('T2','U3'):1.5,
            ('T3','U2'):1.0, ('T3','U3'):2.5,
            ('T4','U2'):1.0, ('T4','U3'):5.0,
            ('T5','U4'):1.5
        }

        # Processing times in time steps (rounded up)
        self.tau = {k: math.ceil(self.tau_p[k] / self.delta) for k in self.tau_p}

        # Minimum and maximum batch sizes
        self.beta_min = {
            ('T1','U1'):10,
            ('T2','U2'):10, ('T2','U3'):10,
            ('T3','U2'):10, ('T3','U3'):10,
            ('T4','U2'):10, ('T4','U3'):10,
            ('T5','U4'):10
        }

        self.beta_max = {
            ('T1','U1'):100,
            ('T2','U2'):50, ('T2','U3'):80,
            ('T3','U2'):50, ('T3','U3'):80,
            ('T4','U2'):50, ('T4','U3'):80,
            ('T5','U4'):200
        }

        # Inventory bounds
        self.upper_s = {
            'S1':4000, 'S2':4000, 'S3':4000, 'S4':1000, 'S5':150,
            'S6':500, 'S7':1000, 'S8':4000, 'S9':4000
        }

        self.lower_s = {k: 0 for k in self.K}  # All states have zero lower bound

        # Time-indexed demand and replenishment (default: zero)
        self.demand = {(k,t): 0 for k in self.K for t in self.T}
        self.replenishment = {(k,t): 0 for k in self.K for t in self.T}

        # Initial inventory levels
        self.S0 = {k: 0 for k in self.K}
        self.S0.update({'S1': 4000, 'S2': 4000, 'S3': 4000})  # Preloaded states

        # Task-unit costs
        self.cost = {
            ('T1','U1'):10,
            ('T2','U2'):15, ('T2','U3'):30,
            ('T3','U2'):5,  ('T3','U3'):25,
            ('T4','U2'):5,  ('T4','U3'):20,
            ('T5','U4'):20
        }

        # State revenues
        self.revenue = {k: 0 for k in self.K}
        self.revenue.update({'S8': 3, 'S9': 4})  # Only final products generate revenue

        # Execution bounds per task-unit pair
        self.upper_n = {
            (i,j): math.floor(self.lastT / self.tau[(i,j)])
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

if __name__ == '__main__':
    a=1