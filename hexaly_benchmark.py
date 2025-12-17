import math
from hexaly.optimizer import HexalyOptimizer, HxInterval, HxParam,HxStatistics
import pyomo.environ as pe
from tabulate import tabulate
import numpy as np
import pandas as pd
from pathlib import Path
from base_formulations import base_mip,base_minp_1,base_minlip_1,base_minp_2,base_minlip_2,base_mip_known_n,base_minp_1_known_n,base_minlip_1_known_n,base_minp_2_known_n,base_minlip_2_known_n

# === Problem data ===
class Data:
    def __init__(self, eta_f=120.0, delta_f=1.0, acc_level=1):
        # Scheduling horizon and time discretization parameters
        self.eta_f = eta_f                  # Total scheduling horizon (e.g., in hours)
        self.delta_f = delta_f              # Base time step
        self.acc_level = acc_level          # Accuracy level multiplier

        self.firstT = 0                     # Start time index (always 0)
        self.delta = acc_level * delta_f    # Actual time step used in the model
        self.lastT = math.floor(eta_f / self.delta)  # Last time index based on horizon and step
        self.eta = self.lastT * self.delta  # Effective horizon covered by discretization

        # Sets of units, tasks, and states
        self.J = ['U1', 'U2', 'U3', 'U4']    # Units
        self.I = ['T1', 'T2', 'T3', 'T4', 'T5']  # Tasks
        self.K = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']  # States

        # Time indices
        self.T = list(range(self.firstT, self.lastT + 1))  # Discrete time steps
        self.Tp = [t * self.delta for t in self.T]         # Physical time points

        # Task-to-state consumption mapping
        self.I_i_k_minus = {
            ('T1','S1'):1,
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
 
# === General STN formualtions (with optional tasks) ===
        
def mip(optimizer,data):

    m, x, s, b=base_mip(optimizer,data)

    # Constraint: time horizon constraint: projects must complete within the scheduling period
    for (i,j) in data.I_i_j_prod:
        for t in (tt for tt in data.T if tt>= data.lastT-data.tau[(i,j)]+1):
            m.constraint(x[i,j,t]==0)

    # Objective
    # Maximize profit: final inventory value minus total task costs
    profit = m.sum(data.revenue[k]*s[k,data.lastT] for k in data.K) \
           - m.sum(data.cost[i,j]*m.sum(x[i,j,t] for t in data.T) for (i, j) in data.I_i_j_prod)
    m.minimize(-profit)

    return m, x, s, b

def minp_1(optimizer,data):

    m, interv, s, b=base_minp_1(optimizer,data)
    # Constraint: time horizon constraint: projects must complete within the scheduling period
    #NOTE: Not needed. Already implicity in interval definition

    # Objective
    # Maximize profit: final inventory value minus total task costs
    # Task cost is counted only for active realizations (length > 0)
    profit = m.sum(data.revenue[k]*s[k,data.lastT] for k in data.K) \
           - m.sum(data.cost[i,j]*m.sum(m.gt(m.length(interv[i,j,q]),0) for q in data.Q[(i,j)])
                   for (i, j) in data.I_i_j_prod)
    m.minimize(-profit)

    return m, interv, s, b

def minlip_1(optimizer,data):

    m, interv, s, b=base_minlip_1(optimizer,data)

    # Constraint: time horizon constraint: projects must complete within the scheduling period
    #NOTE: Not needed. Already implicity in interval definition

    # Objective
    # Maximize profit: final inventory value minus total task costs
    # Task cost is counted only for active realizations (length > 0)
    profit = m.sum(data.revenue[k]*s[k,data.lastT] for k in data.K) \
           - m.sum(data.cost[i,j]*m.sum(m.gt(m.length(interv[i,j,q]),0) for q in data.Q[(i,j)])
                   for (i, j) in data.I_i_j_prod)
    m.minimize(-profit)

    return m, interv, s, b

def minp_2(optimizer,data):

    m, interv, s, b=base_minp_2(optimizer,data)
    # Constraint: time horizon constraint: projects must complete within the scheduling period
    #NOTE: Not needed. Already implicity in interval definition

    # Objective
    # Maximize profit: final inventory value minus total task costs
    # Task cost is counted only for active realizations (length > 0)
    profit = m.sum(data.revenue[k]*s[k,data.lastT] for k in data.K) \
           - m.sum(data.cost[i,j]*m.sum(m.gt(m.length(interv[i,j,q]),0) for q in data.Q[(i,j)])
                   for (i, j) in data.I_i_j_prod)
    m.minimize(-profit)

    return m, interv, s, b

def minlip_2(optimizer,data):

    m, interv, s, b=base_minlip_2(optimizer,data)
    # Constraint: time horizon constraint: projects must complete within the scheduling period
    #NOTE: Not needed. Already implicity in interval definition

    # Objective
    # As in MInP(2): maximize profit as final inventory value minus cost of active tasks
    profit = m.sum(data.revenue[k]*s[k,data.lastT] for k in data.K) \
           - m.sum(data.cost[i,j]*m.sum(m.gt(m.length(interv[i,j,q]),0) for q in data.Q[(i,j)])
                   for (i, j) in data.I_i_j_prod)
    m.minimize(-profit)

    return m, interv, s, b

# === Simplified STN formualtions (without optional tasks) ===

def mip_known_n(optimizer,data,n):

    m, x, s, b=base_mip_known_n(optimizer,data,n)

    # Constraint: time horizon constraint: projects must complete within the scheduling period
    for (i,j) in data.I_i_j_prod:
        for t in (tt for tt in data.T if tt>= data.lastT-data.tau[(i,j)]+1):
            m.constraint(x[i,j,t]==0)

    # Objective
    # Opposite to MIP: cost is computed using fixed number of executions n[i,j] instead of summing over x[i,j,t]
    profit = m.sum(data.revenue[k]*s[k,data.lastT] for k in data.K) \
           - m.sum(data.cost[i,j]*n[i,j] for (i, j) in data.I_i_j_prod)
    m.minimize(-profit)

    return m, x, s, b

def minp_1_known_n(optimizer,data,n):

    m, interv, s, b=base_minp_1_known_n(optimizer,data,n)

    # Constraint: time horizon constraint: projects must complete within the scheduling period
    #NOTE: Not needed. Already implicity in interval definition

    # Objective
    # Opposite to MInP(1): cost is computed using fixed number of realizations rather than checking interval length
    profit = m.sum(data.revenue[k]*s[k,data.lastT] for k in data.K) \
           - m.sum(data.cost[i,j]*m.sum(1 for _ in data.Q[(i,j)]) for (i, j) in data.I_i_j_prod)
    m.minimize(-profit)

    return m, interv, s, b

def minlip_1_known_n(optimizer,data,n):
 
    m, interv, s, b=base_minlip_1_known_n(optimizer,data,n)

    # Constraint: time horizon constraint: projects must complete within the scheduling period
    #NOTE: Not needed. Already implicity in interval definition

    # Objective
    # Opposite to MInLiP(1): cost is computed using fixed number of realizations rather than checking interval length
    profit = m.sum(data.revenue[k]*s[k,data.lastT] for k in data.K) \
           - m.sum(data.cost[i,j]*m.sum(1 for _ in data.Q[(i,j)]) for (i, j) in data.I_i_j_prod)
    m.minimize(-profit)

    return m, interv, s, b

def minp_2_known_n(optimizer,data,n):

    m, interv, s, b=base_minp_2_known_n(optimizer,data,n)

    # Constraint: time horizon constraint: projects must complete within the scheduling period
    #NOTE: Not needed. Already implicity in interval definition

    # Objective
    # Opposite to MInP(2): cost is computed using fixed number of realizations rather than checking interval length
    profit = m.sum(data.revenue[k]*s[k,data.lastT] for k in data.K) \
           - m.sum(data.cost[i,j]*m.sum(1 for _ in data.Q[(i,j)]) for (i, j) in data.I_i_j_prod)
    m.minimize(-profit)

    return m, interv, s, b

def minlip_2_known_n(optimizer,data,n):

    m, interv, s, b=base_minlip_2_known_n(optimizer,data,n)

    # Constraint: time horizon constraint: projects must complete within the scheduling period
    #NOTE: Not needed. Already implicity in interval definition
    
    # Objective
    # Opposite to MInLiP(2): cost is computed using fixed number of realizations rather than checking interval length
    profit = m.sum(data.revenue[k]*s[k,data.lastT] for k in data.K) \
           - m.sum(data.cost[i,j]*m.sum(1 for _ in data.Q[(i,j)]) for (i, j) in data.I_i_j_prod)
    m.minimize(-profit)

    return m, interv, s, b

# === Utilities ===

def obtain_relevant_acc_levels(eta_f, delta_f, tee: bool = True):
    """
    Generates relevant accuracy levels for discretizing a scheduling horizon.

    Parameters:
    - eta_f (float): Total scheduling horizon.
    - delta_f (float): Base time step.
    - tee (bool): If True, prints a summary table of selected accuracy levels.

    Returns:
    - relevant_acc_levels (List[int]): Selected accuracy levels where each level corresponds
      to a granularity of discretization (acc * delta_f).
    """

    # Initialize lists to store all possible accuracy levels and their derived metrics
    all_acc_levels = []  # Accuracy level (acc)
    all_d = []           # Time step size at each accuracy level (acc * delta_f)
    all_nT = []          # Number of discrete time points (floor(eta_f / d))
    all_eta = []         # Effective horizon covered (d * nT)

    acc = 1
    # Generate all accuracy levels where at least one time point fits in the horizon
    while math.floor(eta_f / (acc * delta_f)) > 0:
        all_acc_levels.append(acc)
        all_d.append(acc * delta_f)
        all_nT.append(math.floor(eta_f / (acc * delta_f)))
        all_eta.append(acc * delta_f * math.floor(eta_f / (acc * delta_f)))
        acc += 1

    # Filter out only the relevant accuracy levels
    relevant_acc_levels = []
    if tee:
        # If tee is True, prepare additional lists for printing
        relevant_d = []
        relevant_nT = []
        relevant_eta = []

    acc = len(all_acc_levels) - 1
    # Traverse backwards to select levels with meaningful changes in granularity
    while acc >= 0:
        # Include:
        # - the coarsest level (last)
        # - the finest level (first)
        # - any level where nT drops compared to the next finer level
        if (acc == len(all_acc_levels) - 1 or acc == 0 or (all_nT[acc] - all_nT[acc + 1] > 0)):
            relevant_acc_levels.append(all_acc_levels[acc])
            if tee:
                relevant_d.append(all_d[acc])
                relevant_nT.append(all_nT[acc])
                relevant_eta.append(all_eta[acc])
        acc -= 1

    # Print summary table if tee is enabled
    if tee:
        headers = ["acc_level", "d", "nT", "eta"]
        table = zip(relevant_acc_levels, relevant_d, relevant_nT, relevant_eta)
        print(tabulate(table, headers=headers, floatfmt=".2f", tablefmt="grid"))

    return relevant_acc_levels

# === Helpers ===

def flush_table_to_txt(filename, table_by_acc, formulation_keys):
    from pathlib import Path

    output_path = Path("./hexaly_benchmarking_results")
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = ["Obj", "Bound", "Gap%", "Time", "Status"]
    col_width = 12
    col_width_status = 30  # Wider column for status

    def format_cell(val, is_status=False):
        width = col_width_status if is_status else col_width
        if isinstance(val, float):
            return f"{val:.2f}".ljust(width)
        return str(val).ljust(width)

    with open(output_path / filename, "w") as f:
        headers = ["Accuracy".ljust(col_width)]
        for key in formulation_keys:
            headers.extend([
                f"F{key}_Obj".ljust(col_width),
                f"F{key}_Bound".ljust(col_width),
                f"F{key}_Gap%".ljust(col_width),
                f"F{key}_Time".ljust(col_width),
                f"F{key}_Status".ljust(col_width_status),
            ])
        f.write(" | ".join(headers) + "\n")
        f.write("-" * len(" | ".join(headers)) + "\n")

        for acc in sorted(table_by_acc):
            row = [str(acc).ljust(col_width)]
            for key in formulation_keys:
                vals = table_by_acc[acc].get(key, ["-", "-", "-", "-", "-"])
                for i, val in enumerate(vals):
                    row.append(format_cell(val, is_status=(i == 4)))
            f.write(" | ".join(row) + "\n")

def write_to_excel(filename, results_dict):
    output_path = Path("./hexaly_benchmarking_results")
    writer = pd.ExcelWriter(output_path / filename, engine='xlsxwriter')
    for key, records in results_dict.items():
        df = pd.DataFrame(records, columns=["Accuracy", "Objective", "Obj. bound", "Obj. gap", "Time [s]", "Status"])
        df.to_excel(writer, sheet_name=f"Formulation_{key}", index=False)
    writer.close()

def write_mip_n_excel(n_mip, data):
    output_path = Path("./hexaly_benchmarking_results")
    records = []
    for acc in n_mip:
        row = {"acc": acc}
        for (i, j) in data.I_i_j_prod:
            row[f"{i}_{j}"] = n_mip[acc][i,j]
        records.append(row)
    df = pd.DataFrame(records)
    df.to_excel(output_path / "mip_n.xlsx", index=False)

# === Main Benchmark Loop ===
if __name__ == '__main__':
    # ------------------------
    # Benchmarking Parameters
    # ------------------------
    eta_f = 120         # Scheduling horizon (in time units)
    delta_f = 1         # Base time step (in time units)
    time_limit = 300    # Time limit for optimization (in seconds)
    seed = 1            # Random seed for reproducibility

    # ------------------------
    # Accuracy Levels
    # ------------------------
    # Generate relevant discretization granularities
    relevant_acc_levels = obtain_relevant_acc_levels(eta_f, delta_f,tee=True)

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

    known_n_formulations = {
        1: mip_known_n,
        2: minp_1_known_n,
        3: minlip_1_known_n,
        4: minp_2_known_n,
        5: minlip_2_known_n
    }

    # ------------------------
    # Result Containers
    # ------------------------
    original_results = {}     # Stores raw results for unknown-n formulations
    known_n_results = {}      # Stores results for known-n formulations
    n_mip = {acc_level: {} for acc_level in relevant_acc_levels}  # Stores extracted n from MIP

    original_table = {}       # Tabular format for unknown-n results
    known_table = {}          # Tabular format for known-n results

    # ------------------------
    # Benchmarking Loop
    # ------------------------
    for acc in relevant_acc_levels:
        # Run all unknown-n formulations
        for key, formulation in unknown_n_formulations.items():
            data = Data(eta_f=eta_f, delta_f=delta_f, acc_level=acc)
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
                    acc, objective, objective_bound, objective_gap, comp_time, status
                ])
                original_table.setdefault(acc, {})[key] = [
                    objective, objective_bound, objective_gap, comp_time, status
                ]

                # Extract execution counts from MIP solution
                if key == 1:
                    n_mip[acc] = {
                        (i, j): sum(round(x[i, j, t].value) for t in data.T)
                        for (i, j) in data.I_i_j_prod
                    }

        # Run all known-n formulations using extracted n from MIP
        for key, formulation in known_n_formulations.items():
            data = Data(eta_f=eta_f, delta_f=delta_f, acc_level=acc)
            with HexalyOptimizer() as optimizer:
                m, x, s, b = formulation(optimizer, data, n_mip[acc])
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
                known_n_results.setdefault(key, []).append([
                    acc, objective, objective_bound, objective_gap, comp_time, status
                ])
                known_table.setdefault(acc, {})[key] = [
                    objective, objective_bound, objective_gap, comp_time, status
                ]

        # ------------------------
        # Save Intermediate Tables
        # ------------------------
        flush_table_to_txt("original.txt", original_table, list(unknown_n_formulations.keys()))
        flush_table_to_txt("known_n.txt", known_table, list(known_n_formulations.keys()))

    # ------------------------
    # Save Final Results
    # ------------------------
    write_to_excel("original.xlsx", original_results)
    write_to_excel("known_n.xlsx", known_n_results)

    # Save extracted execution counts from MIP
    write_mip_n_excel(n_mip, data)

    # Print extracted n values for inspection
    print(n_mip)