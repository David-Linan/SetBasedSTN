import math
from hexaly.optimizer import HexalyOptimizer, HxInterval, HxParam,HxStatistics
import pyomo.environ as pe
from tabulate import tabulate
import numpy as np
import pandas as pd
from pathlib import Path

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

def update_data(data, n):
    """
    Updates the data object with a specific execution plan.

    Parameters:
    - data (Data): An instance of the Data class containing model parameters.
    - n (dict): A dictionary mapping (task, unit) pairs to the number of executions.

    Functionality:
    - Filters out inactive task-unit pairs (where n[i,j] == 0).
    - Updates task and unit sets to reflect only active elements.
    - Prunes all relevant dictionaries to include only active pairs.
    - Recomputes processing times, batch bounds, and cost parameters.
    - Updates realization bounds and ranges for interval-based models.
    - Flags whether the given n is feasible based on upper/lower bounds.
    """
    # Add n to data
    data.n = n

    # Filter active (i, j) pairs
    active_pairs = {key for key, val in n.items() if val > 0}

    # Extract active i's and j's based on active pairs
    active_I = {i for (i, j) in active_pairs}
    active_J = {j for (i, j) in active_pairs}

    # Update I and J lists to contain only active elements
    data.I = [i for i in data.I if i in active_I]
    data.J = [j for j in data.J if j in active_J]

    # Update I_i_j_prod
    data.I_i_j_prod = {key: 1 for key in active_pairs}

    # Helper to clean up dictionaries
    def prune_dict(d):
        return {k: v for k, v in d.items() if k in active_pairs}

    data.tau_p = prune_dict(data.tau_p)
    data.tau   = {k: math.ceil(data.tau_p[k] / data.delta) for k in data.tau_p}
    data.beta_min = prune_dict(data.beta_min)
    data.beta_max = prune_dict(data.beta_max)
    data.cost     = prune_dict(data.cost)

    # Update q bounds and ranges
    data.upper_n = {(i,j): math.floor(data.lastT / data.tau[(i,j)]) for (i,j) in active_pairs}
    data.lower_n = {(i,j): 0 for (i,j) in active_pairs}
    data.lower_q = {(i,j): 1 for (i,j) in active_pairs}
    data.upper_q = {(i,j): n[(i,j)] for (i,j) in active_pairs}
    data.Q       = {(i,j): range(data.lower_q[(i,j)], data.upper_q[(i,j)] + 1) for (i,j) in active_pairs}

    # Check feasibility of n
    data.feasible_n = True
    for (i,j) in active_pairs:
        if n[i,j] < data.lower_n[i,j] or n[i,j] > data.upper_n[i,j]:
            data.feasible_n = False
            break
     
# === General STN formualtions (with optional tasks) ===
        
def mip(optimizer,data):

    # Declare the optimization model
    m = optimizer.model

    # Variable: timing
    # x[i,j,t] is a binary variable indicating whether task i on unit j starts at time t
    x = {(i,j,t):m.bool() for (i, j) in data.I_i_j_prod for t in data.T}

    # Variable: amount
    # b[i,j,t] is the amount processed by task i on unit j at time t, bounded by beta_max
    b = {(i, j, t): m.float(0, data.beta_max[(i, j)]) for (i, j) in data.I_i_j_prod for t in data.T}
    
    # Constraint: Non-overlapping
    # Ensures that at most one task is active on unit j at any time t
    for j in data.J:
        f_one_task_at_a_time = m.lambda_function(
            lambda t: m.sum(
                m.sum(
                    m.iif(m.and_(t_aux <= t, t_aux >= t - data.tau[i,j] + 1), x[i,j,t_aux], 0)
                    for t_aux in data.T
                )
                for i in data.I if (i, j) in data.I_i_j_prod
            ) <= 1
        )
        m.constraint(m.and_(m.array(data.T), f_one_task_at_a_time))

    # Constraint: task-unit capacity
    # Enforces that if task i is active at time t on unit j, the processed amount b[i,j,t]
    # must lie within [beta_min, beta_max] bounds
    for (i,j) in data.I_i_j_prod:
        for t in data.T:
            m.constraint(data.beta_min[i,j]*x[i,j,t] <= b[i,j,t])
            m.constraint(data.beta_max[i,j]*x[i,j,t] >= b[i,j,t])

    s = {}
    for k, t in ((k, t) for k in data.K for t in data.T):
        # Expressions: storage state
        # Computes inventory level s[k,t] for material k at time t
        if t == data.firstT:
            # Initial inventory
            s[k,t] = data.S0[k] \
                - m.sum(data.rho_minus[i,k]*b[i,j,t]
                       for i, j in ((i, j) for (i, j) in data.I_i_j_prod if (i,k) in data.I_i_k_minus)) \
                + data.replenishment[k,t] \
                - data.demand[k,t]
        else:
            # Inventory update
            s[k,t] = s[k,t-1] \
                + m.sum(data.rho_plus[i,k]*b[i,j,t-data.tau[i,j]]
                       for i, j in ((i, j) for (i, j) in data.I_i_j_prod
                                    if (i,k) in data.I_i_k_plus and t-data.tau[i,j] >= data.firstT)) \
                - m.sum(data.rho_minus[i,k]*b[i,j,t]
                       for i, j in ((i, j) for (i, j) in data.I_i_j_prod if (i,k) in data.I_i_k_minus)) \
                + data.replenishment[k,t] \
                - data.demand[k,t]

        # Constraints: state tracking
        # Keeps inventory within specified bounds
        m.constraint(s[k,t] <= data.upper_s[k])   
        m.constraint(s[k,t] >= data.lower_s[k])      

    # Objective
    # Maximize profit: final inventory value minus total task costs
    profit = m.sum(data.revenue[k]*s[k,data.lastT] for k in data.K) \
           - m.sum(data.cost[i,j]*m.sum(x[i,j,t] for t in data.T) for (i, j) in data.I_i_j_prod)
    m.minimize(-profit)

    return m, x, s, b

def minp_1(optimizer,data):
    #
    # Declare the optimization model
    #
    m = optimizer.model

    # Variable: timing
    # interv[i,j,q] is an interval variable representing the execution window of realization q of task i on unit j
    # Unlike MIP, timing is modeled directly as intervals rather than binary start indicators
    interv = {(i, j, q): m.interval(data.firstT, data.lastT) for (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]}

    # Variable: amount
    # b[i,j,q] is the amount processed by realization q of task i on unit j
    # Unlike MIP, these are defined per interval rather than per time step
    b = {(i, j, q): m.float(0, data.beta_max[(i, j)]) for (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]}

    # Constraint: task-unit capacity
    # Enforces that each realization is either active with fixed duration and bounded batch size,
    # or inactive with zero duration and zero batch size
    for i, j, q in ((i, j, q) for (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]):
        m.constraint(
            m.or_(
                # Active realization: fixed duration and batch size within bounds
                m.and_(
                    m.eq(m.length(interv[(i, j, q)]), data.tau[(i, j)]),
                    m.geq(b[i,j,q], data.beta_min[i,j]),
                    m.leq(b[i,j,q], data.beta_max[i,j])
                ),
                # Inactive realization: zero duration and zero batch size
                m.and_(
                    m.eq(m.length(interv[(i, j, q)]), 0),
                    m.eq(b[i,j,q], 0)
                )
            )
        )

    # Constraint: Non-overlapping
    # Ensures that at most one task is active on unit j at any time t
    # Expressed directly using interval boundaries, without auxiliary list variables
    for j in data.J:
        f_one_task_at_a_time = m.lambda_function(
            lambda t: m.sum(
                m.geq(t, m.start(interv[i,j,q])) - m.geq(t, m.end(interv[i,j,q]))
                for i in data.I if (i, j) in data.I_i_j_prod for q in data.Q[i,j]
            ) <= 1
        )
        m.constraint(m.and_(m.array(data.T), f_one_task_at_a_time))


    s = {}
    for k, t in ((k, t) for k in data.K for t in data.T):
        # Expressions: storage state
        # Computes inventory level s[k,t] for material k at time t
        # Unlike MIP, these have been modified considering that b[i,j,q] is indexed over q (realization) rather than time
        if t == data.firstT: # Initial inventory
            s[k,t] = data.S0[k] \
                - m.sum(m.iif(m.eq(t, m.start(interv[i,j,q])), data.rho_minus[i,k]*b[i,j,q], 0)
                        for i, j, q in ((i, j, q) for (i, j) in data.I_i_j_prod
                                        if (i,k) in data.I_i_k_minus for q in data.Q[(i, j)])) \
                + data.replenishment[k,t] \
                - data.demand[k,t]

        else: # Inventory update
            s[k,t] = s[k,t-1] \
                + m.sum(m.iif(m.eq(t, m.end(interv[i,j,q])), data.rho_plus[i,k]*b[i,j,q], 0)
                        for i, j, q in ((i, j, q) for (i, j) in data.I_i_j_prod
                                        if (i,k) in data.I_i_k_plus for q in data.Q[(i, j)])) \
                - m.sum(m.iif(m.eq(t, m.start(interv[i,j,q])), data.rho_minus[i,k]*b[i,j,q], 0)
                        for i, j, q in ((i, j, q) for (i, j) in data.I_i_j_prod
                                        if (i,k) in data.I_i_k_minus for q in data.Q[(i, j)])) \
                + data.replenishment[k,t] \
                - data.demand[k,t]

        # Constraints: state tracking
        # Keeps inventory within specified bounds
        m.constraint(s[k,t] <= data.upper_s[k])   
        m.constraint(s[k,t] >= data.lower_s[k])  

    # Objective
    # Maximize profit: final inventory value minus total task costs
    # Task cost is counted only for active realizations (length > 0)
    profit = m.sum(data.revenue[k]*s[k,data.lastT] for k in data.K) \
           - m.sum(data.cost[i,j]*m.sum(m.gt(m.length(interv[i,j,q]),0) for q in data.Q[(i,j)])
                   for (i, j) in data.I_i_j_prod)
    m.minimize(-profit)

    return m, interv, s, b

def minlip_1(optimizer,data):
    #
    # Declare the optimization model
    #
    m = optimizer.model

    # Variable: timing
    # interv[i,j,q] is an interval variable representing the execution window of realization q of task i on unit j
    # As in MInP(1), timing is modeled directly as intervals
    interv = {(i, j, q): m.interval(data.firstT, data.lastT) for (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]}

    # Variable: amount
    # b[i,j,q] is the amount processed by realization q of task i on unit j
    # As in MInP(1), these are defined per interval rather than per time step
    b = {(i, j, q): m.float(0, data.beta_max[(i, j)]) for (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]}

    # Variable: list
    # order[j] is a list variable representing the sequence of active tasks on unit j
    # Unlike MInP(1), list variables are introduced to model optional tasks and sequencing
    order = {j: m.list(sum(data.upper_n[(i, j)] for i in data.I if (i, j) in data.I_i_j_prod)) for j in data.J}

    for j in data.J:
        # Arrays for unit j: intervals, durations, batch sizes, and bounds
        interv_array = m.array([interv[i, j, q] for i in data.I if (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]])
        tau_array = m.array([data.tau[(i, j)] for i in data.I if (i, j) in data.I_i_j_prod for _ in data.Q[(i, j)]])
        b_array = m.array([b[i,j,q] for i in data.I if (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]])
        b_min_array = m.array([data.beta_min[(i, j)] for i in data.I if (i, j) in data.I_i_j_prod for _ in data.Q[(i, j)]])
        b_max_array = m.array([data.beta_min[(i, j)] for i in data.I if (i, j) in data.I_i_j_prod for _ in data.Q[(i, j)]])

        # Constraint: Non-overlapping
        # Ensures that tasks in the list are sequenced without overlap
        f_one_task_at_a_time = m.lambda_function(
            lambda pos: interv_array[order[j][pos-1]] < interv_array[order[j][pos]]
        )
        m.constraint(m.and_(m.range(1, m.count(order[j])), f_one_task_at_a_time))

        # Constraint: interval length
        # Ensures that interval length is zero if task is not in the list, or tau if it is
        f_non_zero_interval_length = m.lambda_function(
            lambda pos: m.length(interv_array[pos]) == m.contains(order[j], pos) * tau_array[pos]
        )
        m.constraint(m.and_(m.range(0, m.count(interv_array)), f_non_zero_interval_length))

        # Constraint: task-unit capacity (lower bound)
        f_min_b = m.lambda_function(
            lambda pos: b_array[pos] >= m.contains(order[j], pos) * b_min_array[pos]
        )
        m.constraint(m.and_(m.range(0, m.count(b_array)), f_min_b))

        # Constraint: task-unit capacity (upper bound)
        f_max_b = m.lambda_function(
            lambda pos: b_array[pos] <= m.contains(order[j], pos) * b_max_array[pos]
        )
        m.constraint(m.and_(m.range(0, m.count(b_array)), f_max_b))


    s = {}
    for k, t in ((k, t) for k in data.K for t in data.T):
        # Expressions: storage state
        # Computes inventory level s[k,t] for material k at time t
        # As in MInP(1), these are indexed over interval realizations rather than time
        if t == data.firstT: # Initial inventory
            s[k,t] = data.S0[k] \
                - m.sum(m.iif(m.eq(t, m.start(interv[i,j,q])), data.rho_minus[i,k]*b[i,j,q], 0)
                        for i, j, q in ((i, j, q) for (i, j) in data.I_i_j_prod
                                        if (i,k) in data.I_i_k_minus for q in data.Q[(i, j)])) \
                + data.replenishment[k,t] \
                - data.demand[k,t]

        else: # Inventory update
            s[k,t] = s[k,t-1] \
                + m.sum(m.iif(m.eq(t, m.end(interv[i,j,q])), data.rho_plus[i,k]*b[i,j,q], 0)
                        for i, j, q in ((i, j, q) for (i, j) in data.I_i_j_prod
                                        if (i,k) in data.I_i_k_plus for q in data.Q[(i, j)])) \
                - m.sum(m.iif(m.eq(t, m.start(interv[i,j,q])), data.rho_minus[i,k]*b[i,j,q], 0)
                        for i, j, q in ((i, j, q) for (i, j) in data.I_i_j_prod
                                        if (i,k) in data.I_i_k_minus for q in data.Q[(i, j)])) \
                + data.replenishment[k,t] \
                - data.demand[k,t]

        # Constraints: state tracking
        # Keeps inventory within specified bounds
        m.constraint(s[k,t] <= data.upper_s[k])   
        m.constraint(s[k,t] >= data.lower_s[k])  

    # Objective
    # Maximize profit: final inventory value minus total task costs
    # Task cost is counted only for active realizations (length > 0)
    profit = m.sum(data.revenue[k]*s[k,data.lastT] for k in data.K) \
           - m.sum(data.cost[i,j]*m.sum(m.gt(m.length(interv[i,j,q]),0) for q in data.Q[(i,j)])
                   for (i, j) in data.I_i_j_prod)
    m.minimize(-profit)

    return m, interv, s, b

def minp_2(optimizer,data):
    #
    # Declare the optimization model
    #
    m = optimizer.model

    # Variable: timing
    # interv[i,j,q] is an interval variable representing the execution window of realization q of task i on unit j
    # Opposed to MIP: timing is modeled using intervals instead of binary variables
    interv = {(i, j, q): m.interval(data.firstT, data.lastT) for (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]}

    # Variable: amount
    # b[i,j,t] is the amount processed by task i on unit j at time t
    # As in MIP: amount variables are defined per time step
    b = {(i, j, t): m.float(0, data.beta_max[(i, j)]) for (i, j) in data.I_i_j_prod for t in data.T}

    # Constraint: task-unit capacity
    # Opposed to MIP: disjunctions are used to connect time-indexed amount variables to interval realizations
    for i, j, t in ((i, j, t) for (i, j) in data.I_i_j_prod for t in data.T):
        m.constraint(
            b[i,j,t] <= data.beta_max[(i, j)] * m.or_(
                m.and_(m.eq(t, m.start(interv[i,j,q])), m.neq(m.length(interv[(i, j, q)]), 0))
                for q in data.Q[(i,j)]
            )
        )
        m.constraint(
            b[i,j,t] >= data.beta_min[(i, j)] * m.or_(
                m.and_(m.eq(t, m.start(interv[i,j,q])), m.neq(m.length(interv[(i, j, q)]), 0))
                for q in data.Q[(i,j)]
            )
        )

    # Constraint: interval activation
    # Opposed to MIP: disjunctions are used to model optional tasks
    for i, j, q in ((i, j, q) for (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]):
        m.constraint(
            m.or_(
                m.and_(
                    m.eq(m.length(interv[(i, j, q)]), data.tau[(i, j)])
                ),
                m.and_(
                    m.eq(m.length(interv[(i, j, q)]), 0)
                )
            )
        )

    # Constraint: Non-overlapping
    # Opposed to MIP: expressed directly in terms of interval variables
    for j in data.J:
        f_one_task_at_a_time = m.lambda_function(
            lambda t: m.sum(
                m.geq(t, m.start(interv[i,j,q])) - m.geq(t, m.end(interv[i,j,q]))
                for i in data.I if (i, j) in data.I_i_j_prod for q in data.Q[i,j]
            ) <= 1
        )
        m.constraint(m.and_(m.array(data.T), f_one_task_at_a_time))

    # Expressions: storage state
    # Computes inventory level s[k,t] for material k at time t
    # As in MIP: inventory is indexed over time and uses time-indexed amount variables
    s = {}
    for k, t in ((k, t) for k in data.K for t in data.T):

        if t == data.firstT: # Initial inventory
            s[k,t] = data.S0[k] \
                - m.sum(data.rho_minus[i,k]*b[i,j,t]
                       for i, j in ((i, j) for (i, j) in data.I_i_j_prod
                                    if (i,k) in data.I_i_k_minus)) \
                + data.replenishment[k,t] \
                - data.demand[k,t]

        else: # Inventory update
            s[k,t] = s[k,t-1] \
                + m.sum(data.rho_plus[i,k]*b[i,j,t-data.tau[i,j]]
                       for i, j in ((i, j) for (i, j) in data.I_i_j_prod
                                    if (i,k) in data.I_i_k_plus and t-data.tau[i,j] >= data.firstT)) \
                - m.sum(data.rho_minus[i,k]*b[i,j,t]
                       for i, j in ((i, j) for (i, j) in data.I_i_j_prod
                                    if (i,k) in data.I_i_k_minus)) \
                + data.replenishment[k,t] \
                - data.demand[k,t]

        # Constraints: state tracking
        # Keeps inventory within specified bounds
        m.constraint(s[k,t] <= data.upper_s[k])   
        m.constraint(s[k,t] >= data.lower_s[k])  

    # Objective
    # Maximize profit: final inventory value minus total task costs
    # Task cost is counted only for active realizations (length > 0)
    profit = m.sum(data.revenue[k]*s[k,data.lastT] for k in data.K) \
           - m.sum(data.cost[i,j]*m.sum(m.gt(m.length(interv[i,j,q]),0) for q in data.Q[(i,j)])
                   for (i, j) in data.I_i_j_prod)
    m.minimize(-profit)

    return m, interv, s, b

def minlip_2(optimizer,data):

    #
    # Declare the optimization model
    #
    m = optimizer.model

    # Variable: timing
    # interv[i,j,q] is an interval variable representing realization q of task i on unit j
    # As in MInP(2): interval variables replace binary timing variables from MIP
    interv = {(i, j, q): m.interval(data.firstT, data.lastT) for (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]}

    # Variable: amount
    # b[i,j,t] is the amount processed by task i on unit j at time t
    # As in MInP(2): amount variables are defined per time step
    b = {(i, j, t): m.float(0, data.beta_max[(i, j)]) for (i, j) in data.I_i_j_prod for t in data.T}

    # Variable: order
    # order[j] is a list variable representing the execution order of tasks on unit j
    # Opposite to MInP(2): list variables are introduced to model optional tasks and sequencing
    order = {j: m.list(sum(data.upper_n[(i, j)] for i in data.I if (i, j) in data.I_i_j_prod)) for j in data.J}


    # Constraint: task-unit capacity
    # As in MInP(2): disjunctions connect time-indexed amount variables to interval realizations
    for i, j, t in ((i, j, t) for (i, j) in data.I_i_j_prod for t in data.T):
        m.constraint(
            b[i,j,t] <= data.beta_max[(i, j)] * m.or_(
                m.and_(m.eq(t, m.start(interv[i,j,q])), m.neq(m.length(interv[(i, j, q)]), 0))
                for q in data.Q[(i,j)]
            )
        )
        m.constraint(
            b[i,j,t] >= data.beta_min[(i, j)] * m.or_(
                m.and_(m.eq(t, m.start(interv[i,j,q])), m.neq(m.length(interv[(i, j, q)]), 0))
                for q in data.Q[(i,j)]
            )
        )


    for j in data.J:
        interv_array = m.array([interv[i, j, q] for i in data.I if (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]])
        tau_array = m.array([data.tau[(i, j)] for i in data.I if (i, j) in data.I_i_j_prod for _ in data.Q[(i, j)]])

        # Constraint: non-overlapping
        # Opposite to MInP(2): sequencing is enforced using list variables
        f_one_task_at_a_time = m.lambda_function(
            lambda pos: interv_array[order[j][pos-1]] < interv_array[order[j][pos]]
        )
        m.constraint(m.and_(m.range(1, m.count(order[j])), f_one_task_at_a_time))

        # Constraint: interval activation
        # Opposite to MInP(2): interval length is tied to list membership
        f_non_zero_interval_length = m.lambda_function(
            lambda pos: m.length(interv_array[pos]) == m.contains(order[j], pos) * tau_array[pos]
        )
        m.constraint(m.and_(m.range(0, m.count(interv_array)), f_non_zero_interval_length))


    s = {}
    for k, t in ((k, t) for k in data.K for t in data.T):

        # Expressions: storage state
        # As in MInP(2): inventory is indexed over time and updated using amount variables
        if t == data.firstT: # Initial inventory
            s[k,t] = data.S0[k] \
                - m.sum(data.rho_minus[i,k]*b[i,j,t]
                       for i, j in ((i, j) for (i, j) in data.I_i_j_prod
                                    if (i,k) in data.I_i_k_minus)) \
                + data.replenishment[k,t] \
                - data.demand[k,t]
        else: # Inventory update
            s[k,t] = s[k,t-1] \
                + m.sum(data.rho_plus[i,k]*b[i,j,t-data.tau[i,j]]
                       for i, j in ((i, j) for (i, j) in data.I_i_j_prod
                                    if (i,k) in data.I_i_k_plus and t-data.tau[i,j] >= data.firstT)) \
                - m.sum(data.rho_minus[i,k]*b[i,j,t]
                       for i, j in ((i, j) for (i, j) in data.I_i_j_prod
                                    if (i,k) in data.I_i_k_minus)) \
                + data.replenishment[k,t] \
                - data.demand[k,t]

        # Constraints: state tracking
        m.constraint(s[k,t] <= data.upper_s[k])   
        m.constraint(s[k,t] >= data.lower_s[k])  

    # Objective
    # As in MInP(2): maximize profit as final inventory value minus cost of active tasks
    profit = m.sum(data.revenue[k]*s[k,data.lastT] for k in data.K) \
           - m.sum(data.cost[i,j]*m.sum(m.gt(m.length(interv[i,j,q]),0) for q in data.Q[(i,j)])
                   for (i, j) in data.I_i_j_prod)
    m.minimize(-profit)

    return m, interv, s, b

# === Simplified STN formualtions (without optional tasks) ===

def mip_known_n(optimizer,data,n):
    update_data(data, n)
    #
    # Declare the optimization model
    #
    m = optimizer.model

    # Variable: timing
    # x[i,j,t] is a binary variable indicating whether task i on unit j starts at time t
    # As in MIP: timing is modeled using binary variables
    x = {(i,j,t):m.bool() for (i, j) in data.I_i_j_prod for t in data.T}

    # Variable: amount
    # b[i,j,t] is the amount processed by task i on unit j at time t
    # As in MIP: amount variables are defined per time step
    b = {(i, j, t): m.float(0, data.beta_max[(i, j)]) for (i, j) in data.I_i_j_prod for t in data.T}

    # Constraint: non-overlapping
    # As in MIP: ensures that at most one task is active on unit j at any time t
    for j in data.J:
        f_one_task_at_a_time = m.lambda_function(
            lambda t: m.sum(
                m.sum(
                    m.iif(m.and_(t_aux <= t, t_aux >= t - data.tau[i,j] + 1), x[i,j,t_aux], 0)
                    for t_aux in data.T
                )
                for i in data.I if (i, j) in data.I_i_j_prod
            ) <= 1
        )
        m.constraint(m.and_(m.array(data.T), f_one_task_at_a_time))

    # Constraint: task-unit capacity
    # As in MIP: enforces bounds on amount if task is active
    # Opposite to MIP: adds constraint to fix the number of executions of task i on unit j
    for (i,j) in data.I_i_j_prod:
        for t in data.T:
            m.constraint(data.beta_min[i,j]*x[i,j,t] <= b[i,j,t])
            m.constraint(data.beta_max[i,j]*x[i,j,t] >= b[i,j,t])
        m.constraint(m.sum(x[i,j,t] for t in data.T) == n[i,j])


    s = {}
    for k, t in ((k, t) for k in data.K for t in data.T):
        # Expressions: storage state
        # As in MIP: inventory is indexed over time and updated using time-indexed amount variables
        if t == data.firstT: # Initial inventory
            s[k,t] = data.S0[k] \
                - m.sum(data.rho_minus[i,k]*b[i,j,t]
                       for i, j in ((i, j) for (i, j) in data.I_i_j_prod
                                    if (i,k) in data.I_i_k_minus)) \
                + data.replenishment[k,t] \
                - data.demand[k,t]
        else: # Inventory update
            s[k,t] = s[k,t-1] \
                + m.sum(data.rho_plus[i,k]*b[i,j,t-data.tau[i,j]]
                       for i, j in ((i, j) for (i, j) in data.I_i_j_prod
                                    if (i,k) in data.I_i_k_plus and t-data.tau[i,j] >= data.firstT)) \
                - m.sum(data.rho_minus[i,k]*b[i,j,t]
                       for i, j in ((i, j) for (i, j) in data.I_i_j_prod
                                    if (i,k) in data.I_i_k_minus)) \
                + data.replenishment[k,t] \
                - data.demand[k,t]

        # Constraints: state tracking
        m.constraint(s[k,t] <= data.upper_s[k])   
        m.constraint(s[k,t] >= data.lower_s[k])  

    # Objective
    # Opposite to MIP: cost is computed using fixed number of executions n[i,j] instead of summing over x[i,j,t]
    profit = m.sum(data.revenue[k]*s[k,data.lastT] for k in data.K) \
           - m.sum(data.cost[i,j]*n[i,j] for (i, j) in data.I_i_j_prod)
    m.minimize(-profit)

    return m, x, s, b

def minp_1_known_n(optimizer,data,n):
    update_data(data, n)

    # Declare the optimization model
    m = optimizer.model

    # Variable: timing
    # interv[i,j,q] is an interval variable representing realization q of task i on unit j
    # As in MInP(1): timing is modeled using interval variables
    interv = {(i, j, q): m.interval(data.firstT, data.lastT) for (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]}

    # Variable: amount
    # b[i,j,q] is the amount processed by realization q of task i on unit j
    # As in MInP(1): amount variables are indexed per interval realization
    b = {(i, j, q): m.float(data.beta_min[(i, j)], data.beta_max[(i, j)]) for (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]}

    # Constraint: interval activation
    # Opposite to MInP(1): intervals are always active with fixed duration; optionality removed
    for i, j, q in ((i, j, q) for (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]):
        m.constraint(m.length(interv[i,j,q]) == data.tau[(i, j)])

    # Constraint: non-overlapping
    # As in MInP(1): ensures that at most one task is active on unit j at any time t
    for j in data.J:
        f_one_task_at_a_time = m.lambda_function(
            lambda t: m.sum(
                m.geq(t, m.start(interv[i,j,q])) - m.geq(t, m.end(interv[i,j,q]))
                for i in data.I if (i, j) in data.I_i_j_prod for q in data.Q[i,j]
            ) <= 1
        )
        m.constraint(m.and_(m.array(data.T), f_one_task_at_a_time))


    s = {}
    for k, t in ((k, t) for k in data.K for t in data.T):
        # Expressions: storage state
        # As in MInP(1): inventory is indexed over time and updated using interval-based amount variables
        if t == data.firstT: # Initial inventory
            s[k,t] = data.S0[k] \
                - m.sum(m.iif(m.eq(t, m.start(interv[i,j,q])), data.rho_minus[i,k]*b[i,j,q], 0)
                       for i, j, q in ((i, j, q) for (i, j) in data.I_i_j_prod
                                       if (i,k) in data.I_i_k_minus for q in data.Q[(i, j)])) \
                + data.replenishment[k,t] \
                - data.demand[k,t]
        else: # Inventory update
            s[k,t] = s[k,t-1] \
                + m.sum(m.iif(m.eq(t, m.end(interv[i,j,q])), data.rho_plus[i,k]*b[i,j,q], 0)
                       for i, j, q in ((i, j, q) for (i, j) in data.I_i_j_prod
                                       if (i,k) in data.I_i_k_plus for q in data.Q[(i, j)])) \
                - m.sum(m.iif(m.eq(t, m.start(interv[i,j,q])), data.rho_minus[i,k]*b[i,j,q], 0)
                       for i, j, q in ((i, j, q) for (i, j) in data.I_i_j_prod
                                       if (i,k) in data.I_i_k_minus for q in data.Q[(i, j)])) \
                + data.replenishment[k,t] \
                - data.demand[k,t]

        # Constraints: state tracking
        m.constraint(s[k,t] <= data.upper_s[k])   
        m.constraint(s[k,t] >= data.lower_s[k])  

    # Objective
    # Opposite to MInP(1): cost is computed using fixed number of realizations rather than checking interval length
    profit = m.sum(data.revenue[k]*s[k,data.lastT] for k in data.K) \
           - m.sum(data.cost[i,j]*m.sum(1 for _ in data.Q[(i,j)]) for (i, j) in data.I_i_j_prod)
    m.minimize(-profit)

    return m, interv, s, b

def minlip_1_known_n(optimizer,data,n):
    update_data(data, n)

    # Declare the optimization model
    m = optimizer.model

    # Variable: timing
    # interv[i,j,q] is an interval variable representing realization q of task i on unit j
    # As in MInLiP(1): timing is modeled using interval variables
    interv = {(i, j, q): m.interval(data.firstT, data.lastT) for (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]}

    # Variable: amount
    # b[i,j,q] is the amount processed by realization q of task i on unit j
    # As in MInLiP(1): amount variables are indexed per interval realization
    b = {(i, j, q): m.float(data.beta_min[(i, j)], data.beta_max[(i, j)]) for (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]}

    # Variable: order
    # order[j] is a list variable representing the execution order of tasks on unit j
    # As in MInLiP(1): list variables are used to enforce sequencing
    order = {j: m.list(sum(n[(i, j)] for i in data.I if (i, j) in data.I_i_j_prod)) for j in data.J}

    # Constraint: interval activation
    # Opposite to MInLiP(1): intervals are always active with fixed duration; optionality removed
    for i, j, q in ((i, j, q) for (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]):
        m.constraint(m.length(interv[i,j,q]) == data.tau[(i, j)])



    for j in data.J:
        # Opposite to MInLiP(1): list size is now fixed to match known number of executions
        m.constraint(m.count(order[j]) == sum(n[(i, j)] for i in data.I if (i, j) in data.I_i_j_prod))

        interv_array = m.array([interv[i, j, q] for i in data.I if (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]])

        # Constraint: non-overlapping
        # As in MInLiP(1): sequencing is enforced using list variables
        f_one_task_at_a_time = m.lambda_function(
            lambda pos: interv_array[order[j][pos-1]] < interv_array[order[j][pos]]
        )
        m.constraint(m.and_(m.range(1, m.count(order[j])), f_one_task_at_a_time))

    s = {}
    for k, t in ((k, t) for k in data.K for t in data.T):
        # Expressions: storage state
        # As in MInLiP(1): inventory is indexed over time and updated using interval-based amount variables
        if t == data.firstT: # Initial inventory
            s[k,t] = data.S0[k] \
                - m.sum(m.iif(m.eq(t, m.start(interv[i,j,q])), data.rho_minus[i,k]*b[i,j,q], 0)
                       for i, j, q in ((i, j, q) for (i, j) in data.I_i_j_prod
                                       if (i,k) in data.I_i_k_minus for q in data.Q[(i, j)])) \
                + data.replenishment[k,t] \
                - data.demand[k,t]
        else: # Inventory update
            s[k,t] = s[k,t-1] \
                + m.sum(m.iif(m.eq(t, m.end(interv[i,j,q])), data.rho_plus[i,k]*b[i,j,q], 0)
                       for i, j, q in ((i, j, q) for (i, j) in data.I_i_j_prod
                                       if (i,k) in data.I_i_k_plus for q in data.Q[(i, j)])) \
                - m.sum(m.iif(m.eq(t, m.start(interv[i,j,q])), data.rho_minus[i,k]*b[i,j,q], 0)
                       for i, j, q in ((i, j, q) for (i, j) in data.I_i_j_prod
                                       if (i,k) in data.I_i_k_minus for q in data.Q[(i, j)])) \
                + data.replenishment[k,t] \
                - data.demand[k,t]

        # Constraints: state tracking
        m.constraint(s[k,t] <= data.upper_s[k])   
        m.constraint(s[k,t] >= data.lower_s[k])  

    # Objective
    # Opposite to MInLiP(1): cost is computed using fixed number of realizations rather than checking interval length
    profit = m.sum(data.revenue[k]*s[k,data.lastT] for k in data.K) \
           - m.sum(data.cost[i,j]*m.sum(1 for _ in data.Q[(i,j)]) for (i, j) in data.I_i_j_prod)
    m.minimize(-profit)

    return m, interv, s, b

def minp_2_known_n(optimizer,data,n):
    update_data(data, n)

    # Declare the optimization model
    m = optimizer.model

    # Variable: timing
    # interv[i,j,q] is an interval variable representing realization q of task i on unit j
    # As in MInP(2): timing is modeled using interval variables
    interv = {(i, j, q): m.interval(data.firstT, data.lastT) for (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]}

    # Variable: amount
    # b[i,j,t] is the amount processed by task i on unit j at time t
    # As in MInP(2): amount variables are defined per time step
    b = {(i, j, t): m.float(0, data.beta_max[(i, j)]) for (i, j) in data.I_i_j_prod for t in data.T}

    # Constraint: task-unit capacity
    # As in MInP(2): disjunctions connect time-indexed amount variables to interval realizations
    # Opposite to MInP(2): intervals are always active with fixed duration; optionality removed
    for i, j, t in ((i, j, t) for (i, j) in data.I_i_j_prod for t in data.T):
        m.constraint(
            b[i,j,t] <= data.beta_max[(i, j)] * m.or_(
                m.eq(t, m.start(interv[i,j,q])) for q in data.Q[(i,j)]
            )
        )
        m.constraint(
            b[i,j,t] >= data.beta_min[(i, j)] * m.or_(
                m.eq(t, m.start(interv[i,j,q])) for q in data.Q[(i,j)]
            )
        )

    # Constraint: interval activation
    # Opposite to MInP(2): intervals are always active with fixed duration
    for i, j, q in ((i, j, q) for (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]):
        m.constraint(m.length(interv[i,j,q]) == data.tau[(i, j)])

    # Constraint: non-overlapping
    # As in MInP(2): ensures that at most one task is active on unit j at any time t
    for j in data.J:
        f_one_task_at_a_time = m.lambda_function(
            lambda t: m.sum(
                m.geq(t, m.start(interv[i,j,q])) - m.geq(t, m.end(interv[i,j,q]))
                for i in data.I if (i, j) in data.I_i_j_prod for q in data.Q[i,j]
            ) <= 1
        )
        m.constraint(m.and_(m.array(data.T), f_one_task_at_a_time))


    s = {}
    for k, t in ((k, t) for k in data.K for t in data.T):
        # Expressions: storage state
        # As in MInP(2): inventory is indexed over time and updated using time-indexed amount variables
        if t == data.firstT: # Initial inventory
            s[k,t] = data.S0[k] \
                - m.sum(data.rho_minus[i,k]*b[i,j,t]
                       for i, j in ((i, j) for (i, j) in data.I_i_j_prod
                                    if (i,k) in data.I_i_k_minus)) \
                + data.replenishment[k,t] \
                - data.demand[k,t]
        else: # Inventory update
            s[k,t] = s[k,t-1] \
                + m.sum(data.rho_plus[i,k]*b[i,j,t-data.tau[i,j]]
                       for i, j in ((i, j) for (i, j) in data.I_i_j_prod
                                    if (i,k) in data.I_i_k_plus and t-data.tau[i,j] >= data.firstT)) \
                - m.sum(data.rho_minus[i,k]*b[i,j,t]
                       for i, j in ((i, j) for (i, j) in data.I_i_j_prod
                                    if (i,k) in data.I_i_k_minus)) \
                + data.replenishment[k,t] \
                - data.demand[k,t]

        # Constraints: state tracking
        m.constraint(s[k,t] <= data.upper_s[k])   
        m.constraint(s[k,t] >= data.lower_s[k])  

    # Objective
    # Opposite to MInP(2): cost is computed using fixed number of realizations rather than checking interval length
    profit = m.sum(data.revenue[k]*s[k,data.lastT] for k in data.K) \
           - m.sum(data.cost[i,j]*m.sum(1 for _ in data.Q[(i,j)]) for (i, j) in data.I_i_j_prod)
    m.minimize(-profit)

    return m, interv, s, b

def minlip_2_known_n(optimizer,data,n):
    update_data(data, n)

    # Declare the optimization model
    m = optimizer.model

    # Variable: timing
    # interv[i,j,q] is an interval variable representing realization q of task i on unit j
    # As in MInLiP(2): timing is modeled using interval variables
    interv = {(i, j, q): m.interval(data.firstT, data.lastT) for (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]}

    # Variable: amount
    # b[i,j,t] is the amount processed by task i on unit j at time t
    # As in MInLiP(2): amount variables are defined per time step
    b = {(i, j, t): m.float(0, data.beta_max[(i, j)]) for (i, j) in data.I_i_j_prod for t in data.T}

    # Variable: order
    # order[j] is a list variable representing the execution order of tasks on unit j
    # As in MInLiP(2): list variables are used to enforce sequencing
    # Opposite to MInLiP(2): list size is now fixed to match known number of executions
    order = {j: m.list(sum(data.n[(i, j)] for i in data.I if (i, j) in data.I_i_j_prod)) for j in data.J}

    # Constraint: task-unit capacity
    # As in MInLiP(2): disjunctions connect time-indexed amount variables to interval realizations
    # Opposite to MInLiP(2): intervals are always active with fixed duration; optionality removed
    for i, j, t in ((i, j, t) for (i, j) in data.I_i_j_prod for t in data.T):
        m.constraint(
            b[i,j,t] <= data.beta_max[(i, j)] * m.or_(
                m.eq(t, m.start(interv[i,j,q])) for q in data.Q[(i,j)]
            )
        )
        m.constraint(
            b[i,j,t] >= data.beta_min[(i, j)] * m.or_(
                m.eq(t, m.start(interv[i,j,q])) for q in data.Q[(i,j)]
            )
        )

    # Constraint: interval activation
    # Opposite to MInLiP(2): intervals are always active with fixed duration
    for i, j, q in ((i, j, q) for (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]):
        m.constraint(m.length(interv[i,j,q]) == data.tau[(i, j)])



    for j in data.J:
        m.constraint(m.count(order[j]) == sum(data.n[(i, j)] for i in data.I if (i, j) in data.I_i_j_prod))

        interv_array = m.array([interv[i, j, q] for i in data.I if (i, j) in data.I_i_j_prod for q in data.Q[(i, j)]])


        # Constraint: non-overlapping
        # As in MInLiP(2): sequencing is enforced using list variables
        f_one_task_at_a_time = m.lambda_function(
            lambda pos: interv_array[order[j][pos-1]] < interv_array[order[j][pos]]
        )
        m.constraint(m.and_(m.range(1, m.count(order[j])), f_one_task_at_a_time))


    s = {}
    for k, t in ((k, t) for k in data.K for t in data.T):
        # Expressions: storage state
        # As in MInLiP(2): inventory is indexed over time and updated using time-indexed amount variables
        if t == data.firstT: # Initial inventory
            s[k,t] = data.S0[k] \
                - m.sum(data.rho_minus[i,k]*b[i,j,t]
                       for i, j in ((i, j) for (i, j) in data.I_i_j_prod
                                    if (i,k) in data.I_i_k_minus)) \
                + data.replenishment[k,t] \
                - data.demand[k,t]
        else: # Inventory update
            s[k,t] = s[k,t-1] \
                + m.sum(data.rho_plus[i,k]*b[i,j,t-data.tau[i,j]]
                       for i, j in ((i, j) for (i, j) in data.I_i_j_prod
                                    if (i,k) in data.I_i_k_plus and t-data.tau[i,j] >= data.firstT)) \
                - m.sum(data.rho_minus[i,k]*b[i,j,t]
                       for i, j in ((i, j) for (i, j) in data.I_i_j_prod
                                    if (i,k) in data.I_i_k_minus)) \
                + data.replenishment[k,t] \
                - data.demand[k,t]

        # Constraints: state tracking
        m.constraint(s[k,t] <= data.upper_s[k])   
        m.constraint(s[k,t] >= data.lower_s[k])  

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
    relevant_acc_levels = obtain_relevant_acc_levels(eta_f, delta_f)

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