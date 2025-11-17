# === General STN formualtions (with optional tasks) ===
        
def base_mip(optimizer,data):

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

    return m, x, s, b

def base_minp_1(optimizer,data):
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


    return m, interv, s, b

def base_minlip_1(optimizer,data):
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

    return m, interv, s, b

def base_minp_2(optimizer,data):
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



    return m, interv, s, b

def base_minlip_2(optimizer,data):

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



    return m, interv, s, b

# === Simplified STN formualtions (without optional tasks) ===
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

def base_mip_known_n(optimizer,data,n):
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


    return m, x, s, b

def base_minp_1_known_n(optimizer,data,n):
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


    return m, interv, s, b

def base_minlip_1_known_n(optimizer,data,n):
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


    return m, interv, s, b

def base_minp_2_known_n(optimizer,data,n):
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


    return m, interv, s, b

def base_minlip_2_known_n(optimizer,data,n):
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


    return m, interv, s, b
