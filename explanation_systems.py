import enum
from dataclasses import dataclass
import math
import mosek
import numpy as np
import numpy.typing as npt
from abcvoting.misc import CandidateSet
from abcvoting.preferences import Profile
import gurobipy as gp

EPS = 1e-7  # Numerical tolerance


class ExplanationRule(enum.Enum):
    CONT_PHRAGMEN = 1
    APPROX_PRICEABILITY = 2
    CONVEX = 3
    WEIGHTED_SPLIT = 4
    EQUAL_SPLIT = 5


@dataclass(frozen=True)
class PriceSystem:
    n: int
    m: int
    residuals: npt.NDArray[np.float64]  # shape (n,)
    payments: npt.NDArray[np.float64]  # shape (n, m)

    def __post_init__(self):
        if self.residuals.shape != (self.n,):
            raise ValueError(f"residuals shape {self.residuals.shape} != ({self.n},)")
        if self.payments.shape != (self.n, self.m):
            raise ValueError(f"payments shape {self.payments.shape} != ({self.n}, {self.m})")

    def get_budgets(self) -> npt.NDArray[np.float64]:
        return self.residuals + self.payments.sum(axis=1)

    def compute_budget_non_uniformity(self) -> float:
        budgets = self.get_budgets()
        return np.abs(budgets[:, None] - budgets).sum() / 2


def weighted_split(profile: Profile, committee: CandidateSet) -> PriceSystem:
    """Weighted split: payments inversely proportional to s_i = Σ_{c' ∈ A_i ∩ W} 1/|V[c']|."""
    n = len(profile)
    m = profile.num_cand
    payments = np.zeros((n, m))

    # Precompute |V[c]| for all candidates in committee
    supporter_counts = {c: sum(1 for v in range(n) if c in profile[v].approved) for c in committee}

    # Compute s_i for each voter: s_i = Σ_{c' ∈ A_i ∩ W} 1/|V[c']|
    s = np.zeros(n)
    for i in range(n):
        approved_in_committee = profile[i].approved & committee
        s[i] = sum(1.0 / supporter_counts[c] for c in approved_in_committee)

    # For each candidate c ∈ W, set payments p(i,c) ∝ 1/s_i for i ∈ V[c]
    for c in committee:
        supporters = [v for v in range(n) if c in profile[v].approved]
        # Weights are 1/s_i, normalize so total payment = 1
        weights = np.array([1.0 / s[v] for v in supporters])
        weights /= weights.sum()
        for idx, v in enumerate(supporters):
            payments[v, c] = weights[idx]

    # Add residuals via post-processing
    ps = PriceSystem(n=n, m=m, payments=payments, residuals=np.zeros(n))
    return post_process_residuals(ps, profile, committee)


def equal_split(profile: Profile, committee: CandidateSet) -> PriceSystem:
    """Equal split: cost of each candidate shared equally among its supporters."""
    n = len(profile)
    m = profile.num_cand
    payments = np.zeros((n, m))

    # For each candidate c ∈ W, each supporter pays 1/|V[c]|
    for c in committee:
        supporters = [v for v in range(n) if c in profile[v].approved]
        payment_per_supporter = 1.0 / len(supporters)
        for v in supporters:
            payments[v, c] = payment_per_supporter

    # Add residuals via post-processing
    ps = PriceSystem(n=n, m=m, payments=payments, residuals=np.zeros(n))
    return post_process_residuals(ps, profile, committee)


def convex_program(profile: Profile, committee: CandidateSet) -> PriceSystem:
    n = len(profile)
    m = profile.num_cand
    k = len(committee)

    # Use larger epsilon for numerical stability in log terms
    LOG_EPS = 1e-4

    with mosek.Env() as env:
        with env.Task(0, 0) as task:
            # Solver parameters for numerical robustness
            task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, 1e-6)
            task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, 1e-6)
            task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, 1e-6)
            ####### Variables #########
            # - r_i for i in [0, n)                    : indices [0, n)
            # - p_{i,c} for i in [0, n), c in [0, m)   : indices [n, n + n*m)
            num_vars = n + n * m
            task.appendvars(num_vars)

            def r_idx(i):
                """Helper function to get variable index for r_i (residual of voter i)"""
                return i

            def p_i_c_idx(i, c):
                """Helper function to get variable index for p_{i,c} (payment of voter i to candidate c)"""
                return n + i * m + c

            # Set bounds: all variables >= 0
            for j in range(num_vars):
                task.putvarbound(j, mosek.boundkey.lo, 0.0, 0.0)

            # Cap r_i <= k for each voter (may be unbounded e.g. if voter does not approve any unelected cand.)
            for i in range(n):
                task.putvarbound(r_idx(i), mosek.boundkey.ra, 0.0, float(k))

            ####### Constraints #########
            num_constraints = 0

            # For each candidate c,
            #   - if c in W: sum_i p_{i,c} = 1
            #   - if c not in W: sum_i p_{i,c} = 0
            for c in range(m):
                task.appendcons(1)
                indices = [p_i_c_idx(i, c) for i in range(n)]
                coeffs = [1.0] * n
                task.putarow(num_constraints, indices, coeffs)

                if c in committee:
                    task.putconbound(num_constraints, mosek.boundkey.fx, 1.0, 1.0)
                else:
                    task.putconbound(num_constraints, mosek.boundkey.fx, 0.0, 0.0)
                num_constraints += 1

            # For all i, for all c not approved by i: p_{i,c} = 0
            for i in range(n):
                for c in range(m):
                    if c not in profile[i].approved:
                        task.putvarbound(p_i_c_idx(i, c), mosek.boundkey.fx, 0.0, 0.0)

            # For single-supporter candidates in W: fix payment at 1.0 for numerical stability
            for c in committee:
                supporters = [v for v in range(n) if c in profile[v].approved]
                if len(supporters) == 1:
                    i = next(iter(supporters))  # Get the single supporter
                    task.putvarbound(p_i_c_idx(i, c), mosek.boundkey.fx, 1.0, 1.0)

            # Stability constraint: For each c not in W, sum_{i in supporters of c} r_i <= 1
            for c in range(m):
                if c not in committee:
                    supporters = [v for v in range(n) if c in profile[v].approved]
                    if len(supporters) > 0:
                        task.appendcons(1)
                        indices = [r_idx(i) for i in supporters]
                        coeffs = [1.0] * len(supporters)
                        task.putarow(num_constraints, indices, coeffs)
                        task.putconbound(num_constraints, mosek.boundkey.up, 0.0, 1.0)
                        num_constraints += 1

            # 1-stability constraint: For each c not in W and c' in W: sum_{i in V[c] \ V[c']} r_i + sum_{i in V[c']\cap  V[c] } p_{i,c'} <= 1
            for c in range(m):
                if c not in committee:
                    supporters_c = {v for v in range(n) if c in profile[v].approved}
                    if len(supporters_c) > 0:
                        for c_prime in committee:
                            supporters_c_prime = {v for v in range(n) if c_prime in profile[v].approved}
                            exclusive_supporters = supporters_c - supporters_c_prime  # V[c] \ V[c']: supporters of c but not c'
                            intersection_supporters = supporters_c & supporters_c_prime  # Supporters in intersection V[c']\cap  V[c]

                            if len(exclusive_supporters) > 0:
                                task.appendcons(1)
                                indices = []

                                # Add r_i for i in V[c] \ V[c']
                                for i in exclusive_supporters:
                                    indices.append(r_idx(i))

                                # Add p_{i,c'} for i in V[c'] V[c']\cap  V[c]
                                for i in intersection_supporters:
                                    indices.append(p_i_c_idx(i, c_prime))

                                coeffs = [1.0] * len(indices)
                                assert len(indices) == len(supporters_c)
                                task.putarow(num_constraints, indices, coeffs)
                                task.putconbound(num_constraints, mosek.boundkey.up, 0.0, 1.0)
                                num_constraints += 1

            ####### Objective #########
            # Objective: maximize sum_{i in V} log(r_i) + sum_{c in W, i in V[c]} log(p_{i,c})
            # For each summand, add aux. variable t with exponential cone constraint: t <= log(x) iff (x, 1, t) in K_exp

            # Count how many log terms we have (skip single-supporter candidates as their payment is fixed to 1)
            num_log_terms = n  # one for each r_i
            for c in committee:
                supporters = [v for v in range(n) if c in profile[v].approved]
                if len(supporters) > 1:  # Only count if not single-supporter
                    num_log_terms += len(supporters)

            # Add auxiliary variables for log terms
            aux_start_idx = num_vars
            task.appendvars(num_log_terms)

            aux_idx = 0

            # For each r_i: term log(r_i + epsilon)
            for i in range(n):
                t_var = aux_start_idx + aux_idx
                task.putvarbound(t_var, mosek.boundkey.fr, 0.0, 0.0)

                # Add exponential cone: (r_i + epsilon, 1, t_var) in K_exp
                # This enforces: t_var <= log(r_i + epsilon)
                task.appendafes(3)
                afe_count = task.getnumafe()

                task.putafefentry(afe_count - 3, r_idx(i), 1.0)  # AFE 0 = r_i
                task.putafeg(afe_count - 3, LOG_EPS)  # AFE 0  + epsilon
                task.putafeg(afe_count - 2, 1.0)  # AFE 1 = 1 (constant)
                task.putafefentry(afe_count - 1, t_var, 1.0)  # AFE 2 = t_var

                task.appendacc(
                    task.appendprimalexpconedomain(),
                    [afe_count - 3, afe_count - 2, afe_count - 1],
                    None
                )
                aux_idx += 1

            # For each (i, c) where c in W and i in V[c]: term log(p_{i,c} + epsilon)
            for c in committee:
                supporters = [v for v in range(n) if c in profile[v].approved]
                # Skip log terms for single-supporter candidates (payment forced to 1, log(1)=0)
                if len(supporters) == 1:
                    continue
                for i in supporters:
                    t_var = aux_start_idx + aux_idx
                    task.putvarbound(t_var, mosek.boundkey.fr, 0.0, 0.0)

                    # Add exponential cone: (p_{i,c} + epsilon, 1, t_var) in K_exp
                    # This enforces: t_var <= log(p_{i,c} + epsilon)
                    task.appendafes(3)
                    afe_count = task.getnumafe()

                    task.putafefentry(afe_count - 3, p_i_c_idx(i, c), 1.0)  # AFE 0 = p_{i,c}
                    task.putafeg(afe_count - 3, LOG_EPS)  # AFE 0 = + epsilon
                    task.putafeg(afe_count - 2, 1.0)  # AFE 1 = 1 (constant)
                    task.putafefentry(afe_count - 1, t_var, 1.0)  # AFE 2 = t_var

                    task.appendacc(
                        task.appendprimalexpconedomain(),
                        [afe_count - 3, afe_count - 2, afe_count - 1],
                        None
                    )
                    aux_idx += 1

            # Set objective: maximize sum of all auxiliary variables
            task.putobjsense(mosek.objsense.maximize)
            for i in range(num_log_terms):
                task.putcj(aux_start_idx + i, 1.0)

            # Solve
            task.optimize()

            # Check solution status
            solsta = task.getsolsta(mosek.soltype.itr)
            prosta = task.getprosta(mosek.soltype.itr)

            if solsta != mosek.solsta.optimal:
                raise RuntimeError(
                    f"[WARNING] Solution not optimal! Problem status: {prosta} - Solution status: {solsta}")

            # Get solution
            solution = [0.0] * task.getnumvar()
            task.getxx(mosek.soltype.itr, solution)

            # Extract residuals and payments
            residuals = np.array(solution[:n])
            payments = np.zeros((n, m))
            for i in range(n):
                for c in range(m):
                    payments[i, c] = solution[p_i_c_idx(i, c)]

            # Prune residuals to max of maximum total payment or budget of some bounded voter
            total_payments = payments.sum(axis=1)
            budgets = residuals + total_payments

            bounded_voters = [i for i in range(n) if residuals[i] <= k - EPS]
            if bounded_voters:
                target_budget = max(budgets[bounded_voters].max(), total_payments.max())
                for v in range(n):
                    if budgets[v] > target_budget:
                        residuals[v] -= (budgets[v] - target_budget)

            return PriceSystem(n=n, m=m, residuals=residuals, payments=payments)


def approximate_priceability(profile: Profile, committee: CandidateSet) -> PriceSystem:
    n = len(profile)
    m = profile.num_cand

    model = gp.Model()
    model.setParam("LogToConsole", 0)

    # Variables
    b = model.addVars(n, vtype=gp.GRB.CONTINUOUS, name="budget", lb=0)
    p = model.addVars(n, m, vtype=gp.GRB.CONTINUOUS, name="payment", lb=0)
    r = model.addVars(n, vtype=gp.GRB.CONTINUOUS, name="residual", lb=0)

    # Residual constraint
    model.addConstrs((r[i] == b[i] - gp.quicksum(p[i, c] for c in range(m)) for i in range(n)))

    # Budget differences for objective
    d = model.addVars([(i, j) for i in range(n) for j in range(i + 1, n)], vtype=gp.GRB.CONTINUOUS, name="diff", lb=0)
    model.addConstrs((d[i, j] >= b[i] - b[j] for i, j in d))
    model.addConstrs((d[i, j] >= b[j] - b[i] for i, j in d))

    # Constraints
    # C1: No payment to unapproved candidates
    for i in range(n):
        for c in range(m):
            if c not in profile[i].approved:
                model.addConstr(p[i, c] == 0)

    # C2: Budget limit
    model.addConstrs((gp.quicksum(p[i, c] for c in range(m)) <= b[i] for i in range(n)))

    # C3: Winning candidates get funded (price = 1)
    model.addConstrs((gp.quicksum(p[i, c] for i in range(n)) == 1.0 for c in committee))

    # C4: No payment for unelected
    model.addConstrs((gp.quicksum(p[i, c] for i in range(n)) == 0 for c in range(m) if c not in committee))

    # C5: residual stability constraint - unelected candidates not affordable
    for c in range(m):
        if c not in committee:
            supporters = [i for i in range(n) if c in profile[i].approved]
            if supporters:
                model.addConstr(gp.quicksum(r[i] for i in supporters) <= 1.0)

    # Objective: minimize sum of budget differences
    model.setObjective(gp.quicksum(d[i, j] for i, j in d), gp.GRB.MINIMIZE)

    model.optimize()

    if model.status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Optimization failed with status {model.status}")

    threads = model.Params.Threads
    threads_str = "auto" if threads == 0 else str(threads)
    print(
        f"Solved model for n={n} and m={m} with {threads_str} threads and peak memory usage {model.MaxMemUsed:.4f} GB.")
    # Extract solution
    residuals = np.array([r[i].X for i in range(n)])
    payments = np.array([[p[i, c].X for c in range(m)] for i in range(n)])

    return PriceSystem(n=n, m=m, residuals=residuals, payments=payments)


def cont_phragmen(profile: Profile, committee: CandidateSet) -> PriceSystem:
    explanation_rule = ContPhragmen(profile, committee)
    return explanation_rule.run_cont_phragmen()


def post_process_residuals(ps: PriceSystem, profile: Profile, committee: CandidateSet) -> PriceSystem:
    explanation_rule = ContPhragmen(profile, committee, initial_price_system=ps)
    explanation_rule.post_process_residuals()
    print("done post-processing residuals")
    return explanation_rule.get_result()


class ContPhragmen:
    def __init__(self, profile: Profile, committee: CandidateSet, initial_price_system: PriceSystem = None):
        # Instance
        self.n = len(profile)
        self.m = profile.num_cand
        self.committee = committee
        if not all(c in profile.approved_candidates() for c in committee):
            raise ValueError("All committee candidates must have supporters!")
        self.approval_matrix = np.array([[c in profile[i].approved for c in range(self.m)] for i in range(self.n)],
                                        dtype=bool)  # Store profile as approval matrix of shape (n, m)

        # Solution
        if initial_price_system is None:
            self.payments = np.zeros((self.n, self.m))
            self.residuals = np.zeros(self.n)
        else:
            if self.n != initial_price_system.n or self.m != initial_price_system.m:
                raise ValueError("Given price system must have same number of voters and candidates!")
            self.payments = initial_price_system.payments
            self.residuals = initial_price_system.residuals

        # Algorithm Variables
        self.earning = np.ones(self.n, bool)
        self.remaining_price = np.array([1.0 if c in committee else 0.0 for c in range(self.m)])
        self.is_unblocking: set[int] = set()  # Candidates whose supporters are currently unblocked to fund candidate.
        self.payment_dist = np.zeros((self.n,
                                      self.m))  # fraction of earning to each candidate
        self.residual_dist = np.zeros(self.n)  # fraction of earning to residual
        # Helpers/Aggregates
        self.inflow_per_cand = np.zeros(self.m)  # Aggregate total inflow per candidate (maintained from payment_dist)

    def get_result(self) -> PriceSystem:
        return PriceSystem(n=self.n, m=self.m, residuals=self.residuals, payments=self.payments)

    def _partition_supporters(self, c: int, c_prime: int) -> tuple[
        npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Partition supporters of candidate c based on whether they also support c_prime. """
        supporters = np.flatnonzero(self.approval_matrix[:, c])
        supports_both = self.approval_matrix[supporters, c_prime]
        supporters_both = supporters[supports_both]
        supporters_only_c = supporters[~supports_both]
        return supporters, supporters_both, supporters_only_c

    def _get_blockable_supporters(self, c: int) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Get supporters and blockable (earning and do not support cand in is_unblocking) supporters for candidate c."""
        supporters = np.flatnonzero(self.approval_matrix[:, c])
        supports_unblocking = self.approval_matrix[supporters][:, list(self.is_unblocking)].any(axis=1)
        blockable = self.earning[supporters] & ~supports_unblocking
        blockable_supporters = supporters[blockable]
        return supporters, blockable_supporters

    def _reset_distribution(self):
        self.payment_dist = np.zeros((self.n, self.m))
        self.residual_dist = np.zeros(self.n)

    def update_distribution(self):
        t_c_values = self.compute_t_c()
        self._reset_distribution()

        has_remaining_price = self.remaining_price > EPS

        for i in range(self.n):
            if self.earning[i]:
                supported_remaining = np.flatnonzero(self.approval_matrix[i, :] & has_remaining_price)
                if supported_remaining.size:
                    t_c_min = np.min(t_c_values[supported_remaining])
                    candidates_at_min = supported_remaining[
                        np.array([math.isclose(x, t_c_min) for x in t_c_values[supported_remaining]])]
                    self.payment_dist[i, candidates_at_min] = 1.0 / candidates_at_min.size
                else:
                    self.residual_dist[i] = 1.0

        self.inflow_per_cand = self.payment_dist.sum(axis=0)

    def compute_t_c(self) -> npt.NDArray[np.float64]:
        t_c_values = np.full(self.m, np.inf)
        earning_t_c = np.copy(self.earning)

        has_remaining_price = self.remaining_price > EPS

        while earning_t_c.any():
            active_supporters = self.approval_matrix[earning_t_c].sum(axis=0)
            unfunded_with_active_supp = np.flatnonzero(has_remaining_price & (active_supporters > 0))
            if not unfunded_with_active_supp.size:
                return t_c_values
            time_ratios = self.remaining_price[unfunded_with_active_supp] / active_supporters[unfunded_with_active_supp]
            t_min = np.min(time_ratios)
            candidates_at_min = unfunded_with_active_supp[np.array([math.isclose(x, t_min) for x in time_ratios])]

            t_c_values[candidates_at_min] = t_min
            earning_t_c &= ~self.approval_matrix[:, candidates_at_min].any(
                axis=1)  # Set earning to False for all supporters of some candidate_at_min

        return t_c_values

    def time_to_next_event(self) -> float:
        min_step = float("inf")
        # Time until a candidate is fully funded
        candidates_w_progress = (self.inflow_per_cand > 0.0) & (self.remaining_price > EPS)
        if candidates_w_progress.any():
            min_step = min(min_step,
                           (self.remaining_price[candidates_w_progress] / self.inflow_per_cand[
                               candidates_w_progress]).min())

        # Time until a candidate causes a residual stability or 1-stability violation
        for c in range(self.m):
            time_to_block_c = self.time_to_blocking(c)[0]
            if time_to_block_c == 0.0: return 0.0
            min_step = min(min_step, time_to_block_c)

        return min_step

    def time_to_blocking(self, c: int) -> tuple[float, list[int]]:
        """Returns time when under current distribution a deviation to c would be a 1-stability violation and returns the blockable voters contributing to it."""
        min_step = float("inf")  # smallest time until deviation to c
        contributors = []  # voters who need to be blocked since letting them continue to earn causes violation under current dist
        if c in self.committee:
            return float("inf"), []

        supporters, blockable_supporters = self._get_blockable_supporters(c)

        if blockable_supporters.size:
            # Stability violation
            available_now = self.residuals[supporters].sum()
            increase = self.residual_dist[supporters].sum()
            contributors_blockable = blockable_supporters[self.residual_dist[
                                                              blockable_supporters] > 0.0]
            if contributors_blockable.size and increase > 0.0:
                if available_now >= 1.0:
                    return 0.0, contributors_blockable.tolist()
                time_delta = (1.0 - available_now) / increase
                if time_delta < min_step:
                    min_step = time_delta
                    contributors = contributors_blockable.tolist()

            # 1-stability violation
            for c_prime in self.committee:
                _, supporters_both, supporters_only_c = self._partition_supporters(c, c_prime)

                available_now = self.payments[supporters_both, c_prime].sum() + self.residuals[supporters_only_c].sum()
                increase = self.payment_dist[supporters_both, c_prime].sum() + self.residual_dist[
                    supporters_only_c].sum()
                contrib_residual = supporters_only_c[self.residual_dist[supporters_only_c] > 0.0]
                contrib_payment = supporters_both[self.payment_dist[supporters_both, c_prime] > 0.0]
                contributors_blockable = np.intersect1d(np.concatenate([contrib_residual, contrib_payment]),
                                                        blockable_supporters)

                if contributors_blockable.size and increase > 0.0:
                    if available_now >= 1.0:
                        return 0.0, contributors_blockable.tolist()
                    time_delta = (1.0 - available_now) / increase
                    if time_delta < min_step:
                        min_step = time_delta
                        contributors = contributors_blockable.tolist()
        return min_step, contributors

    def check_and_block(self) -> None:
        voters_to_block = []
        for c in range(self.m):
            blocking_check = self.time_to_blocking(c)
            if blocking_check[0] < EPS:
                voters_to_block.extend(blocking_check[1])
        if voters_to_block:
            self.earning[voters_to_block] = False

    def unblock_supporters(self) -> None:
        voters_to_unblock = []
        cand_to_check = np.flatnonzero((self.remaining_price > EPS))
        for c in cand_to_check.tolist():
            supporters = np.flatnonzero(self.approval_matrix[:, c])
            if not self.earning[supporters].any():
                voters_to_unblock.extend(supporters)
                self.is_unblocking.add(c)
        if voters_to_unblock:
            self.earning[voters_to_unblock] = True

    def run_cont_phragmen(self) -> PriceSystem:
        while np.any(self.remaining_price > EPS):
            self.update_distribution()
            dt = self.time_to_next_event()
            assert dt < float("inf")

            # Advance by time step dt
            self.payments += dt * self.payment_dist
            self.residuals += dt * self.residual_dist
            self.remaining_price -= dt * self.inflow_per_cand

            # Check if earning voters supporting unblocking candidates are paying to any candidate
            candidates_possible_violation = set()
            if self.is_unblocking:
                unblocking_list = list(self.is_unblocking)
                earning_supporting_unblocking = self.earning & self.approval_matrix[:, unblocking_list].any(axis=1)
                candidates_possible_violation.update(
                    np.flatnonzero((self.payment_dist[earning_supporting_unblocking] > 0).any(axis=0)))

            for c in candidates_possible_violation:
                self.reduce_residuals(c)

            for c in range(self.m):
                if self.remaining_price[c] < EPS:
                    self.is_unblocking.discard(c)

            self.update_distribution()
            self.check_and_block()
            self.unblock_supporters()

        self.post_process_residuals()
        return self.get_result()

    def reduce_residuals(self, c_prime: int):
        # Check if there is unelected c such that we have 1-stability violation with c_prime, then reduce residuals
        max_fraction_to_reduce = np.zeros(self.n)

        for c in range(self.m):
            if c in self.committee:
                continue

            _, supporters_both, supporters_only_c = self._partition_supporters(c, c_prime)

            violation_amount = (
                                       self.payments[supporters_both, c_prime].sum() +
                                       self.residuals[supporters_only_c].sum()) - 1
            if violation_amount > EPS:
                # Compute fraction to reduce so that violation amount would be zero
                total_residual = self.residuals[supporters_only_c].sum()
                if total_residual <= 0:
                    raise RuntimeError(f"Violation of {violation_amount} but no residual to reduce!")

                fraction_to_reduce = violation_amount / total_residual
                if fraction_to_reduce > 1 + EPS:
                    raise RuntimeError(f" Violation {violation_amount:.6f} exceeds residual!")
                fraction_to_reduce = min(fraction_to_reduce, 1.0)  # Correct slight numerical instabilities

                # Track max fraction for each voter
                max_fraction_to_reduce[supporters_only_c] = np.maximum(
                    max_fraction_to_reduce[supporters_only_c], fraction_to_reduce)

        # Apply all reductions at once using the max fractions
        self.residuals *= (1 - max_fraction_to_reduce)

    def post_process_residuals(self):
        self.earning = np.ones(self.n, dtype=bool)
        self.is_unblocking = set()

        budgets = self.residuals + self.payments.sum(axis=1)
        b_min_earning = budgets[self.earning].min()

        # Zero payment for candidates, only residuals change
        self.payment_dist = np.zeros((self.n, self.m))
        self.inflow_per_cand = np.zeros(self.m)

        while self.earning.any() and not (math.isclose(b_min_earning,
                                                       budgets.max())):

            # Set distributions: 1.0 for min-budget earners, 0.0 otherwise
            is_min_earner = self.earning & np.array([math.isclose(x, b_min_earning) for x in budgets])
            self.residual_dist = np.where(is_min_earner, 1.0, 0.0)

            time_to_blocking = self.time_to_next_event()

            # Time until we need to update min-budget earners or reach budget-uniformity
            earning_budgets = budgets[self.earning]
            not_min = ~np.array([math.isclose(x, b_min_earning) for x in earning_budgets])
            if not_min.any():
                second_min = earning_budgets[not_min].min()
                time_to_update_min = second_min - b_min_earning
            else:
                time_to_update_min = budgets.max() - b_min_earning
                if time_to_update_min <= time_to_blocking:
                    self.residuals += time_to_update_min * self.residual_dist
                    return  # Only max budget voters are earning, we are done

            dt = min(time_to_blocking, time_to_update_min)
            self.residuals += dt * self.residual_dist

            self.check_and_block()

            budgets = self.residuals + self.payments.sum(axis=1)
            if self.earning.any():
                b_min_earning = budgets[self.earning].min()
