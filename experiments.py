import hashlib
import os
import random
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import abcvoting.abcrules as rules
import math
import numpy as np
from abcvoting.misc import CandidateSet
from abcvoting.preferences import Profile

import explanation_systems
import instances
import plotting
from explanation_systems import PriceSystem, ExplanationRule
from instances import Euclidean2DVCRGenerator

CACHE_DIR = "cached_results"
NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1)
slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
if slurm_cpus:
    NUM_WORKERS = int(slurm_cpus) - 2


def _get_cache_key(profile: Profile, committee: CandidateSet) -> str:
    """Generate a unique hash key for profile, committee pair."""
    committee_data = tuple(sorted(committee))
    profile_data = tuple(
        tuple(sorted(profile[i].approved)) for i in range(len(profile))
    )
    combined = (profile.num_cand, profile_data, committee_data)
    return hashlib.sha256(str(combined).encode()).hexdigest()[:16]


def _get_cache_path(explanation_system_name: str, cache_key: str) -> str:
    """Get the file path for a cached result."""
    cache_subdir = os.path.join(CACHE_DIR, explanation_system_name)
    os.makedirs(cache_subdir, exist_ok=True)
    return os.path.join(cache_subdir, f"{cache_key}.npz")


def _load_cached_result(cache_path: str) -> PriceSystem | None:
    """Load a cached PriceSystem if it exists."""
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            return PriceSystem(
                n=int(data["n"]),
                m=int(data["m"]),
                residuals=data["residuals"],
                payments=data["payments"]
            )
        except (zipfile.BadZipFile, OSError) as e:
            print(f"Warning: Cache file corrupted or incomplete, recomputing: {cache_path} ({e})")
            return None
    return None


def _save_cached_result(cache_path: str, result: PriceSystem) -> None:
    """Save a PriceSystem to cache."""
    np.savez(cache_path, n=result.n, m=result.m, residuals=result.residuals, payments=result.payments)


def compute_explanation_rule(explanation_rule: ExplanationRule, profile: Profile,
                             committee: CandidateSet) -> PriceSystem:
    """Compute explanation rule with caching."""
    cache_key = _get_cache_key(profile, committee)
    cache_path = _get_cache_path(explanation_rule.name, cache_key)

    cached = _load_cached_result(cache_path)
    if cached is not None:
        print("Using cached result")
        return cached

    print("No cached result found, computing explanation rule")
    match explanation_rule:
        case ExplanationRule.CONT_PHRAGMEN:
            result = explanation_systems.cont_phragmen(profile, committee)
        case ExplanationRule.APPROX_PRICEABILITY:
            result = explanation_systems.approximate_priceability(profile, committee)
        case ExplanationRule.CONVEX:
            result = explanation_systems.convex_program(profile, committee)
        case ExplanationRule.WEIGHTED_SPLIT:
            result = explanation_systems.weighted_split(profile, committee)
        case ExplanationRule.EQUAL_SPLIT:
            result = explanation_systems.equal_split(profile, committee)
        case _:
            raise ValueError("Unsupported explanation rule")

    _save_cached_result(cache_path, result)
    return result


def compute_alpha_ejr_plus(profile: Profile, committee: CandidateSet,
                           budgets: np.ndarray | None = None) -> float:
    """Compute alpha-EJR+ value for a profile and committee, optionally weighted by budgets if given."""
    k, n = len(committee), len(profile)
    approval_counts = np.array([len(profile[v].approved & committee) for v in range(n)])

    # Compute weights by budgets provided, otherwise uniform weight 1
    weights = np.ones(n, dtype=int)
    if budgets is not None:
        total_budgets = budgets.sum()
        weights = budgets * (n / total_budgets)

    max_alpha = 0.0  # Largest alpha value for which violation exists
    for c in profile.approved_candidates().difference(committee):
        supporters_mask = np.array([c in profile[v].approved for v in range(n)])
        for l in range(1, k + 1):
            not_satisfied = supporters_mask & (approval_counts < l)

            weight_sum = weights[not_satisfied].sum()
            if weight_sum > 0:
                max_alpha = max(max_alpha, (weight_sum * k) / (l * n))

    return max_alpha


def compute_price_system_measures(
        ps: PriceSystem, profile: Profile, committee: CandidateSet,
        check_one_cost_stability: bool = True
) -> dict[str, float | np.ndarray]:
    """
    Compute general measures for a given price system and instance.
    Returns a dict with:
    - 'avg_residual': Average residual across voters
    - 'budget_non_uniformity': Sum of pairwise absolute budget differences
    - 'avg_payment_std': Average std of payments per candidate over supporters
    - 'max_deviation_money': Array of max available money for deviation to each c not in W
    """
    n, m = ps.n, ps.m
    avg_residual = ps.residuals.mean()
    budget_non_uniformity = ps.compute_budget_non_uniformity()

    # Average std of payments per candidate
    payment_stds = []
    for c in committee:
        supporters = [v for v in range(n) if c in profile[v].approved]
        if len(supporters) > 1:
            payments_to_c = ps.payments[supporters, c]
            if np.allclose(payments_to_c, payments_to_c[0]):
                payment_stds.append(0.0)
            else:
                payment_stds.append(payments_to_c.std())
    avg_payment_std = np.mean(payment_stds) if payment_stds else 0.0

    # max available money for deviation to each unelected candidate
    unelected = [c for c in range(m) if c not in committee]
    max_deviation_money = []

    for c in unelected:
        supporters_c = [v for v in range(n) if c in profile[v].approved]
        if not supporters_c:
            continue

        available_stability = ps.residuals[supporters_c].sum()
        max_one_cost = 0.0
        for c_prime in committee:
            supporters_c_prime = set(v for v in range(n) if c_prime in profile[v].approved)
            supporters_c_set = set(supporters_c)

            exclusive_supporters = supporters_c_set - supporters_c_prime  # V[c] \ V[c']
            intersection_supporters = supporters_c_set & supporters_c_prime  # V[c] ∩ V[c']
            deviation_money = (
                    ps.residuals[list(exclusive_supporters)].sum() +
                    ps.payments[list(intersection_supporters), c_prime].sum()
            )
            max_one_cost = max(max_one_cost, deviation_money)

        # Take max over residual stability and all 1-stability deviations for c
        max_deviation_money.append(max(available_stability, max_one_cost))

    return {
        'avg_residual': avg_residual,
        'budget_non_uniformity': budget_non_uniformity,
        'avg_payment_std': avg_payment_std,
        'max_deviation_money': np.array(max_deviation_money),
    }


def analysis_budgets_ejr_plus(instances_list: list[tuple[Profile, CandidateSet]], dataset_name: str,
                              explanation_rules: list[ExplanationRule],
                              clip_percentile: int | None = 95):
    # Compute uniform alpha-EJR+ for all instances
    instances_with_alpha = []
    uniform_alpha_values = []
    for profile, committee in instances_list:
        alpha_uniform = compute_alpha_ejr_plus(profile, committee)
        instances_with_alpha.append((profile, committee, alpha_uniform))
        uniform_alpha_values.append(alpha_uniform)

    # Plot uniform alpha-EJR+ histogram
    plotting.histogram(
        uniform_alpha_values,
        save_path=f"visualizations/ejr_plus_uniform/{dataset_name}.png",
        xlabel="alpha for EJR+",
        ylabel="Number of committees",
        title=f"EJR+ Violation (Uniform) - {dataset_name}"
    )

    # For each explanation system, compute budgets and do analysis
    for explanation_rule in explanation_rules:
        def process_instance(inst):
            profile, committee, alpha_uniform = inst
            ps = compute_explanation_rule(explanation_rule, profile, committee)
            check_one_cost = explanation_rule is not ExplanationRule.APPROX_PRICEABILITY
            measures = compute_price_system_measures(ps, profile, committee, check_one_cost_stability=check_one_cost)
            return ps.get_budgets(), alpha_uniform, measures

        num_workers_for_rule = NUM_WORKERS
        if explanation_rule is ExplanationRule.APPROX_PRICEABILITY:
            num_workers_for_rule = 2  # Limit parallelization for LP due to large model memory footprint
        print(f"Analysis for {explanation_rule.name} - parallelizing over {num_workers_for_rule} workers")
        with ThreadPoolExecutor(max_workers=num_workers_for_rule) as executor:
            results = list(executor.map(process_instance, instances_with_alpha))

        budgets = [r[0] for r in results]
        max_ejr_plus_violations = [r[1] for r in results]
        measures_list = [r[2] for r in results]

        # Collect price system measures
        all_max_deviation_money = []
        avg_residuals = []
        budget_non_uniformities = []
        avg_payment_stds = []
        for measures in measures_list:
            all_max_deviation_money.extend(measures['max_deviation_money'].tolist())
            avg_residuals.append(measures['avg_residual'])
            budget_non_uniformities.append(measures['budget_non_uniformity'])
            avg_payment_stds.append(measures['avg_payment_std'])

        # Compute min budget fractions and variance coefficients
        min_budget_fractions = []
        variance_coefficients = []
        for idx, (profile, committee) in enumerate(instances_list):
            k, n = len(committee), len(profile)
            b = budgets[idx]
            min_budget_fractions.append(b.min() / (k / n))
            variance_coefficients.append(b.std() / b.mean() if b.mean() > 0 else 0.0)

        # Analysis alpha-EJR+ vs. min budget fraction
        correlation = np.corrcoef(min_budget_fractions, max_ejr_plus_violations)[0, 1]
        print(f"Correlation between min budget fraction and max EJR+ violation: {correlation:.4f}")

        plotting.scatterplot_with_hist(
            min_budget_fractions,
            max_ejr_plus_violations,
            xlabel="Min Budget Fraction (of k/n)",
            ylabel="Min Alpha for EJR+",
            title=f"",
            save_path=f"visualizations/ejr_plus_vs_min_budget/{dataset_name}_{explanation_rule.name}.png",
            clip_percentile=clip_percentile
        )

        # Analysis alpha-EJR+ vs. variance coefficient
        correlation_vc = np.corrcoef(variance_coefficients, max_ejr_plus_violations)[0, 1]
        print(f"Correlation between variance coefficient and max EJR+ violation: {correlation_vc:.4f}")
        # Find the largest budget fraction with EJR+ violation (alpha > 1)
        violated = [i for i, alpha in enumerate(max_ejr_plus_violations) if alpha > 1]
        if violated:
            max_min_budget = max(min_budget_fractions[i] for i in violated)
            print(f"EJR+ violations: {len(violated)}/{len(max_ejr_plus_violations)}, "
                  f"max min budget fraction: {max_min_budget:.4f}")

        plotting.scatterplot_with_hist(
            variance_coefficients,
            max_ejr_plus_violations,
            xlabel="Variance Coefficient (std/mean)",
            ylabel="Min Alpha for EJR+",
            title=f"Budget Variance vs EJR+ (r={correlation_vc:.2f}) - {explanation_rule.name}, {dataset_name}",
            save_path=f"visualizations/ejr_plus_vs_var_coeff/{dataset_name}_{explanation_rule.name}.png",
            clip_percentile=clip_percentile
        )

        # Compute budget-weighted alpha-EJR+
        max_ejr_plus_weighted = []
        for idx, (profile, committee) in enumerate(instances_list):
            alpha_weighted = compute_alpha_ejr_plus(profile, committee, budgets[idx])
            max_ejr_plus_weighted.append(alpha_weighted)

        # Histogram of weighted alpha-EJR+ values
        plotting.histogram(
            max_ejr_plus_weighted,
            save_path=f"visualizations/ejr_plus_weighted/{dataset_name}_{explanation_rule.name}.png",
            xlabel="alpha for EJR+",
            ylabel="Number of committees",
            title=f"EJR+ Violation (Budget-Weighted) - {explanation_rule.name}, {dataset_name}"
        )

        # Plot price system measures
        plotting.price_system_measures_plot(
            all_max_deviation_money=all_max_deviation_money,
            avg_residuals=avg_residuals,
            budget_non_uniformities=budget_non_uniformities,
            avg_payment_stds=avg_payment_stds,
            save_path=f"visualizations/measures/{dataset_name}_{explanation_rule.name}.png",
            title=f"Price System Measures - {explanation_rule.name}, {dataset_name}"
        )


def weight_budgets_euclidean(
        explanation_rules: list[ExplanationRule],
        rule_ids: list[str], committee_fraction: float,
        num_samples: int, min_num_voters: int, max_num_voters: int,
        min_num_cand: int, max_num_cand: int,
        seed: int = 42):
    """Analyze budget-weight correlations for Euclidean profiles with gaussian weighting."""
    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    dataset_name = f"Euclidean2D_spatial_k={committee_fraction}"
    for rule_id in rule_ids:
        correlations = {er.name: [] for er in explanation_rules}
        alpha_ejr_plus_uniform = []
        alpha_ejr_plus_weighted = {er.name: [] for er in explanation_rules}
        measures_data = {er.name: {
            'all_max_deviation_money': [],
            'avg_residuals': [],
            'budget_non_uniformities': [],
            'avg_payment_stds': []
        } for er in explanation_rules}

        generator = Euclidean2DVCRGenerator(
            min_num_voters=min_num_voters, max_num_voters=max_num_voters,
            min_num_cand=min_num_cand, max_num_cand=max_num_cand,
            rng=py_rng
        )

        samples_generated = 0
        while samples_generated < num_samples:
            profile, voter_positions = generator.generate_with_positions()
            profile_unweighted = profile.copy()
            n = len(profile)
            m = profile.num_cand
            k = math.floor(m * committee_fraction)
            if k < 1 or k > len(profile.approved_candidates()):
                continue  # Skip invalid instances

            # Sample center and compute Gaussian weights by distance: exp(-d² / (2σ²))
            center = np_rng.uniform(-0.5, 0.5, size=2)
            distances = np.linalg.norm(voter_positions - center, axis=1)
            sigma = 0.3
            weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
            weights = np.maximum(weights, 0.1)  # Very small weights seem to trigger instability in MES

            for i in range(n):
                profile[i].weight = weights[i]

            # Run voting rule to get committee
            try:
                committee = rules.compute(rule_id, profile, k, resolute=True)[0]
            except ValueError as e:
                print(f"Error during MES computation: {e}")
                continue

            # Skip if some committee members have no supporters
            if not all(c in profile.approved_candidates() for c in committee):
                print(f"Skipping instance: {rule_id} produced committee with unsupported members")
                continue

            # Compute alpha-EJR+ for unweighted instance
            alpha_uniform = compute_alpha_ejr_plus(profile_unweighted, committee)
            alpha_ejr_plus_uniform.append(alpha_uniform)

            # Compute budgets for each explanation system
            for er in explanation_rules:
                ps = compute_explanation_rule(er, profile_unweighted, committee)
                budgets = ps.get_budgets()

                # Compute correlation between budgets and weights
                if budgets.std() > 0 and weights.std() > 0:
                    corr = np.corrcoef(budgets, weights)[0, 1]
                else:
                    corr = 0.0
                correlations[er.name].append(corr)

                # Compute budget-weighted alpha-EJR+
                alpha_weighted = compute_alpha_ejr_plus(profile_unweighted, committee, budgets)
                alpha_ejr_plus_weighted[er.name].append(alpha_weighted)

                # Collect price system measures
                check_one_cost = er is not ExplanationRule.APPROX_PRICEABILITY
                measures = compute_price_system_measures(ps, profile_unweighted, committee,
                                                         check_one_cost_stability=check_one_cost)
                measures_data[er.name]['all_max_deviation_money'].extend(
                    measures['max_deviation_money'].tolist())
                measures_data[er.name]['avg_residuals'].append(measures['avg_residual'])
                measures_data[er.name]['budget_non_uniformities'].append(measures['budget_non_uniformity'])
                measures_data[er.name]['avg_payment_stds'].append(measures['avg_payment_std'])

            samples_generated += 1

        # Plot correlation histograms for each explanation rule
        for er in explanation_rules:
            plotting.histogram(
                correlations[er.name],
                save_path=f"visualizations/budget_weight_correlation/{dataset_name}_{rule_id}_{er.name}.png",
                xlabel="Correlation (Budget vs Weight)",
                ylabel="Number of Instances",
                title=f"Budget vs Weight Correlation - {er.name}, {rule_id}"
            )

        # Print correlation statistics for each explanation rule
        print("\n" + "=" * 60)
        print(f"PCC for k={committee_fraction}")
        for er in explanation_rules:
            corrs = np.array(correlations[er.name])
            n_instances = len(corrs)
            n_very_strong = np.sum(corrs > 0.9)
            n_strong = np.sum(corrs > 0.7)
            n_moderate = np.sum(corrs > 0.4)
            pct_very_strong = 100 * n_very_strong / n_instances
            pct_strong = 100 * n_strong / n_instances
            pct_moderate = 100 * n_moderate / n_instances
            print(f"{er.name} for {n_instances} instances:")
            print(f"  PCC > 0.9: {n_very_strong}/{n_instances} ({pct_very_strong:.1f}%)")
            print(f"  PCC > 0.7: {n_strong}/{n_instances} ({pct_strong:.1f}%)")
            print(f"  PCC > 0.4: {n_moderate}/{n_instances} ({pct_moderate:.1f}%)")
        print("=" * 60 + "\n")

        # Plot uniform alpha-EJR+ histogram
        plotting.histogram(
            alpha_ejr_plus_uniform,
            save_path=f"visualizations/ejr_plus_uniform/{dataset_name}_{rule_id}.png",
            xlabel="alpha for EJR+",
            ylabel="Number of Instances",
            title=f"EJR+ Violation (Uniform) - {rule_id}"
        )

        # Plot budget-weighted alpha-EJR+ histograms
        for er in explanation_rules:
            plotting.histogram(
                alpha_ejr_plus_weighted[er.name],
                save_path=f"visualizations/ejr_plus_weighted/{dataset_name}_{rule_id}_{er.name}.png",
                xlabel="alpha for EJR+",
                ylabel="Number of Instances",
                title=f"EJR+ Violation (Budget-Weighted) - {er.name}, {rule_id}"
            )

        # Plot price system measures for each explanation rule
        for er in explanation_rules:
            m_data = measures_data[er.name]
            plotting.price_system_measures_plot(
                all_max_deviation_money=m_data['all_max_deviation_money'],
                avg_residuals=m_data['avg_residuals'],
                budget_non_uniformities=m_data['budget_non_uniformities'],
                avg_payment_stds=m_data['avg_payment_stds'],
                save_path=f"visualizations/measures/{dataset_name}_{rule_id}_{er.name}.png",
                title=f"Price System Measures - {er.name}, {rule_id}"
            )


def text_sum_bg(explanation_rules: list[ExplanationRule], print_full=False):
    profile, statements, topic_dict = instances.get_bg_instances()
    committee = instances.get_committee_bg_from_summarizer(method="LSA", k=10)

    n = len(statements)
    committee_size = len(committee)

    def get_topic(idx: int) -> str:
        topic = topic_dict.get(idx, "")
        return "[No Topic]" if topic == "" else topic

    def compute_topic_counts(indices) -> dict[str, int]:
        counts = defaultdict(int)
        for i in indices:
            counts[get_topic(i)] += 1
        return dict(counts)

    def print_topic_distribution(topic_counts: dict[str, int], total: int,
                                 dataset_context: dict[str, int] | None = None):
        for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = 100 * count / total
            output = f"{topic}: {count}/{total} ({percentage:.1f}%)"
            if dataset_context:
                dataset_count = dataset_context.get(topic, 0)
                dataset_pct = 100 * dataset_count / n
                output += f" [Dataset: {dataset_count}/{n} ({dataset_pct:.1f}%)]"
            print(output)

    # Print input statements
    print("=" * 80)
    print("INPUT STATEMENTS")
    print("=" * 80)
    for i in range(n):
        print(f"[{i}] Topic: '{get_topic(i)}' | Statement: {statements[i]}...")

    # Print profile
    if print_full:
        print("=" * 80)
        print("VOTER APPROVAL SETS")
        print("=" * 80)
        for voter_id in range(n):
            print(f"\nVoter {voter_id} - Topic: '{get_topic(voter_id)}'")
            print(f"  Statement: {statements[voter_id][:100]}...")
            print(f"  Approves:")
            for approved_id in sorted(profile[voter_id].approved):
                print(f"    [{approved_id}] '{get_topic(approved_id)}': {statements[approved_id][:80]}...")
    else:
        print("=" * 80)
        print("GENERATED PROFILE")
        print("=" * 80)
        for i in range(n):
            print(f"Voter {i}: {sorted(profile[i].approved)}")

    # Print topic distribution in dataset
    print("=" * 80)
    print("TOPIC DISTRIBUTION IN DATASET")
    print("=" * 80)
    topic_counts = compute_topic_counts(range(n))
    print_topic_distribution(topic_counts, n)

    # Print committee statements
    print("=" * 80)
    print("COMMITTEE STATEMENTS")
    print("=" * 80)
    for i in sorted(committee):
        print(f"[{i}] Topic: '{get_topic(i)}' | Statement: {statements[i]}")

    # Print topic distribution in committee
    print("=" * 80)
    print("TOPIC DISTRIBUTION IN COMMITTEE")
    print("=" * 80)
    committee_topic_counts = compute_topic_counts(committee)
    print_topic_distribution(committee_topic_counts, committee_size, topic_counts)

    for explanation_rule in explanation_rules:
        ps = compute_explanation_rule(explanation_rule, profile, committee)
        budgets = ps.get_budgets()

        print("=" * 80)
        print(f"EXPLANATION RULE: {explanation_rule.name}")
        print("\nPRICE SYSTEM:")
        print(f"\nBudgets:")

        # Print budgets for each voter
        for i in range(ps.n):
            total_payment = ps.payments[i].sum()
            print(f"  Voter {i}: budget={budgets[i]:.4f} "
                  f"(residual={ps.residuals[i]:.4f}, paid={total_payment:.4f})")

        # Compute and print average budget by topic
        print(f"\nAVERAGE BUDGET BY TOPIC:")
        topic_budget_sums = defaultdict(float)
        topic_voter_counts = defaultdict(int)

        for voter_id in range(ps.n):
            topic = get_topic(voter_id)
            topic_budget_sums[topic] += budgets[voter_id]
            topic_voter_counts[topic] += 1

        for topic in sorted(topic_budget_sums.keys(), key=lambda t: topic_counts.get(t, 0), reverse=True):
            avg_budget = topic_budget_sums[topic] / topic_voter_counts[topic]
            print(f"  {topic}: avg_budget={avg_budget:.4f} (voters={topic_voter_counts[topic]})")

        # Print voters with budget below k/n
        print(f"\nVOTERS WITH BUDGET BELOW K/N:")
        budget_threshold = committee_size / n
        below_avg = np.where(budgets < budget_threshold)[0]
        print(f"Count: {len(below_avg)}")
        for i in below_avg:
            print(f"  Voter {i} has budget={budgets[i]:.4f} - {statements[i]}")

        # Print payments
        print(f"\nPayments:")
        for i in range(ps.n):
            payments_for_voter = [
                f"c{cand_id}={ps.payments[i, cand_id]:.4f}"
                for cand_id in committee
                if ps.payments[i, cand_id] > 1e-6
            ]
            if payments_for_voter:
                print(f"  Voter {i}: {', '.join(payments_for_voter)}")
