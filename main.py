import enum
import random

import numpy as np
from abcvoting.misc import CandidateSet
from abcvoting.preferences import Profile

import experiments
import instances
from explanation_systems import ExplanationRule


class Datasets(enum.Enum):
    EUCLIDEAN_2D_VCR = 1
    RESAMPLING = 2
    PABULIB = 3


############# CONSTANTS ############
DATASET = [Datasets.EUCLIDEAN_2D_VCR, Datasets.RESAMPLING, Datasets.PABULIB]
EXPLANATION_RULES = [ExplanationRule.EQUAL_SPLIT, ExplanationRule.APPROX_PRICEABILITY, ExplanationRule.CONT_PHRAGMEN]
# Parameters for synthetic datasets
SAMPLES = 1000
MIN_NUM_VOTERS = 10
MAX_NUM_VOTERS = 100
MIN_NUM_CANDS = 10
MAX_NUM_CANDS = 100
COMMITTEE_FRACTIONS = [0.125, 0.25, 0.5]  # Committee will contain floor(fraction * m) candidates

# Parameters for Pabulib
MAX_NUM_VOTERS_PABULIB = 500

# Parameters for non-uniform weight experiment
RULES = ["equal-shares"]


############# END CONSTANTS ############
def get_instances(dataset: Datasets, committee_fraction: float) -> list[
    tuple[Profile, CandidateSet]]:
    match dataset:
        case Datasets.EUCLIDEAN_2D_VCR:
            generator = instances.Euclidean2DVCRGenerator(min_num_voters=MIN_NUM_VOTERS, max_num_voters=MAX_NUM_VOTERS,
                                                          min_num_cand=MIN_NUM_CANDS, max_num_cand=MAX_NUM_CANDS)
            return generator.generate_with_committees(SAMPLES, committee_fraction)
        case Datasets.RESAMPLING:
            generator = instances.ResamplingGenerator(min_num_voters=MIN_NUM_VOTERS, max_num_voters=MAX_NUM_VOTERS,
                                                      min_num_cand=MIN_NUM_CANDS, max_num_cand=MAX_NUM_CANDS)
            return generator.generate_with_committees(SAMPLES, committee_fraction)
        case Datasets.PABULIB:
            return instances.pabulib_with_random_committees(committee_fraction=committee_fraction,
                                                            max_n=MAX_NUM_VOTERS_PABULIB)
        case _:
            raise ValueError("Unsupported dataset")


def get_profiles(dataset: Datasets) -> list[Profile]:
    match dataset:
        case Datasets.EUCLIDEAN_2D_VCR:
            generator = instances.Euclidean2DVCRGenerator(min_num_voters=MIN_NUM_VOTERS, max_num_voters=MAX_NUM_VOTERS,
                                                          min_num_cand=MIN_NUM_CANDS, max_num_cand=MAX_NUM_CANDS)
            return generator.generate_many(num_samples=SAMPLES)
        case Datasets.RESAMPLING:
            generator = instances.ResamplingGenerator(min_num_voters=MIN_NUM_VOTERS, max_num_voters=MAX_NUM_VOTERS,
                                                      min_num_cand=MIN_NUM_CANDS, max_num_cand=MAX_NUM_CANDS)
            return generator.generate_many(num_samples=SAMPLES)
        case Datasets.PABULIB:
            return instances.load_pabulib(MAX_NUM_VOTERS_PABULIB)
        case _:
            raise ValueError("Unsupported dataset")


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    # Run experiments on relationship of budgets to EJR+
    for dataset in DATASET:
        for committee_fraction in COMMITTEE_FRACTIONS:
            input_instances = get_instances(dataset, committee_fraction)
            dataset_name = f"{dataset.name} with k= m*{committee_fraction}"
            print(f"EJR+ Analysis for {dataset_name}")
            experiments.analysis_budgets_ejr_plus(input_instances, dataset_name, EXPLANATION_RULES)

    # Run experiments for Euclidean spatial weighting
    for committee_fraction in COMMITTEE_FRACTIONS:
        print(f"Euclidean spatial weight analysis for k=m*{committee_fraction}")
        experiments.weight_budgets_euclidean(
            EXPLANATION_RULES, RULES, committee_fraction,
            num_samples=SAMPLES,
            min_num_voters=MIN_NUM_VOTERS, max_num_voters=MAX_NUM_VOTERS,
            min_num_cand=MIN_NUM_CANDS, max_num_cand=MAX_NUM_CANDS
        )

    # Run experiments for bowling green data
    experiments.text_sum_bg(EXPLANATION_RULES)
