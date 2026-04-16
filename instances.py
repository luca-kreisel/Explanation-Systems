import csv
import os
import random
from abc import abstractmethod
from pathlib import Path

import math
import numpy as np
import openai
from abcvoting.misc import CandidateSet
from abcvoting.preferences import Profile, Voter
from prefsampling.approval import euclidean_vcr
from prefsampling.approval import resampling as prefsampling_resampling
from prefsampling.core.euclidean import EuclideanSpace, sample_election_positions
from pabutools import election as pb_election

from sumy.nlp.stemmers import Stemmer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.utils import get_stop_words
from sumy.models.dom import Sentence, Paragraph, ObjectDocumentModel


class ProfileGenerator:
    def __init__(self, min_num_voters: int, max_num_voters: int,
                 min_num_cand: int, max_num_cand: int):
        self.min_num_voters = min_num_voters
        self.max_num_voters = max_num_voters
        self.min_num_cand = min_num_cand
        self.max_num_cand = max_num_cand

        self._count = 0  # Use as seed to ensure reproducibility between runs

    @abstractmethod
    def generate(self) -> Profile:
        """Generate a single profile."""
        pass

    def generate_many(self, num_samples: int) -> list[Profile]:
        """Generate multiple profiles."""
        return [self.generate() for _ in range(num_samples)]

    def generate_with_committees(self, num_samples: int, committee_fraction: float) -> list[
        tuple[Profile, CandidateSet]]:
        """Generate profiles with random committees."""
        samples = []
        while len(samples) < num_samples:
            profile = self.generate()
            m = profile.num_cand
            k = math.floor(m * committee_fraction)
            approved = profile.approved_candidates()
            if k > len(approved) or k == 0:
                continue
            committee = CandidateSet(random.sample(sorted(approved), k))
            samples.append((profile, committee))
        return samples


class ResamplingGenerator(ProfileGenerator):
    def generate(self) -> Profile:
        num_voters = random.randint(self.min_num_voters, self.max_num_voters)
        num_cand = random.randint(self.min_num_cand, self.max_num_cand)
        phi = random.random()
        p = random.random()
        votes = prefsampling_resampling(num_voters, num_cand, phi=phi, rel_size_central_vote=p, seed=self._count)
        profile = Profile(num_cand)
        for vote in votes:
            profile.add_voter(Voter(vote, num_cand=num_cand))
        self._count += 1
        return profile


class Euclidean2DVCRGenerator(ProfileGenerator):
    def __init__(self, min_num_voters: int, max_num_voters: int,
                 min_num_cand: int, max_num_cand: int, rng: random.Random | None = None):
        super().__init__(min_num_voters, max_num_voters, min_num_cand, max_num_cand)
        self.rng = rng if rng is not None else random  # Use provided RNG or global random state

    def generate(self) -> Profile:
        profile, _ = self.generate_with_positions()
        return profile

    def generate_with_positions(self) -> tuple[Profile, 'np.ndarray']:
        """Generate a profile and return voter positions as well."""
        num_voters = self.rng.randint(self.min_num_voters, self.max_num_voters)
        num_cand = self.rng.randint(self.min_num_cand, self.max_num_cand)
        voter_radius = self.rng.uniform(0.05, 0.3)

        # Sample positions first
        voter_positions, candidate_positions = sample_election_positions(
            num_voters, num_cand, 2,
            EuclideanSpace.UNIFORM_CUBE, EuclideanSpace.UNIFORM_CUBE,
            seed=self._count
        )

        # Use positions with euclidean_vcr
        votes = euclidean_vcr(
            num_voters,
            num_cand,
            voter_radius,
            0.0,
            2,
            voter_positions,  # Pass pre-sampled positions
            candidate_positions,
            seed=self._count
        )
        profile = Profile(num_cand)
        for vote in votes:
            profile.add_voter(Voter(vote, num_cand=num_cand))
        self._count += 1
        return profile, voter_positions

    def generate_many_with_positions(self, num_samples: int) -> list[tuple[Profile, 'np.ndarray']]:
        """Generate multiple profiles with their voter positions."""
        return [self.generate_with_positions() for _ in range(num_samples)]


def _read_pabulib_deterministic(file_path: str) -> tuple[Profile, int]:
    """Read pabulib file with deterministic candidate and voter ordering."""
    instance, pb_profile = pb_election.parse_pabulib(file_path)

    # Sort projects by name for deterministic candidate ordering
    sorted_projects = sorted(instance, key=lambda p: p.name)
    num_cand = len(sorted_projects)
    projects_to_cand = {p: i for i, p in enumerate(sorted_projects)}

    profile = Profile(num_cand=num_cand, cand_names=[p.name for p in sorted_projects])

    # Sort ballots for deterministic voter ordering
    ballots_with_multiplicity = [
        (pb_ballot, pb_profile.multiplicity(pb_ballot)) for pb_ballot in pb_profile
    ]
    ballots_with_multiplicity.sort(key=lambda x: tuple(sorted(projects_to_cand[a] for a in x[0])))

    for pb_ballot, multiplicity in ballots_with_multiplicity:
        for _ in range(multiplicity):
            profile.add_voter(Voter([projects_to_cand[a] for a in pb_ballot], num_cand=num_cand))

    return profile, int(instance.budget_limit)


def load_pabulib(max_n: int | None = None) -> list[Profile]:
    """Load pabulib profiles, subsampled to max_n voters if given."""
    profiles = []
    for f in sorted(Path("datasets/pabulib").glob("*.pb")):
        profile, _ = _read_pabulib_deterministic(str(f))
        n = len(profile)
        if max_n is not None and n > max_n:
            # Use deterministic seed based on filename for reproducible subsampling
            rng = random.Random(f.name)
            subsampled_profile = Profile(profile.num_cand)
            voter_indices = rng.sample(range(n), max_n)
            for i in voter_indices:
                subsampled_profile.add_voter(profile[i])
            profile = subsampled_profile
        profiles.append(profile)
    return profiles


def pabulib_with_random_committees(committee_fraction: float, max_n: int | None = None) -> list[
    tuple[Profile, CandidateSet]]:
    """Returns list of (profile, committee) tuples, subsampled to max_n voters if given."""
    profiles = load_pabulib(max_n)
    samples = []
    for profile in profiles:
        m = profile.num_cand
        k = math.floor(m * committee_fraction)
        approved = profile.approved_candidates()
        if k > len(approved) or k == 0:
            continue

        committee = CandidateSet(random.sample(sorted(approved), k))
        samples.append((profile, committee))
    return samples


def get_committee_bg_from_summarizer(method: str, k: int) -> CandidateSet:
    """Get size-k committee from Bowling Green dataset using specified summarizer method on statements."""
    LANGUAGE = "english"
    file_path = "datasets/bowling_green/comments_w_topics.csv"

    statements = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            statements.append(row[0].strip())

    # Setup summarizer
    tokenizer = Tokenizer(LANGUAGE)
    sentences = [s.replace('\n', '') for s in statements]
    document = ObjectDocumentModel([Paragraph([Sentence(s, tokenizer) for s in sentences])])

    stemmer = Stemmer(LANGUAGE)
    if method == 'TEXT_RANK':
        summarizer = TextRankSummarizer(stemmer)
    elif method == 'LUHN':
        summarizer = LuhnSummarizer(stemmer)
    elif method == 'LEX_RANK':
        summarizer = LexRankSummarizer(stemmer)
    elif method == 'LSA':
        summarizer = LsaSummarizer(stemmer)
    else:
        raise ValueError(f"Unknown method: {method}")

    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary = summarizer(document, k)

    # Get indices of selected statements
    summary_statements = set([str(s) for s in summary])
    summary_ids = set()
    for sentence_id, sentence in enumerate(sentences):
        if sentence in summary_statements:
            summary_ids.add(sentence_id)
            summary_statements.remove(sentence)
    return CandidateSet(summary_ids)


def get_bg_instances(similarity_percentile: float = 80.0) -> tuple[Profile, list[str], dict[int, str]]:
    file_path = "datasets/bowling_green/comments_w_topics.csv"

    statements = []
    topics = dict()
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        idx = 0
        for row in reader:
            statements.append(row[0].strip())
            topics[idx] = row[1].strip() if row[1].strip() else ""
            idx += 1

    embeddings = embedding_openai_api(statements)
    utility_matrix = np.dot(embeddings, embeddings.T)  # embeddings are normalized

    # compute threshold based on percentage
    n = utility_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    all_similarities = utility_matrix[mask]
    threshold = np.percentile(all_similarities, similarity_percentile)

    # Build approval profile based on threshold
    profile = Profile(num_cand=n)
    for voter_id in range(n):
        approved_candidates = np.where(utility_matrix[voter_id] > threshold)[0].tolist()
        profile.add_voter(Voter(approved_candidates, num_cand=n))

    return profile, statements, topics


def embedding_openai_api(
        statements: list[str],
        embeddings_path: str = "datasets/bowling_green/embeddings.npy") -> np.ndarray:
    """
    Generate embeddings using OpenAI API and cache them to a file, uses cached embeddings when available.
    """
    if os.path.exists(embeddings_path):
        print(f"Loading cached embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
        return embeddings

    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    client = openai.OpenAI()
    model = "text-embedding-3-large"

    n = len(statements)
    print(f"Generating OpenAI embeddings for {n} statements using model: {model}")

    try:
        response = client.embeddings.create(
            input=statements,
            model=model
        )

        embeddings = np.array([item.embedding for item in response.data])
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        np.save(embeddings_path, embeddings)
        print(f"Saved embeddings to {embeddings_path} with shape {embeddings.shape}", flush=True)
        return embeddings

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise


def sample_bowling_green_comments(
        input_path: str = "datasets/bowling_green/comments.csv",
        output_path: str = "datasets/bowling_green/comments_samples.txt",
        num_samples: int = 75) -> None:
    """Function to generate subsampled comment file"""
    comments = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 8 or "timestamp" in row:
                continue

            comment_body = row[7].strip().strip('"')
            if comment_body:
                comments.append(comment_body)

    print(f"Read {len(comments)} comments from {input_path}")

    sampled_comments = random.sample(comments, min(num_samples, len(comments)))
    print(f"Sampled {len(sampled_comments)} comments")

    # Write sampled comments to output file
    with open(output_path, "w", encoding="utf-8") as f:
        for comment in sampled_comments:
            f.write(comment + "\n")

    print(f"Saved sampled comments to {output_path}")
