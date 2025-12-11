import collections
import itertools as itt
from collections.abc import Sequence

import igraph as ig
import joblib as jl
import numpy as np
import scipy.special
from tqdm import tqdm


def sliding_window(iterable, n):
    "Collect data into overlapping fixed-length chunks or blocks."
    # sliding_window('ABCDEFG', 4) → ABCD BCDE CDEF DEFG
    iterator = iter(iterable)
    window = collections.deque(itt.islice(iterator, n - 1), maxlen=n)
    for x in iterator:
        window.append(x)
        yield tuple(window)


def get_k_grams_in_trajectories(
    trajectories: Sequence[Sequence[np.integer | int]], k: int = 5
) -> list[list[tuple[int]]]:
    kgrams = []
    for traj in trajectories:
        kgrams.append([w for w in sliding_window(traj, k)])
    return kgrams


def compute_polar_distance(kgrams_a: list[tuple[int]], kgrams_b: list[tuple[int]]) -> float:
    merged_kgrams = sorted(set(kgrams_a + kgrams_b))
    kgrams_a_counter = collections.Counter(kgrams_a)
    kgrams_b_counter = collections.Counter(kgrams_b)

    vec_a = np.array([kgrams_a_counter.get(k, 0) for k in merged_kgrams], dtype=float)
    vec_b = np.array([kgrams_b_counter.get(k, 0) for k in merged_kgrams], dtype=float)

    # compute "polar distance" as explained in the paper, i.e., the arccos of cosine simil.
    dot_prod = np.dot(vec_a, vec_b)
    norm_1 = np.linalg.norm(vec_a)
    norm_2 = np.linalg.norm(vec_b)

    # clip just to be safe from float errors, since arccos would return nan for out
    # of range values
    cosine_similarity = np.clip(dot_prod / (norm_1 * norm_2), -1.0, 1.0)

    return np.arccos(cosine_similarity) / np.pi


def _process_user_pair(user_1, user_2, k: int) -> tuple[int, int, float] | None:
    user1_id, user1_traj = user_1
    user2_id, user2_traj = user_2
    kgrams = get_k_grams_in_trajectories((user1_traj, user2_traj), k=k)
    polar_distance = compute_polar_distance(kgrams[0], kgrams[1])
    if polar_distance < 0.5:
        return user1_id, user2_id, polar_distance
    return None


def cluster_sessions(user_trajectories: list[Sequence[np.integer | int]], n_jobs: int, parallel_batch_size=1000):
    # TODO: create a graph with users as nodes, and there are edges if the polar distance is lower than 0.5
    # For clustering, the paper uses divisive hierarchical clustering. We could either use or some other classical community discovery stuff

    named_user_trajs = list(enumerate(user_trajectories))

    combinations = itt.combinations(named_user_trajs, 2)
    n_combs = scipy.special.comb(len(named_user_trajs), 2)
    elements_per_batch = n_combs // n_jobs + 1

    # Flatten results
    edge_list = []
    for batch_result in results:
        edge_list.extend(batch_result)

    # create a graph from the edge list
    graph = ig.Graph.TupleList(edge_list, directed=False, weights=True)
