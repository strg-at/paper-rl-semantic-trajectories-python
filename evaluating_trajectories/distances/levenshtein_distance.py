from collections.abc import Sequence
from typing import Any, Callable
import operator

import numpy as np
import numpy.typing as npt
from numba import njit, prange

DistanceFn = Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], float | np.floating]


@njit
def euclid_sim(vector_a: npt.NDArray[np.floating], vector_b: npt.NDArray[np.floating]) -> np.floating:
    return np.linalg.norm(vector_a - vector_b)


@njit
def cos_dist(vector_a: npt.NDArray[np.floating], vector_b: npt.NDArray[np.floating]) -> np.floating:
    r"""
    `Cosine similarity <https://en.wikipedia.org/wiki/Cosine_similarity#Definition>`_ implementation, where, given the ``threshold`` :math:`t`:

    This function is compiled with `numba <https://numba.readthedocs.io/>`_

    .. math::
      \text{cos} &= \frac{A \cdot B}{||A|| \, ||B||} \\ \\
      \text{cos} &= \begin{cases}
        \text{cos}, & \text{if cos} > t \\
        -abs(\text{cos}), & \text{if cos} < t \\
        -1, & \text{if cos} = 0
        \end{cases}

    """
    cos = (vector_a @ vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))  # type: ignore
    return 1 - cos


@njit
def exact_comparison_distance(element_a: int, element_b: int) -> float:
    return int(element_a == element_b)


@njit
def levenshtein_distance(
    trajectory_a: list[npt.NDArray[np.floating]],
    trajectory_b: list[npt.NDArray[np.floating]],
    dist_fn: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], float],
    penalty=1,
) -> tuple[float, npt.NDArray[np.floating]]:
    """
    Compute the Levenshtein distance algorithm.

    This function is compiled with `numba <https://numba.readthedocs.io/>`_

    :param trajectory_a: first trajectory
    :param trajectory_b: second trajectory
    :param dist_fn: similarity function. Should be something like :py:func:`cos_dist`.
    :return: returns the computed scores, path and the Needleman-Wunsch matrix.
    """
    n_rows = len(trajectory_a) + 1
    n_cols = len(trajectory_b) + 1
    nw_matrix = np.zeros((n_rows, n_cols), dtype=float)
    for i in range(1, n_rows):
        nw_matrix[i, 0] = nw_matrix[i - 1, 0] + penalty
    for i in range(1, n_cols):
        nw_matrix[0, i] = nw_matrix[0, i - 1] + penalty

    # compute scores
    for i in range(1, n_rows):
        for j in range(1, n_cols):
            cost_down = nw_matrix[i - 1, j] + penalty
            cost_right = nw_matrix[i, j - 1] + penalty
            cost_diag = nw_matrix[i - 1, j - 1] + dist_fn(trajectory_a[i - 1], trajectory_b[j - 1])
            nw_matrix[i, j] = min(cost_down, cost_right, cost_diag)

    i = n_rows - 1
    j = n_cols - 1
    score = nw_matrix[i, j]
    max_score = penalty * max(n_rows - 1, n_cols - 1)
    if max_score > 0:
        score /= max_score
    return score, nw_matrix


@njit(parallel=True)
def mean_levenshtein_distance(
    trajectories_a: list[list[npt.NDArray[np.floating]]],
    trajectories_b: list[list[npt.NDArray[np.floating]]],
    dist_fn: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], float],
    penalty=2,
) -> float:
    """
    Compute the mean Levenshtein distance algorithm.

    This function is compiled with `numba <https://numba.readthedocs.io/>`_

    :param trajectory_a: list of first trajectories, e.g. expert trajectories
    :param trajectory_b: list of second trajectories, e.g. imitated trajectories
    :param dist_fn: distance function. Should be something like :py:func:`cos_dist`.
    :return: returns the mean levenshtein distance over all trajectory pairs
    """
    n_rows = len(trajectories_a)
    n_cols = len(trajectories_b)
    scores = np.zeros((n_rows, n_cols), dtype=float)
    for i in prange(n_rows):
        for j in range(n_cols):
            scores[i, j] = levenshtein_distance(trajectories_a[i], trajectories_b[j], dist_fn, penalty)[  # type: ignore
                0
            ]
    return np.mean(scores).item()


def compare_groups(group_1, group_2, graph, cos_dist):
    results = {}
    for group1_id, group1_traj in group_1.items():
        for group2_id, group2_traj in group_2.items():
            group_1_res = results.setdefault(group1_id, {})
            ig_group1 = operator.itemgetter(*group1_traj)
            ig_group2 = operator.itemgetter(*group2_traj)
            embs_1 = list(map(lambda d: d["textEmbedding"], ig_group1(graph.vs)))
            embs_2 = list(map(lambda d: d["textEmbedding"], ig_group2(graph.vs)))
            group_1_res[group2_id] = levenshtein_distance(embs_1, embs_2, cos_dist, penalty=2)
    return results
