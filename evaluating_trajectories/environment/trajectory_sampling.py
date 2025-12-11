import operator
import pickle
import random
from collections import Counter

import duckdb
import faiss
import igraph as ig
import numpy as np
import numpy.typing as npt
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from evaluating_trajectories.utils import consts


def load_environment(
    graph_filepath: str,
    trajectories_filepath: str,
    exclude_sessions: npt.NDArray,
) -> tuple[ig.Graph, list[npt.NDArray[np.integer]]]:
    with open(graph_filepath, "rb") as f:
        graph = pickle.load(f)

    # We use anti join to remove all sessions that we shouldn' take. If eexclude_sessions is empty, then nothing will be removed
    trajectories = duckdb.sql(
        f"SELECT trajectory FROM '{trajectories_filepath}' t ANTI JOIN exclude_sessions e ON t.user_session = e.column0"
    ).fetchnumpy()["trajectory"]

    return graph, trajectories


def is_trajectory_valid(trajectory: npt.NDArray[np.integer], graph: ig.Graph, min_length: int, max_length: int) -> bool:
    if len(trajectory) < min_length or len(trajectory) >= max_length:
        return False
    return np.all(trajectory < len(graph.vs))


def remove_consecutive_duplicate_node_visits(
    trajectories: list[npt.NDArray[np.integer]],
    return_as_list_of_idx=True,
) -> list[npt.NDArray[np.integer]]:
    filtered_trajectories = []
    for trajectory in trajectories:
        filtered_trajectory = [trajectory[0]] if not return_as_list_of_idx else [0]
        for i in range(1, len(trajectory)):
            if trajectory[i] != trajectory[i - 1]:
                if return_as_list_of_idx:
                    filtered_trajectory.append(i)
                else:
                    filtered_trajectory.append(trajectory[i])
        filtered_trajectories.append(np.array(filtered_trajectory))
    return filtered_trajectories


def _prepare_trajectory_embeddings(
    trajectories: list[npt.NDArray[np.integer]],
    graph: ig.Graph,
    remove_consecutive_duplicates: bool,
    min_traj_length: int,
    max_traj_length: int,
) -> tuple[list[npt.NDArray[np.integer]], npt.NDArray[np.floating]]:
    if remove_consecutive_duplicates:
        trajectories = remove_consecutive_duplicate_node_visits(trajectories, return_as_list_of_idx=False)

    trajectories = [
        t
        for t in tqdm(trajectories, desc="Random sampling: filtering out invalid trajectories")
        if t is not None and is_trajectory_valid(t, graph, min_traj_length, max_traj_length)
    ]

    # Extract mean embeddings for each trajectory
    # in order to later compute cosine similarity and do our sampling
    traj_embs = [
        np.array(graph.vs[t]["embedding"]).mean(axis=0)
        for t in tqdm(trajectories, desc="Random sampling: extracting trajectory embeddings")
    ]

    bool_mask = np.array(
        [
            not np.all(np.isnan(t))
            for t in tqdm(traj_embs, desc="Random sampling: filtering out trajectories with no embeddings")
        ]
    )
    keep_indices = np.where(bool_mask)[0]
    itg = operator.itemgetter(*keep_indices)

    filtered_trajs_id = itg(trajectories)
    filtered_trajs_emb = np.array(itg(traj_embs))
    return filtered_trajs_id, filtered_trajs_emb


def sample_with_faiss_kmeans(
    trajectories: list[npt.NDArray[np.integer]],
    graph: ig.Graph,
    num_trajectories: int,
    min_traj_length: int,
    max_traj_length: int,
    remove_consecutive_duplicates: bool = False,
    num_clusters: int = 500,
    select_cluster_closest_to_num_trajectories: bool = True,
) -> tuple[list[npt.NDArray[np.integer]], list[npt.NDArray[np.floating]], list[int]]:
    trajectories, trajectories_embs = _prepare_trajectory_embeddings(
        trajectories, graph, remove_consecutive_duplicates, min_traj_length, max_traj_length
    )
    kmeans = faiss.Kmeans(trajectories_embs.shape[1], num_clusters, niter=20, verbose=False, gpu=True)

    kmeans.train(trajectories_embs)
    _, clusters = kmeans.index.search(trajectories_embs, 1)

    clusters = clusters.flatten()

    if select_cluster_closest_to_num_trajectories:
        choosen_cluster: int = duckdb.sql(
            "SELECT column0 FROM (SELECT column0, count() c FROM clusters GROUP BY column0 ORDER BY c) WHERE c >= ? LIMIT 1",
            params=(num_trajectories,),
        ).fetchone()[
            0
        ]  # pyright: ignore[reportOptionalSubscript]
    else:
        choosen_cluster = np.random.choice(num_clusters, size=1, replace=False)[0]

    cluster_mask = np.where(clusters == choosen_cluster)[0]

    itg = operator.itemgetter(*cluster_mask)
    cluster_trajectories = itg(trajectories)
    cluster_trajectories_embs = itg(trajectories_embs)

    if len(cluster_trajectories) > num_trajectories:
        sampled_indices = np.random.choice(len(cluster_trajectories), num_trajectories, replace=False)
        itg = operator.itemgetter(*sampled_indices)
        cluster_trajectories = list(itg(cluster_trajectories))
        cluster_trajectories_embs = np.array(itg(cluster_trajectories_embs))

    sampled_cluster_trajectories_embs = [np.array(graph.vs[t]["embedding"]) for t in cluster_trajectories]

    return cluster_trajectories, sampled_cluster_trajectories_embs, [t[0] for t in cluster_trajectories]


def select_random_trajectory(
    trajectories: list[npt.NDArray[np.integer]], graph: ig.Graph, min_length: int, max_length: int
) -> npt.NDArray[np.integer]:
    random.shuffle(trajectories)
    trajectory = next(filter(lambda t: is_trajectory_valid(t, graph, min_length, max_length), trajectories))

    return trajectory


def sample_n_trajectories(
    trajectories: list[npt.NDArray[np.integer]],
    embeddings: npt.NDArray[np.floating],
    num_trajectories: int,
    graph: ig.Graph,
    min_traj_length: int,
    max_traj_length: int,
    max_env_steps: int,
) -> tuple[list[npt.NDArray[np.integer]], list[npt.NDArray[np.floating]], list[int]]:
    sampled_trajectories_id = []
    sampled_trajectories_emb = []
    starting_locations = []

    for _ in range(num_trajectories):
        trajectory = select_random_trajectory(trajectories, graph, min_traj_length, max_traj_length)
        starting_locations.append(trajectory[0])

        padded_trajectory = np.pad(
            trajectory, (0, max_env_steps - len(trajectory)), constant_values=len(embeddings) - 1
        )
        sampled_trajectories_id.append(padded_trajectory)
        sampled_trajectories_emb.append(embeddings[padded_trajectory])

    return sampled_trajectories_id, sampled_trajectories_emb, starting_locations


def get_trajectories_from_random_location(
    trajectories: list[npt.NDArray[np.integer]],
) -> list[npt.NDArray[np.integer]]:
    n = random.randint(0, 100)
    counter = Counter(t[0] for t in trajectories)
    traj_id, _ = counter.most_common()[n]
    trajs = [t for t in trajectories if t[0] == traj_id]
    return trajs


def sample_n_semantically_similar_trajectories(
    trajectories: list[npt.NDArray[np.integer]],
    num_trajectories: int,
    graph: ig.Graph,
    min_traj_length: int,
    max_traj_length: int,
    max_env_steps: int,
    exit_id: int,
    min_similarity: float = 0.90,
    remove_consecutive_duplicates: bool = False,
    do_padding: bool = False,
) -> tuple[list[npt.NDArray[np.integer]], list[npt.NDArray[np.floating]], list[int]]:
    """
    Sample a set of semantically similar trajectories from a larger collection.

    This function filters valid trajectories, computes their embeddings, selects
    a random trajectory, and then samples additional trajectories that are
    semantically similar to the chosen one based on cosine similarity.

    Args:
        trajectories: List of trajectory arrays, each containing node IDs
        num_trajectories: Number of similar trajectories to sample
        graph: igraph Graph object containing node embeddings
        min_traj_length: Minimum valid trajectory length
        max_traj_length: Maximum valid trajectory length
        max_env_steps: Maximum environment steps (used for padding)
        exit_id: ID to use for padding trajectories
        min_similarity: Minimum cosine similarity threshold (default: 0.90)

    Returns:
        tuple containing:
            - List of sampled and padded trajectory arrays
            - Array of embeddings for the sampled trajectories
            - List of starting locations for each sampled trajectory
    """
    trajectories, trajectories_embs = _prepare_trajectory_embeddings(
        trajectories, graph, remove_consecutive_duplicates, min_traj_length, max_traj_length
    )

    # Select a random trajectory and find semantically similar ones
    candidates = np.array([])
    rng = np.random.default_rng()
    while len(candidates) < num_trajectories:
        random_traj = rng.choice(trajectories_embs)
        similarities = cosine_similarity(random_traj.reshape(1, -1), trajectories_embs).flatten()
        candidates = np.where(similarities > min_similarity)[0]

    sample = rng.choice(candidates, size=num_trajectories, replace=False)
    sample_itg = operator.itemgetter(*sample)
    sampled_trajectories_id = sample_itg(trajectories)

    starting_locations = [t[0] for t in sampled_trajectories_id]
    if do_padding:
        sampled_trajectories_id = [
            np.pad(t, (0, max_env_steps - len(t)), constant_values=exit_id) for t in sampled_trajectories_id
        ]
    sampled_trajectories_emb = [np.array(graph.vs[s]["embedding"]) for s in sampled_trajectories_id]

    return sampled_trajectories_id, sampled_trajectories_emb, starting_locations
