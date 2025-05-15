import pickle
import random
import typing
import operator
from collections import Counter

import duckdb
import igraph as ig
import numpy as np
import numpy.typing as npt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from evaluating_trajectories.utils import consts


def load_environment(
    graph_filepath: str,
    trajectories_filepath: str,
) -> tuple[ig.Graph, list[npt.NDArray[np.integer]]]:
    with open(graph_filepath, "rb") as f:
        graph = pickle.load(f)

    trajectories = duckdb.sql(f"SELECT trajectory FROM '{trajectories_filepath}'").fetchnumpy()["trajectory"]

    return graph, trajectories


def is_trajectory_valid(trajectory: npt.NDArray[np.integer], graph: ig.Graph, min_length: int, max_length: int) -> bool:
    if len(trajectory) < min_length or len(trajectory) >= max_length:
        return False
    return np.all(trajectory < len(graph.vs))


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
    trajectories = [
        t
        for t in tqdm(trajectories, desc="Random sampling: filtering out invalid trajectories")
        if t is not None and is_trajectory_valid(t, graph, min_traj_length, max_traj_length)
    ]

    # Extract mean embeddings for each trajectory
    traj_embs = [
        np.array(graph.vs[t]["embedding"]).mean(axis=0)
        for t in tqdm(trajectories, desc="Random sampling: extracting trajectory embeddings")
    ]

    # Filter out trajectories with NaN embeddings
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

    # Select a random trajectory and find semantically similar ones
    rng = np.random.default_rng()
    random_traj = rng.choice(filtered_trajs_emb)
    similarities = cosine_similarity(random_traj.reshape(1, -1), filtered_trajs_emb).flatten()
    candidates = np.where(similarities > min_similarity)[0]
    sample = rng.choice(candidates, size=num_trajectories)

    # Prepare the final output: sampled trajectories, embeddings, and starting locations
    sample_itg = operator.itemgetter(*sample)
    sampled_trajectories_id = sample_itg(filtered_trajs_id)
    starting_locations = [t[0] for t in sampled_trajectories_id]
    sampled_trajectories_id = [
        np.pad(t, (0, max_env_steps - len(t)), constant_values=exit_id) for t in sampled_trajectories_id
    ]
    sampled_trajectories_emb = [np.array(graph.vs[s]["embedding"]) for s in sampled_trajectories_id]

    return sampled_trajectories_id, sampled_trajectories_emb, starting_locations
