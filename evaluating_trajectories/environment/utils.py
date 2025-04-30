import pickle
import random

import igraph as ig
import numpy as np
import numpy.typing as npt

from evaluating_trajectories.utils import consts


def load_environment(
    graph_filepath: str,
    trajectories_filepath: str,
) -> tuple[ig.Graph, list[npt.NDArray[np.integer]]]:
    with open(graph_filepath, "rb") as f:
        graph = pickle.load(f)
    with open(trajectories_filepath, "rb") as f:
        trajectories = pickle.load(f)

    return graph, trajectories


def is_trajectory_valid(trajectory: npt.NDArray[np.integer], graph: ig.Graph, min_length: int, max_length: int) -> bool:
    if len(trajectory) < min_length or trajectory >= max_length:
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
) -> tuple[list[npt.NDArray[np.floating]], list[int]]:
    sampled_trajectories = []
    starting_locations = []

    for _ in range(num_trajectories):
        trajectory = select_random_trajectory(trajectories, graph, min_traj_length, max_traj_length)
        starting_locations.append(trajectory[0])

        padded_trajectory = np.pad(
            trajectory, (0, max_env_steps - len(trajectory)), constant_values=len(embeddings) - 1
        )
        sampled_trajectories.append(embeddings[padded_trajectory])

    return sampled_trajectories, starting_locations
