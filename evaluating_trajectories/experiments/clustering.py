import pickle

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

from evaluating_trajectories.environment.website_env import WebsiteEnvironment
from evaluating_trajectories.evaluation.levenshtein_distance import (
    cos_dist,
    levenshtein_distance,
)

embs_file = "evaluating_trajectories/dataset/embs.npy"
trajectories_file = "evaluating_trajectories/dataset/trajectories.pkl"
graph_file = "evaluating_trajectories/dataset/graph.pkl"
max_steps = 16

min_trajectory_length = 3
max_trajectory_length = 16

embs = np.load(embs_file)

with open(trajectories_file, "rb") as f:
    trajectories = pickle.load(f)

with open(graph_file, "rb") as f:
    graph = pickle.load(f)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
mask_embedding = model.encode(model.tokenizer.special_tokens_map["mask_token"])  # type: ignore

env = WebsiteEnvironment(graph, [0], max_trajectory_length, embs, mask_embedding)  # type: ignore


def dist(a: npt.NDArray, b: npt.NDArray):
    a = a.reshape((max_trajectory_length, -1))
    b = b.reshape((max_trajectory_length, -1))
    return levenshtein_distance([arr for arr in a], [arr for arr in b], cos_dist)[0]  # type: ignore


def trajectory_valid(trajectory) -> bool:
    if (
        len(trajectory) < min_trajectory_length
        or len(trajectory) >= max_trajectory_length
    ):
        return False
    for node_id in trajectory:
        if node_id >= len(env.graph.vs):
            return False
    return True


converted_trajectories = []
for traj in trajectories[:100]:
    if not trajectory_valid(traj):
        continue
    padded_trajectory = np.full(max_steps, len(env.embeddings) - 1, dtype=np.int32)
    for i, node_id in enumerate(traj):
        location = np.array(node_id)
        done = i == len(traj) - 1
        padded_trajectory[i] = location
        if done:
            padded_trajectory[i + 1] = env.exit_action
            obs = env.embeddings[padded_trajectory]
            converted_trajectories.append(obs.copy())

trajectories = np.stack(converted_trajectories)
print(trajectories.shape)
trajectories = trajectories.reshape((trajectories.shape[0], -1))
clustering = DBSCAN(eps=3, min_samples=2, metric=dist).fit(embs)  # type: ignore

print(clustering.labels_)
