import pickle

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from evaluating_trajectories.environment.website_env import WebsiteEnvironment
from evaluating_trajectories.iqlearn.buffer import MaskableReplayBuffer
from evaluating_trajectories.iqlearn.iqlearn import IQLearn

#####################
#     Variables     #
#####################

device = "cuda"
train_steps = 5000
min_trajectory_length = 3
max_trajectory_length = 10

embs_file = "evaluating_trajectories/dataset/embs.npy"
graph_file = "evaluating_trajectories/dataset/graph.pkl"
trajectories_file = "evaluating_trajectories/dataset/trajectories.pkl"

############################
#     Load Environment     #
############################


embs = np.load(embs_file)

with open(graph_file, "rb") as f:
    graph = pickle.load(f)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
mask_embedding = model.encode(model.tokenizer.special_tokens_map["mask_token"])  # type: ignore

env = WebsiteEnvironment(graph, [0], max_trajectory_length, embs, mask_embedding)  # type: ignore

with open(trajectories_file, "rb") as f:
    trajectories = pickle.load(f)


####################################################
#     convert trajectory to IQLearn compatible     #
####################################################


def valid_actions(location) -> np.ndarray:
    mask = np.zeros(env.action_space.n, dtype=bool)  # type: ignore
    valid_actions = [env.exit_action] + env.graph.neighbors(location)
    mask[valid_actions] = True
    return mask


def trajectory_valid(trajectory) -> bool:
    if (
        len(trajectories[traj_idx]) < min_trajectory_length
        or len(trajectories[traj_idx]) >= max_trajectory_length
    ):
        return False
    for node_id in trajectory:
        if node_id >= len(env.graph.vs):
            return False
    return True


num_trajectories = 10
buffer = MaskableReplayBuffer(100, env.observation_space, env.action_space, device=device)  # type: ignore
starting_locations = []

print(len(env.graph.vs))
for _ in range(num_trajectories):
    traj_idx = np.random.randint(len(trajectories))
    while not trajectory_valid(trajectories[traj_idx]):
        traj_idx = np.random.randint(len(trajectories))
    print(f"imitating trajectory {traj_idx}")
    obs = None
    action_mask = None
    padded_trajectory = np.full(env.max_steps, len(env.embeddings) - 1, dtype=np.int32)
    for i, node_id in enumerate(trajectories[traj_idx]):
        if i == 0:
            starting_locations.append(node_id)
        location = np.array(node_id)
        print(location)
        action_mask = valid_actions(location)
        done = i == len(trajectories[traj_idx]) - 1
        if not done:
            padded_trajectory[i] = location
            next_obs = padded_trajectory.copy()
            next_obs = env.embeddings[padded_trajectory]
        else:
            next_obs = padded_trajectory.copy()
            next_obs = env.embeddings[padded_trajectory]
        if obs is not None:
            buffer.add(obs, next_obs, location, 0, False, [{}], action_mask)  # type: ignore
        if done:
            break
        obs = next_obs
        # action_mask = next_action_mask  # type: ignore

env.starting_location = starting_locations

##############################
#     train with IQLEARN     #
##############################
iqlearn = IQLearn(env, sac_args={"device": device})
iqlearn.set_demonstration_buffer(buffer)
iqlearn.learn(train_steps)

# observations = []
# possible_actions = []
# obs, info = env.reset()
# action_embeddings = env.map_action_ids_to_embeddings(env.valid_actions())
# action_embeddings = torch.tensor(action_embeddings)
# obs = torch.tensor(obs)
# observations.append(obs)
# possible_actions.append(action_embeddings)
# for i in range(5):
#     obs, _, _, _, _ = env.step(np.random.choice(env.valid_actions()[1:]))
#     obs = torch.tensor(obs)
#     action_embeddings = env.map_action_ids_to_embeddings(env.valid_actions())
#     action_embeddings = torch.tensor(action_embeddings)
#     observations.append(obs)
#     possible_actions.append(action_embeddings)
#
# print(f"{iqlearn.compute_values(observations, possible_actions)=}")

for _ in range(100):
    obs, info = env.reset()
    print(env.agent_location)
    while True:
        action = iqlearn.predict(obs, deterministic=True)
        print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("")
            break
