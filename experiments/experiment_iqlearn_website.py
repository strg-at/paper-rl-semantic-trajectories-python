import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from evaluating_trajectories.environment.website_env import WebsiteEnvironment
from evaluating_trajectories.iqlearn.buffer import MaskableReplayBuffer
from evaluating_trajectories.iqlearn.iqlearn import IQLearn

#####################
#     Variables     #
#####################

device = f"cuda:{sys.argv[1]}"
print(f"running on {device}")
experiment_name = "single_traj"

alpha = float(sys.argv[2])
q_lr = 5e-5
print(f"using alpha={alpha}, q_lr={q_lr}")

num_trajectories = 1
override_traj_idxs = [
    3540384,
]  # if contains num_trajectories entries, use these trajectories
min_trajectory_length = 3
max_trajectory_length = 16

eval_frequency = 100  # in steps
eval_episodes = 100
train_intervals = 100  # how often training is run, total steps will be eval_frequency * train_intervals

embs_file = "evaluating_trajectories/dataset/embs.npy"
graph_file = "evaluating_trajectories/dataset/graph.pkl"
trajectories_file = "evaluating_trajectories/dataset/trajectories.pkl"

####################################################
#     Create Output Folder from Experiment Hash    #
####################################################

important_variables = {"alpha": alpha, "traj_idx": override_traj_idxs, "q_lr": q_lr}
json_string = json.dumps(important_variables, sort_keys=True)
hyperparameter_hash = hashlib.md5(json_string.encode("utf-8")).hexdigest()[:8]
print(f"experiment hash: {hyperparameter_hash}")
output_folder = f"experiments/{experiment_name}/trajectories_{hyperparameter_hash}/run_{time.time()}"
# make sure path exists
Path(output_folder).mkdir(parents=True, exist_ok=True)

with open(
    f"experiments/{experiment_name}/trajectories_{hyperparameter_hash}/parameters.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(important_variables, f, ensure_ascii=False, indent=4)


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
    if len(trajectories[traj_idx]) < min_trajectory_length or len(trajectories[traj_idx]) >= max_trajectory_length:
        return False
    for node_id in trajectory:
        if node_id >= len(env.graph.vs):
            return False
    return True


buffer = MaskableReplayBuffer(100, env.observation_space, env.action_space, device=device)  # type: ignore
starting_locations = []

group_trajectories = []
traj_idxs = []
for j in range(num_trajectories):
    if len(override_traj_idxs) != num_trajectories:
        traj_idx = np.random.randint(len(trajectories))
        while not trajectory_valid(trajectories[traj_idx]):
            traj_idx = np.random.randint(len(trajectories))
    else:
        traj_idx = override_traj_idxs[j]
    traj_idxs.append(traj_idx)
    obs = None
    action_mask = None
    padded_trajectory = np.full(env.max_steps, len(env.embeddings) - 1, dtype=np.int32)
    print(trajectories[traj_idx])
    for i, node_id in enumerate(trajectories[traj_idx]):
        if i == 0:
            starting_locations.append(node_id)
        location = np.array(node_id)
        next_action_mask = valid_actions(location)
        done = i == len(trajectories[traj_idx]) - 1
        padded_trajectory[i] = location
        next_obs = padded_trajectory.copy()
        next_obs = env.embeddings[padded_trajectory]
        if obs is not None:
            buffer.add(obs, next_obs, location, 0, done, [{}], action_mask)  # type: ignore
        obs = next_obs
        action_mask = next_action_mask
        if done:
            padded_trajectory[i + 1] = env.exit_action
            next_obs = padded_trajectory.copy()
            next_obs = env.embeddings[padded_trajectory]
            buffer.add(obs, next_obs, np.array(env.exit_action), 0, True, [{}], action_mask)  # type: ignore
            group_trajectories.append(next_obs.copy())
        # action_mask = next_action_mask  # type: ignore

print(f"trajectory indices: {traj_idxs}")
with open(f"{output_folder}/trajectories_expert.pkl", "wb") as f:
    pickle.dump(group_trajectories, f)
env.starting_location = starting_locations

##############################
#     train with IQLEARN     #
##############################
iqlearn = IQLearn(env, sac_args={"device": device, "alpha": {0: alpha}, "q_lr": q_lr})
iqlearn.set_demonstration_buffer(buffer)

for i in range(train_intervals):
    print(f"evaluation round {i}")
    trajectories = []
    for _ in range(eval_episodes):
        obs, info = env.reset()
        while True:
            action = iqlearn.predict(obs)[0]
            obs, reward, terminated, truncated, info = env.step(action)  # type: ignore
            if terminated or truncated:
                trajectories.append(obs)
                break

    with open(f"{output_folder}/trajectories_{i * eval_frequency}.pkl", "wb") as f:
        pickle.dump(trajectories, f)

    print(f"training round {i}")
    iqlearn.learn(eval_frequency)

print(f"evaluation round {i + 1}")  # type:ignore
trajectories = []
for _ in range(eval_episodes):
    obs, info = env.reset()
    while True:
        action = iqlearn.predict(obs)[0]
        obs, reward, terminated, truncated, info = env.step(action)  # type: ignore
        if terminated or truncated:
            trajectories.append(obs)
            break

with open(f"{output_folder}/trajectories_{(i + 1) * eval_frequency}.pkl", "wb") as f:  # type:ignore
    pickle.dump(trajectories, f)
