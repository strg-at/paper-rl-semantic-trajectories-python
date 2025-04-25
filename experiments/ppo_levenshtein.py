import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sentence_transformers import SentenceTransformer

from evaluating_trajectories.environment.website_env import WebsiteEnvironment
from evaluating_trajectories.environment.rewards import LevenshteinReward

#####################
#     Variables     #
#####################

device = f"cuda:{sys.argv[1]}"
print(f"running on {device}")
experiment_name = "ppo_levenshtein_min"

n_steps = 1024
gamma = 0.9

num_trajectories = 1
override_traj_idxs = [
    1829803,
    5255935,
    3619652,
    5034976,
    794293,
    2561893,
    3680468,
    432399,
    2834087,
    394507,
]
override_traj_idxs = [
    3540384,
]  # if contains num_trajectories entries, use these trajectories
min_trajectory_length = 3
max_trajectory_length = 16

eval_frequency = 4096  # in steps
eval_episodes = 100
train_intervals = 1000  # how often training is run, total steps will be eval_frequency * train_intervals

embs_file = "evaluating_trajectories/dataset/embs.npy"
graph_file = "evaluating_trajectories/dataset/graph.pkl"
trajectories_file = "evaluating_trajectories/dataset/trajectories.pkl"

####################################################
#     Create Output Folder from Experiment Hash    #
####################################################

important_variables = {"n_steps": n_steps, "gamma": gamma}
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


#################################################
#    Get Starting Locations for Trajectories    #
#################################################


def trajectory_valid(trajectory) -> bool:
    if len(trajectories[traj_idx]) < min_trajectory_length or len(trajectories[traj_idx]) >= max_trajectory_length:
        return False
    for node_id in trajectory:
        if node_id >= len(env.graph.vs):
            return False
    return True


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
    starting_locations.append(trajectories[traj_idx][0])
    padded_trajectory = np.full(env.max_steps, len(env.embeddings) - 1, dtype=np.int32)
    for i, node_id in enumerate(trajectories[traj_idx]):
        location = np.array(node_id)
        print(location)
        done = i == len(trajectories[traj_idx]) - 1
        padded_trajectory[i] = location
        if done:
            padded_trajectory[i + 1] = env.exit_action
            obs = env.embeddings[padded_trajectory]
            group_trajectories.append(obs.copy())

print(f"trajectory indices: {traj_idxs}")
with open(f"{output_folder}/trajectories_expert.pkl", "wb") as f:
    pickle.dump(group_trajectories, f)
env.starting_location = starting_locations

##########################################################
#    Convert group_trajectories to levenshtein format    #
##########################################################

group_trajectories = [[obs for obs in trajectory] for trajectory in group_trajectories]


env.reward = LevenshteinReward(group_trajectories)
eval_env = WebsiteEnvironment(
    graph, starting_locations, max_trajectory_length, embs, mask_embedding, reward=LevenshteinReward(group_trajectories)
)  # type: ignore

#################
#    Run PPO    #
#################

# if you run into a problem with the logits during training, set `validate_args=False` in sb3_contrib/common/maskable/distributions.py:68, see also https://github.com/DLR-RM/stable-baselines3/issues/1596
# unfortunately this is an issue with sb3 (or rather torch) and not with our code

model = MaskablePPO("MlpPolicy", env, gamma=gamma, verbose=1, device=device, n_steps=n_steps)
for i in range(train_intervals):
    print(f"evaluation round {i}")
    trajectories = []
    for _ in range(eval_episodes):
        obs, info = eval_env.reset()
        while True:
            action = model.predict(obs)[0]
            obs, reward, terminated, truncated, info = eval_env.step(action)  # type: ignore
            if terminated or truncated:
                trajectories.append(obs)
                break

    with open(f"{output_folder}/trajectories_{i * eval_frequency}.pkl", "wb") as f:
        pickle.dump(trajectories, f)

    print(f"training round {i}")
    model.learn(eval_frequency, reset_num_timesteps=(i == 0))

print(f"evaluation round {i + 1}")  # type:ignore
trajectories = []
for _ in range(eval_episodes):
    obs, info = eval_env.reset()
    while True:
        action = model.predict(obs)[0]
        obs, reward, terminated, truncated, info = eval_env.step(action)  # type: ignore
        if terminated or truncated:
            trajectories.append(obs)
            break

with open(
    f"{output_folder}/trajectories_{(i + 1) * eval_frequency}.pkl",
    "wb",  # type:ignore
) as f:  # type:ignore
    pickle.dump(trajectories, f)
