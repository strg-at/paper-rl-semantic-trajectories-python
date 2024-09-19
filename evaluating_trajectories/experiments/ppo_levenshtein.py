import pickle
import sys
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sentence_transformers import SentenceTransformer

from evaluating_trajectories.environment.website_env import (
    RewardClass,
    WebsiteEnvironment,
)
from evaluating_trajectories.evaluation.levenshtein_distance import (
    cos_dist,
    levenshtein_distance,
)

#####################
#     Variables     #
#####################

device = f"cuda:{sys.argv[1]}"
print(f"running on {device}")
experiment_name = "ppo_levenshtein"

num_trajectories = 1
override_traj_idxs = [
    3540384,
]  # if contains num_trajectories entries, use these trajectories
min_trajectory_length = 3
max_trajectory_length = 16

eval_frequency = 100  # in steps
eval_episodes = 100
total_train_steps = 100_000

embs_file = "evaluating_trajectories/dataset/embs.npy"
graph_file = "evaluating_trajectories/dataset/graph.pkl"
trajectories_file = "evaluating_trajectories/dataset/trajectories.pkl"

####################################################
#     Create Output Folder from Experiment Hash    #
####################################################

output_folder = f"experiments/{experiment_name}/run_{time.time()}"
# make sure path exists
Path(output_folder).mkdir(parents=True, exist_ok=True)

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
    if (
        len(trajectories[traj_idx]) < min_trajectory_length
        or len(trajectories[traj_idx]) >= max_trajectory_length
    ):
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

##########################
#    Construct Reward    #
##########################


assert len(traj_idxs) == 1, "Reward right now only works with one trajectory..."


class LevenshteinReward(RewardClass):
    def compute_reward(self, trajectory: npt.NDArray[np.float32]) -> float:
        trajectory = [obs for obs in trajectory]  # type:ignore
        return 1 - np.abs(levenshtein_distance(group_trajectories[0], trajectory, cos_dist)[0])  # type: ignore


env.reward = LevenshteinReward()

#########################
#    Train PPO agent    #
#########################

model = MaskablePPO("MlpPolicy", env, gamma=0.9, verbose=1, device=device)
model.learn(50_000)

#################
#    Run PPO    #
#################

obs, _ = env.reset()
i = 0
while True:
    # Retrieve current action mask
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    print(action)
    obs, reward, terminated, truncated, info = env.step(action)  # type:ignore
    if terminated or truncated:
        obs, _ = env.reset()
        i += 1
        if i == 10:
            break
        print("")
        print("next")
