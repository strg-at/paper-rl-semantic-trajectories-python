import pickle
import sys

import numpy as np
from sb3_contrib import MaskablePPO
from tqdm import tqdm

from evaluating_trajectories import environment
from evaluating_trajectories.dataset import preprocessing
from evaluating_trajectories.environment.rewards import LevenshteinReward
from evaluating_trajectories.environment.website_env import WebsiteEnvironment
from evaluating_trajectories.utils import utils

#####################
#     Variables     #
#####################

device = f"cuda:{sys.argv[1]}"
print(f"running on {device}")
experiment_name = "ppo_levenshtein_min"

n_steps = 1024
gamma = 0.9

num_trajectories = 1
min_trajectory_length = 3
max_trajectory_length = 16

eval_frequency = 4096  # in steps
eval_episodes = 100
train_intervals = 1000  # how often training is run, total steps will be eval_frequency * train_intervals

embs_file = "evaluating_trajectories/dataset/embs.npy"
graph_file = "evaluating_trajectories/dataset/graph.pkl"
trajectories_file = "evaluating_trajectories/dataset/trajectories.pkl"

glove_vocab_file = "data/glove_emb_out_2019-10-01_2019-11-01/vocab.txt"
glove_vectors_file = "data/glove_emb_out_2019-10-01_2019-11-01/vectors.txt"

output_folder, _ = utils.create_experiment_hash(n_steps, gamma, ".experiments")


############################
#     Load Environment     #
############################


embs_with_vocab = preprocessing.load_glove_embeddings(glove_vocab_file, glove_vectors_file)

graph, trajectories = environment.utils.load_environment(
    graph_filepath=graph_file, trajectories_filepath=trajectories_file
)
# Filter the graph so that we obtain the subgraph of only those products for which we have embeddings.
print("Loading subgraph...")
available_nodes = set(embs_with_vocab.vocab.keys())
graph = graph.subgraph(graph.vs.select(name_in=available_nodes))

for vert in tqdm(graph.vs, desc="Loading embeddings onto graph"):
    idx = embs_with_vocab.vocab[vert["name"]]
    vert["embedding"] = embs_with_vocab.embeddings_norm[idx]

graph.add_vertex(name="exit", embedding=embs_with_vocab.mask_embedding)
embeddings = np.array(graph.vs["embedding"])


#################################################
#    Get Starting Locations for Trajectories    #
#################################################

group_trajectories, starting_locations = environment.utils.sample_n_trajectories(
    trajectories,
    embeddings,
    num_trajectories,
    graph,
    min_trajectory_length,
    max_trajectory_length,
    max_trajectory_length,
)
group_trajectories = [[obs for obs in trajectory] for trajectory in group_trajectories]
env = WebsiteEnvironment(
    graph, [0], max_trajectory_length, embeddings.min(), embeddings.max(), reward=LevenshteinReward(group_trajectories)
)
eval_env = WebsiteEnvironment(
    graph,
    starting_locations,
    max_trajectory_length,
    embeddings.min(),
    embeddings.max(),
    reward=LevenshteinReward(group_trajectories),
)

with open(f"{output_folder}/trajectories_expert.pkl", "wb") as f:
    pickle.dump(group_trajectories, f)


#################
#    Run PPO    #
#################

# if you run into a problem with the logits during training, set `validate_args=False` in sb3_contrib/common/maskable/distributions.py:68, see also https://github.com/DLR-RM/stable-baselines3/issues/1596
# unfortunately this is an issue with sb3 (or rather torch) and not with our code

model = MaskablePPO("MlpPolicy", env, gamma=gamma, verbose=1, device=device, n_steps=n_steps)
i = 0  # avoids type checker complaints
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

print(f"evaluation round {i + 1}")
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
    "wb",
) as f:
    pickle.dump(trajectories, f)
