import argparse
import os
import pickle

import numpy as np
from sb3_contrib import MaskablePPO
from tqdm import tqdm
from dataclasses import dataclass
import dotenv

from evaluating_trajectories import environment
from evaluating_trajectories.dataset import preprocessing
from evaluating_trajectories.environment.rewards import LevenshteinReward
from evaluating_trajectories.environment.website_env import WebsiteEnvironment
from evaluating_trajectories.utils import utils

dotenv.load_dotenv()


@dataclass
class TrainingConfig:
    device: str
    experiment_name: str
    n_steps: int
    gamma: float
    num_trajectories: int
    min_trajectory_length: int
    max_trajectory_length: int
    eval_frequency: int
    eval_episodes: int
    train_intervals: int
    embs_file: str
    graph_file: str
    trajectories_file: str
    glove_vocab_file: str
    glove_vectors_file: str


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train PPO agent on website environment")

    parser.add_argument(
        "--device", type=str, default=os.getenv("DEVICE", "cuda:0"), help="Device to run on (e.g. cuda:0)"
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=os.getenv("EXPERIMENT_NAME", "ppo_levenshtein_min"),
        help="Name of the experiment",
    )

    parser.add_argument(
        "--n-steps", type=int, default=int(os.getenv("N_STEPS", "1024")), help="Number of steps per PPO update"
    )

    parser.add_argument("--gamma", type=float, default=float(os.getenv("GAMMA", "0.9")), help="Discount factor")

    parser.add_argument(
        "--num-trajectories", type=int, default=int(os.getenv("NUM_TRAJECTORIES", "10")), help="Number of trajectories"
    )

    parser.add_argument(
        "--min-trajectory-length",
        type=int,
        default=int(os.getenv("MIN_TRAJECTORY_LENGTH", "3")),
        help="Minimum trajectory length",
    )

    parser.add_argument(
        "--max-trajectory-length",
        type=int,
        default=int(os.getenv("MAX_TRAJECTORY_LENGTH", "16")),
        help="Maximum trajectory length",
    )

    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=int(os.getenv("EVAL_FREQUENCY", "4096")),
        help="Evaluation frequency in steps",
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=int(os.getenv("EVAL_EPISODES", "100")),
        help="Number of evaluation episodes",
    )

    parser.add_argument(
        "--train-intervals",
        type=int,
        default=int(os.getenv("TRAIN_INTERVALS", "1000")),
        help="How often training is run",
    )

    parser.add_argument(
        "--embs-file",
        type=str,
        default=os.getenv("EMBS_FILE", "evaluating_trajectories/dataset/embs.npy"),
        help="Path to embeddings file",
    )

    parser.add_argument(
        "--graph-file",
        type=str,
        default=os.getenv("GRAPH_FILE", "evaluating_trajectories/dataset/graph.pkl"),
        help="Path to graph file",
    )

    parser.add_argument(
        "--trajectories-file",
        type=str,
        default=os.getenv("TRAJECTORIES_FILE", "evaluating_trajectories/dataset/trajectories.pkl"),
        help="Path to trajectories file",
    )

    parser.add_argument(
        "--glove-vocab-file",
        type=str,
        default=os.getenv("GLOVE_VOCAB_FILE", "data/glove_emb_out_2019-10-01_2019-11-01/vocab.txt"),
        help="Path to GloVe vocab file",
    )

    parser.add_argument(
        "--glove-vectors-file",
        type=str,
        default=os.getenv("GLOVE_VECTORS_FILE", "data/glove_emb_out_2019-10-01_2019-11-01/vectors.txt"),
        help="Path to GloVe vectors file",
    )

    args = parser.parse_args()

    return TrainingConfig(
        device=args.device,
        experiment_name=args.experiment_name,
        n_steps=args.n_steps,
        gamma=args.gamma,
        num_trajectories=args.num_trajectories,
        min_trajectory_length=args.min_trajectory_length,
        max_trajectory_length=args.max_trajectory_length,
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes,
        train_intervals=args.train_intervals,
        embs_file=args.embs_file,
        graph_file=args.graph_file,
        trajectories_file=args.trajectories_file,
        glove_vocab_file=args.glove_vocab_file,
        glove_vectors_file=args.glove_vectors_file,
    )


def print_config_summary(args: TrainingConfig):
    print("\n=== Configuration Summary ===")
    print(f"{'Parameter':<25} {'Value'}")
    print("=" * 40)

    # Get all attributes of args that don't start with '_'
    for arg_name, value in sorted(vars(args).items()):
        if not arg_name.startswith("_"):
            # Replace underscores with spaces and capitalize for better readability
            pretty_name = arg_name.replace("_", " ").title()
            print(f"{pretty_name:<25} {value}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    config = parse_args()
    print_config_summary(config)

    output_folder, _ = utils.create_experiment_hash(config.n_steps, config.gamma, ".experiments")

    print("Loading embeddings...")
    embs_with_vocab = preprocessing.load_glove_embeddings(config.glove_vocab_file, config.glove_vectors_file)

    print("Loading environment...")
    graph, trajectories = environment.utils.load_environment(
        graph_filepath=config.graph_file, trajectories_filepath=config.trajectories_file
    )

    # map original trajectories to graph node ids
    trajectories = [
        np.array(list(map(embs_with_vocab.vocab.get, map(str, t))))
        for t in tqdm(trajectories, desc="Mapping trajectories")
    ]

    # Filter the graph so that we obtain the subgraph of only those products for which we have embeddings.
    print("Loading subgraph...")
    available_nodes = set(embs_with_vocab.vocab.keys())
    graph = graph.subgraph(graph.vs.select(name_in=available_nodes))

    for vert in tqdm(graph.vs, desc="Loading embeddings onto graph"):
        idx = embs_with_vocab.vocab[vert["name"]]
        vert["embedding"] = embs_with_vocab.embeddings_norm[idx]

    graph.add_vertex(name="exit", embedding=embs_with_vocab.mask_embedding)
    embeddings = np.array(graph.vs["embedding"])

    with open(f"{output_folder}/graph_with_embeddings.pkl", "wb") as f:
        pickle.dump(graph, f)

    print("Preparing environment...")

    # We select a random starting location and we take those as the group trajectories.
    # We want to avoid clustering trajectories with Levenshtein if then we're training with Levenshtein as the reward function
    trajectories_from_location = environment.utils.get_trajectories_from_random_location(trajectories)
    group_trajectories_id, group_trajectories_emb, starting_locations = environment.utils.sample_n_trajectories(
        trajectories_from_location,
        embeddings,
        config.num_trajectories,
        graph,
        config.min_trajectory_length,
        config.max_trajectory_length,
        config.max_trajectory_length,
    )
    env = WebsiteEnvironment(
        graph,
        starting_locations,
        config.max_trajectory_length,
        embeddings.min(),
        embeddings.max(),
        reward=LevenshteinReward(group_trajectories_emb),
    )
    eval_env = WebsiteEnvironment(
        graph,
        starting_locations,
        config.max_trajectory_length,
        embeddings.min(),
        embeddings.max(),
        reward=LevenshteinReward(group_trajectories_emb),
    )

    with open(f"{output_folder}/trajectories_target.pkl", "wb") as f:
        pickle.dump({"trajectory_id": group_trajectories_id, "trajectory_emb": group_trajectories_emb}, f)

    print("Training...")

    # if you run into a problem with the logits during training, set `validate_args=False` in sb3_contrib/common/maskable/distributions.py:68, see also https://github.com/DLR-RM/stable-baselines3/issues/1596
    # unfortunately this is an issue with sb3 (or rather torch) and not with our code
    model = MaskablePPO("MlpPolicy", env, gamma=config.gamma, verbose=1, device=config.device, n_steps=config.n_steps)
    i = 0  # avoids type checker complaints
    for i in range(config.train_intervals):
        print(f"evaluation round {i}")
        trajectories = []
        for _ in range(config.eval_episodes):
            obs, info = eval_env.reset()
            while True:
                action = model.predict(obs)[0]
                obs, reward, terminated, truncated, info = eval_env.step(int(action))  # type: ignore
                if terminated or truncated:
                    trajectories.append(eval_env.trajectory)
                    break

        with open(f"{output_folder}/trajectories_{i * config.eval_frequency}.pkl", "wb") as f:
            pickle.dump(trajectories, f)

        print(f"training round {i}")
        model.learn(config.eval_frequency, reset_num_timesteps=(i == 0))

    print(f"evaluation round {i + 1}")
    trajectories = []
    for _ in range(config.eval_episodes):
        obs, info = eval_env.reset()
        while True:
            action = model.predict(obs)[0]
            obs, reward, terminated, truncated, info = eval_env.step(action)  # type: ignore
            if terminated or truncated:
                trajectories.append(obs)
                break

    with open(
        f"{output_folder}/trajectories_{(i + 1) * config.eval_frequency}.pkl",
        "wb",
    ) as f:
        pickle.dump(trajectories, f)
