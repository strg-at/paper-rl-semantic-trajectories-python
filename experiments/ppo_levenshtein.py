import argparse
import os
import pickle

import numpy as np
import numpy.typing as npt
import igraph as ig
from sb3_contrib import MaskablePPO
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from tqdm import tqdm
from dataclasses import dataclass
from torch.distributions import Distribution
import dotenv

from evaluating_trajectories.environment import trajectory_sampling
from evaluating_trajectories.dataset import preprocessing
from evaluating_trajectories.environment.rewards import LevenshteinReward
from evaluating_trajectories.distances.levenshtein_distance import exact_comparison_distance
from evaluating_trajectories.environment.website_env import WebsiteEnvironment
from evaluating_trajectories.utils import utils
from experiments.save_trajectory_callback import SaveTrajectoryCallback

dotenv.load_dotenv()


@dataclass
class TrainingConfig:
    device: str
    experiment_name: str
    n_steps: int
    num_experiments: int
    gamma: float
    num_trajectories: int
    min_trajectory_length: int
    max_trajectory_length: int
    eval_frequency: int
    eval_episodes: int
    train_timesteps: int
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

    parser.add_argument(
        "--num-experiments", type=int, default=int(os.getenv("NUM_EXPERIMENTS", "10")), help="Number of experiments"
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
        "--train-timesteps",
        type=int,
        default=int(os.getenv("TRAIN_TIMESTEPS", "160000")),
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
        num_experiments=args.num_experiments,
        gamma=args.gamma,
        num_trajectories=args.num_trajectories,
        min_trajectory_length=args.min_trajectory_length,
        max_trajectory_length=args.max_trajectory_length,
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes,
        train_timesteps=args.train_timesteps,
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


def do_train_and_eval(
    model: MaskablePPO,
    eval_env: WebsiteEnvironment,
    config: TrainingConfig,
    output_folder: str,
    model_name: str,
    callback: MaybeCallback = None,
):
    model.learn(config.train_timesteps, callback=callback)

    trajectories = []
    for _ in range(config.eval_episodes):
        obs, _ = eval_env.reset()
        while True:
            action = model.predict(obs)[0]
            obs, _, terminated, truncated, _ = eval_env.step(int(action))
            if terminated or truncated:
                trajectories.append(eval_env.trajectory)
                break

    with open(
        os.path.join(output_folder, f"{model_name}_trajectories_aftertraining.pkl"),
        "wb",
    ) as f:
        pickle.dump(trajectories, f)


def prepare_environment(
    starting_locations,
    target_trajectories_emb,
    target_trajectories_id,
    graph: ig.Graph,
    embeddings_min: float,
    embeddings_max: float,
):
    emb_env = WebsiteEnvironment(
        graph,
        starting_locations,
        config.max_trajectory_length,
        embeddings_min,
        embeddings_max,
        reward=LevenshteinReward(target_trajectories_emb, penalty=2),
        reward_needs_embeddings=True,
    )
    emb_eval_env = WebsiteEnvironment(
        graph,
        starting_locations,
        config.max_trajectory_length,
        embeddings_min,
        embeddings_max,
        reward=LevenshteinReward(target_trajectories_emb, penalty=2),
        reward_needs_embeddings=True,
    )
    nonemb_env = WebsiteEnvironment(
        graph,
        starting_locations,
        config.max_trajectory_length,
        embeddings_min,
        embeddings_max,
        reward=LevenshteinReward(target_trajectories_id, distance=exact_comparison_distance, penalty=1),
        reward_needs_embeddings=False,
    )
    nonemb_eval_env = WebsiteEnvironment(
        graph,
        starting_locations,
        config.max_trajectory_length,
        embeddings_min,
        embeddings_max,
        reward=LevenshteinReward(target_trajectories_id, distance=exact_comparison_distance, penalty=1),
        reward_needs_embeddings=False,
    )
    return emb_env, emb_eval_env, nonemb_env, nonemb_eval_env


if __name__ == "__main__":
    config = parse_args()
    print_config_summary(config)

    output_folder, _ = utils.create_experiment_hash(config.n_steps, config.gamma, ".experiments")

    print("Loading embeddings...")
    embs_with_vocab = preprocessing.load_glove_embeddings(config.glove_vocab_file, config.glove_vectors_file)

    print("Loading environment...")
    graph, trajectories = trajectory_sampling.load_environment(
        graph_filepath=config.graph_file, trajectories_filepath=config.trajectories_file
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

    graph_name_id_map = {v["name"]: v.index for v in graph.vs}
    # map original trajectories to graph node ids
    trajectories = [
        np.array(list(map(graph_name_id_map.get, map(str, t)))) for t in tqdm(trajectories, desc="Mapping trajectories")
    ]

    # We remove any trajectory which can't be mapped to the graph
    trajectories = [t for t in tqdm(trajectories, desc="Filtering trajectories") if not np.any(t == None)]  # noqa: E711

    with open(f"{output_folder}/graph_with_embeddings.pkl", "wb") as f:
        pickle.dump(graph, f)

    # The line below is because of this issue during training https://github.com/DLR-RM/stable-baselines3/issues/1596
    # Feel free to remove if/when that issue is fixed (notice, github's issue is already closed at the time of writing, but not fixed)
    Distribution.set_default_validate_args(False)

    for e in tqdm(range(config.num_experiments), desc=f"Running {config.num_experiments} experiments..."):
        group_trajectories_id, group_trajectories_emb, starting_locations = (
            trajectory_sampling.sample_n_semantically_similar_trajectories(
                trajectories,
                num_trajectories=config.num_trajectories,
                graph=graph,
                min_traj_length=config.min_trajectory_length,
                max_traj_length=config.max_trajectory_length,
                max_env_steps=config.max_trajectory_length,
                exit_id=len(graph.vs) - 1,
            )
        )
        emb_env, emb_eval_env, nonemb_env, nonemb_eval_env = prepare_environment(
            starting_locations,
            group_trajectories_emb,
            group_trajectories_id,
            graph,
            embeddings.min(),
            embeddings.max(),
        )

        exp_folder = os.path.join(output_folder, f"experiment_{e + 1}")
        os.makedirs(exp_folder, exist_ok=True)

        with open(os.path.join(output_folder, "trajectories_target.pkl"), "wb") as f:
            pickle.dump({"trajectory_id": group_trajectories_id, "trajectory_emb": group_trajectories_emb}, f)

        lev_emb_model = MaskablePPO(
            "MlpPolicy", emb_env, gamma=config.gamma, verbose=1, device=config.device, n_steps=config.n_steps
        )
        lev_nonemb_model = MaskablePPO(
            "MlpPolicy", nonemb_env, gamma=config.gamma, verbose=1, device=config.device, n_steps=config.n_steps
        )

        early_stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10)

        emb_model_name = "levenshtein_emb"
        nonemb_model_name = "levenshtein_nonemb"

        emb_save_traj_callback = SaveTrajectoryCallback(config.eval_episodes, exp_folder, emb_model_name)
        emb_callback = EvalCallback(
            emb_eval_env, callback_after_eval=early_stop_callback, callback_on_new_best=emb_save_traj_callback
        )

        noemb_save_traj_callback = SaveTrajectoryCallback(config.eval_episodes, exp_folder, nonemb_model_name)
        nonemb_callback = EvalCallback(
            nonemb_eval_env, callback_after_eval=early_stop_callback, callback_on_new_best=noemb_save_traj_callback
        )

        do_train_and_eval(lev_emb_model, emb_eval_env, config, exp_folder, emb_model_name, callback=emb_callback)
        do_train_and_eval(
            lev_nonemb_model, nonemb_eval_env, config, exp_folder, nonemb_model_name, callback=nonemb_callback
        )
