import argparse
import logging
import os
import pickle
from dataclasses import asdict, dataclass
from typing import Literal

import dotenv
import igraph as ig
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.type_aliases import MaybeCallback
from torch.distributions import Distribution
from tqdm import tqdm

from evaluating_trajectories.dataset import preprocessing
from evaluating_trajectories.distances.levenshtein_distance import exact_comparison_distance
from evaluating_trajectories.environment import trajectory_sampling
from evaluating_trajectories.environment.rewards import LevenshteinReward
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
    skip_consecutive_duplicates: bool
    pad_target_trajectories: bool
    sampling_strategy: Literal["cosine_simil", "spherical_kmeans"]
    select_cluster_closest_to_num_trajectories: bool
    levenshtein_reward_strategy: Literal["plain", "diff", "shift"]
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
        "--num-trajectories",
        type=int,
        default=int(os.getenv("NUM_TRAJECTORIES", "256")),
        help="Number of trajectories (will split in train and test)",
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
        "--skip-consecutive-duplicates",
        action="store_true",
        default=os.getenv("SKIP_CONSECUTIVE_DUPLICATES", "False") in ["True", "true", "1"],
        help="Remove duplicate node visits when they occur consecutively in sequence",
    )
    parser.add_argument(
        "--pad-target-trajectories",
        action="store_true",
        default=os.getenv("PAD_TARGET_TRAJECTORIES", "False") in ["True", "true", "1"],
        help="Pad target trajectories to max trajectory length",
    )
    parser.add_argument(
        "--sampling-strategy",
        default=os.getenv("SAMPLING_STRATEGY", "cosine_simil"),
        choices=["cosine_simil", "kmeans"],
        help="Sampling strategy for selecting trajectories",
    )
    parser.add_argument(
        "--select-cluster-closest-to-num-trajectories",
        action="store_true",
        default=os.getenv("SELECT_CLUSTER_CLOSEST_TO_NUM_TRAJECTORIES", "False") in ["True", "true", "1"],
        help="Select cluster closest to the number of trajectories (instead of random)",
    )
    parser.add_argument(
        "--levenshtein-reward-strategy",
        default=os.getenv("LEVENSHTEIN_REWARD_STRATEGY", "plain"),
        choices=["plain", "diff", "shift"],
        help="Levenshtein reward strategy (see comment in environment/rewards.py)",
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
        skip_consecutive_duplicates=args.skip_consecutive_duplicates,
        pad_target_trajectories=args.pad_target_trajectories,
        sampling_strategy=args.sampling_strategy,
        select_cluster_closest_to_num_trajectories=args.select_cluster_closest_to_num_trajectories,
        levenshtein_reward_strategy=args.levenshtein_reward_strategy,
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
            action_masks = get_action_masks(eval_env)
            action = model.predict(obs, action_masks=action_masks)[0]
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
        reward=LevenshteinReward(target_trajectories_emb, strategy=config.levenshtein_reward_strategy, penalty=2),
        reward_needs_embeddings=True,
    )
    emb_eval_env = WebsiteEnvironment(
        graph,
        starting_locations,
        config.max_trajectory_length,
        embeddings_min,
        embeddings_max,
        reward=LevenshteinReward(target_trajectories_emb, strategy=config.levenshtein_reward_strategy, penalty=2),
        reward_needs_embeddings=True,
    )
    nonemb_env = WebsiteEnvironment(
        graph,
        starting_locations,
        config.max_trajectory_length,
        embeddings_min,
        embeddings_max,
        reward=LevenshteinReward(
            target_trajectories_id,
            distance=exact_comparison_distance,
            strategy=config.levenshtein_reward_strategy,
            penalty=1,
        ),
        reward_needs_embeddings=False,
    )
    nonemb_eval_env = WebsiteEnvironment(
        graph,
        starting_locations,
        config.max_trajectory_length,
        embeddings_min,
        embeddings_max,
        reward=LevenshteinReward(
            target_trajectories_id,
            distance=exact_comparison_distance,
            strategy=config.levenshtein_reward_strategy,
            penalty=1,
        ),
        reward_needs_embeddings=False,
    )
    return emb_env, emb_eval_env, nonemb_env, nonemb_eval_env


if __name__ == "__main__":
    config = parse_args()
    print_config_summary(config)

    output_folder = utils.create_experiment_hash_dir(asdict(config), ".experiments")

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
        if config.sampling_strategy == "cosine_simil":
            group_trajectories_id, group_trajectories_emb, starting_locations = (
                trajectory_sampling.sample_n_semantically_similar_trajectories(
                    trajectories,
                    num_trajectories=config.num_trajectories,
                    graph=graph,
                    min_traj_length=config.min_trajectory_length,
                    max_traj_length=config.max_trajectory_length,
                    max_env_steps=config.max_trajectory_length,
                    exit_id=len(graph.vs) - 1,
                    min_similarity=0.85,
                    remove_consecutive_duplicates=config.skip_consecutive_duplicates,
                    do_padding=config.pad_target_trajectories,
                )
            )
        else:
            group_trajectories_id, group_trajectories_emb, starting_locations = (
                trajectory_sampling.sample_with_faiss_kmeans(
                    trajectories,
                    num_trajectories=config.num_trajectories,
                    graph=graph,
                    min_traj_length=config.min_trajectory_length,
                    max_traj_length=config.max_trajectory_length,
                    remove_consecutive_duplicates=config.skip_consecutive_duplicates,
                    select_cluster_closest_to_num_trajectories=config.select_cluster_closest_to_num_trajectories,
                )
            )

        if len(group_trajectories_id) > config.num_trajectories and len(group_trajectories_id) != len(
            group_trajectories_emb
        ):
            logging.warning(
                f"Experiment {e}: number of sampled trajectories does not match the expected number ({config.num_trajectories}) or the sampled trajectories are inconsistent ({len(group_trajectories_id)}, {len(group_trajectories_emb)}). Skipping..."
            )
            continue

        total_num_samples = len(group_trajectories_id)
        training_size = round(total_num_samples * 0.8)
        logging.info(
            f"Experiment {e}: Total number of sampled trajectories: {total_num_samples}, "
            f"Training size: {training_size}, Evaluation size: {total_num_samples - training_size}"
        )
        group_trajectories_id_tr = group_trajectories_id[:training_size]
        group_trajectories_emb_tr = group_trajectories_emb[:training_size]
        group_trajectories_id_te = group_trajectories_id[training_size:]
        group_trajectories_emb_te = group_trajectories_emb[training_size:]

        emb_env, emb_eval_env, nonemb_env, nonemb_eval_env = prepare_environment(
            starting_locations,
            group_trajectories_emb_tr,
            group_trajectories_id_tr,
            graph,
            embeddings.min(),
            embeddings.max(),
        )

        exp_folder = os.path.join(output_folder, f"experiment_{e + 1}")
        os.makedirs(exp_folder, exist_ok=True)

        with open(os.path.join(exp_folder, "trajectories_target.pkl"), "wb") as f:
            pickle.dump({"trajectory_id": group_trajectories_id, "trajectory_emb": group_trajectories_emb}, f)

        with open(os.path.join(exp_folder, "trajectories_target_train.pkl"), "wb") as f:
            pickle.dump(
                {
                    "trajectory_id": group_trajectories_id_tr,
                    "trajectory_emb": group_trajectories_emb_tr,
                },
                f,
            )

        with open(os.path.join(exp_folder, "trajectories_target_test.pkl"), "wb") as f:
            pickle.dump(
                {
                    "trajectory_id": group_trajectories_id_te,
                    "trajectory_emb": group_trajectories_emb_te,
                },
                f,
            )

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

        lev_emb_model.save(os.path.join(exp_folder, "model_with_cosine_levensth.zip"))
        lev_nonemb_model.save(os.path.join(exp_folder, "model_with_exact_levensth.zip"))
