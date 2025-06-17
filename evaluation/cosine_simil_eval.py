import argparse
import functools
import itertools
import json
import multiprocessing as mp
import os
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Literal

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from evaluating_trajectories.distances import wasserstein_distance
from evaluating_trajectories.utils import utils

agent_traj_pattern = re.compile(r"levenshtein_(\w+)_trajectories_(.+)\.pkl")


def groupby_model(model_trajectories: list[re.Match[str]]):
    mtraj = sorted(model_trajectories, key=lambda m: m.groups()[0])
    first_model = mtraj[0].groups()[0]
    group = []
    for i in range(len(mtraj)):
        traj = mtraj[i]
        if traj.groups()[0] == first_model:
            group.append(traj.string)
        else:
            yield group, first_model
            yield [m.string for m in mtraj[i:]], traj.groups()[0]
            break


def collect_experiment_results(
    exp_folder: str,
    experiment_name: str,
    evaluate_all_checkpoints: bool,
    graph: ig.Graph,
    target_group: Literal["all", "train", "test"] = "all",
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    model_trajectories = os.listdir(exp_folder)
    model_trajectories = list(filter(lambda p: p is not None, map(agent_traj_pattern.match, model_trajectories)))

    match target_group:
        case "all":
            target_group_fname = "trajectories_target.pkl"
        case "train":
            target_group_fname = "trajectories_target_train.pkl"
        case "test":
            target_group_fname = "trajectories_target_test.pkl"
        case _:
            raise ValueError(f"Unknown target group: {target_group}")

    with open(os.path.join(exp_folder, target_group_fname), "rb") as f:
        target_group = pickle.load(f)

    groups_it = groupby_model(model_trajectories)  # pyright: ignore[reportArgumentType]

    flat_results = []
    wasserstein_results = {experiment_name: {}}
    for group, group_name in groups_it:
        latest_checkpoint = next(filter(lambda t: "aftertraining" in t, group))

        if evaluate_all_checkpoints:
            checkpoints_to_evaluate = group
        else:
            checkpoints_to_evaluate = [latest_checkpoint]

        for model_file in checkpoints_to_evaluate:
            with open(os.path.join(exp_folder, model_file), "rb") as f:
                agent_trajectories = pickle.load(f)

            max_traj_length = max(
                itertools.chain(map(len, target_group["trajectory_id"]), map(len, agent_trajectories))
            )
            target_traj_padded = np.array(
                [
                    np.pad(
                        traj,
                        (0, max_traj_length - len(traj)),
                        mode="constant",
                        constant_values=len(graph.vs),
                    )
                    for traj in target_group["trajectory_id"]
                ]
            )
            agent_traj_padded = np.array(
                [
                    np.pad(
                        traj,
                        (0, max_traj_length - len(traj)),
                        mode="constant",
                        constant_values=len(graph.vs),
                    )
                    for traj in agent_trajectories
                ]
            )

            wass_dist = wasserstein_distance.wasserstein_uniform(target_traj_padded, agent_traj_padded)
            wasserstein_results[experiment_name][group_name] = wass_dist

            for i, traj in enumerate(agent_trajectories):
                avg_embs = np.array(graph.vs[traj]["embedding"]).mean(axis=0).reshape(1, -1)
                for j, group in enumerate(target_group["trajectory_emb"]):
                    avg_group = group.mean(axis=0).reshape(1, -1)
                    score = cosine_similarity(avg_embs, avg_group).item()
                    agent_entropy = entropy(list(Counter(traj).values())).item()
                    target_entropy = entropy(list(Counter(target_group["trajectory_id"][j]).values())).item()
                    flat_results.append(
                        {
                            "experiment_name": experiment_name,
                            "model": group_name,
                            "agent_id": f"agent_{i}",
                            "user_id": f"user_{j}",
                            "score": score,
                            "agent_entropy": agent_entropy,
                            "user_entropy": target_entropy,
                        }
                    )
    return pd.DataFrame(flat_results), wasserstein_results


def process_experiment_directory(experiment_dir, args, graph):
    df_exp, wass_results = collect_experiment_results(
        os.path.join(args.exp_folder, experiment_dir),
        experiment_dir,
        args.evaluate_all_checkpoints,
        graph,
        target_group=args.target_group,
    )
    return df_exp, wass_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cosine similarity of trajectories")
    parser.add_argument(
        "--exp_folder",
        type=str,
        help="Path to the experiment folder containing the runs",
    )
    parser.add_argument(
        "--evaluate-all-checkpoints",
        action="store_true",
        help="Evaluate all checkpoints in the experiment folder",
    )
    parser.add_argument(
        "--evaluate-latest-checkpoint",
        action="store_true",
        help="Evaluate the latest checkpoint in the experiment folder",
    )
    parser.add_argument(
        "--target-group",
        choices=["all", "train", "test"],
        help="Target group to evaluate trajectories against",
        default="all",
    )
    parser.add_argument(
        "--njobs", type=int, default=1, help="Number of parallel jobs to run for processing experiments"
    )
    args = parser.parse_args()

    if not os.path.exists(args.exp_folder):
        print(f"{args.exp_folder} does not exist")
        exit(1)

    with open(os.path.join(args.exp_folder, "graph_with_embeddings.pkl"), "rb") as f:
        graph = pickle.load(f)

    experiment_directories = list(filter(lambda d: "experiment_" in d, os.listdir(args.exp_folder)))

    worker_func = functools.partial(process_experiment_directory, args=args, graph=graph)
    with mp.Pool(processes=args.njobs) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(
                    worker_func, experiment_directories, chunksize=len(experiment_directories) // args.njobs
                ),
                total=len(experiment_directories),
                desc="Collecting experiment data...",
            )
        )

    df = pd.DataFrame()
    wasserstein_dict = {}
    for df_exp, wass_results in results:
        df = pd.concat((df, df_exp), ignore_index=True)
        wasserstein_dict |= wass_results

    df["model"] = df["model"].replace({"emb": "Cos-Levensht.", "nonemb": "Trad-Levensht."})
    wass_df = pd.DataFrame(wasserstein_dict).T
    wass_df = wass_df.rename(columns={"emb": "Cos-Levensht.", "nonemb": "Trad-Levensht."})

    path = Path(args.exp_folder).parent / "parameters.json"
    with open(path, "r") as f:
        params = json.load(f)

    experiment_hash = utils.get_params_hash_str(params)
    plot_path = "evaluation/plots"
    os.makedirs(plot_path, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="model", y="score")
    plt.title(f"Cosine Sim. Score Distribution. Total number of traj. {params['num_trajectories']}")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.savefig(Path(plot_path) / f"cosine_simil_score_distribution_{experiment_hash}.png")
    plt.close()

    wass_melted = wass_df.reset_index().melt(
        id_vars="index",
        value_vars=["Cos-Levensht.", "Trad-Levensht."],
        var_name="model",
        value_name="wasserstein_distance",
    )

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=wass_melted, x="model", y="wasserstein_distance")
    plt.title(f"Wasserstein Distance Distribution by Model. Total number of traj. {params['num_trajectories']}")
    plt.xlabel("Model")
    plt.ylabel("Wasserstein Distance")
    plt.savefig(Path(plot_path) / f"wasserstein_distance_distribution_{experiment_hash}.png")
    plt.close()

    # Entropy plot
    plot_data = []

    cos_agent = df[df["model"] == "Cos-Levensht."]["agent_entropy"]
    plot_data.extend([{"category": "Agent (Cos-Levensht.)", "entropy": val} for val in cos_agent])

    trad_agent = df[df["model"] == "Trad-Levensht."]["agent_entropy"]
    plot_data.extend([{"category": "Agent (Trad-Levensht.)", "entropy": val} for val in trad_agent])

    user_entropy = df["user_entropy"].unique()
    plot_data.extend([{"category": "User", "entropy": val} for val in user_entropy])

    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_df, x="category", y="entropy")
    plt.title(f"Entropy across experiments. Total number of traj. {params['num_trajectories']}")
    plt.xlabel("Model")
    plt.ylabel("Entropy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(plot_path) / f"entropy_distribution_{experiment_hash}.png")
    plt.close()
