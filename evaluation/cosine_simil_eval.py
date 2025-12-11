import argparse
import functools
import itertools
import json
import multiprocessing as mp
import os
import pickle
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import duckdb
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

agent_traj_pattern = re.compile(r"(levenshtein|abid_zou)_?(\w+)?_trajectories_(.+)\.pkl")


def plot_score_heatmaps(
    df, path: str, parameters: dict[str, Any], exp_hash: str, n_experiments=1, max_users=10, random_seed=42
):
    """
    CLAUDE GENERATED FUNCTION TO PLOT THE HEATMAPS
    Create heatmaps comparing agents vs users for score metrics.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: experiment_name, model, agent_id, user_id, score
    n_experiments : int, default=1
        Number of experiments to display (1-4 recommended)
        If 1: creates side-by-side heatmaps for each model
        If >1: creates grid with experiments as rows, models as columns
    random_seed : int, default=42
        Random seed for reproducibility
    """
    np.random.seed(random_seed)

    # Select random experiments
    all_experiments = df["experiment_name"].unique()
    selected_experiments = np.random.choice(all_experiments, size=n_experiments, replace=False)

    # Get unique models
    models = df["model"].unique()

    if n_experiments == 1:
        # Single experiment: side-by-side heatmaps for each model
        experiment = selected_experiments[0]
        df_filtered = df[df["experiment_name"] == experiment]

        # Get number of unique users and randomly select agents
        n_users = min(df_filtered["user_id"].nunique(), max_users)
        selected_agents = np.random.choice(df_filtered["agent_id"].unique(), size=n_users, replace=False)
        selected_users = np.random.choice(df_filtered["user_id"].unique(), size=n_users, replace=False)
        df_filtered = df_filtered[
            df_filtered["agent_id"].isin(selected_agents) & df_filtered["user_id"].isin(selected_users)
        ]

        # Create figure
        fig, axes = plt.subplots(1, len(models), figsize=(10 * len(models), 8))

        # Handle case where there's only one model
        if len(models) == 1:
            axes = [axes]

        for idx, model in enumerate(models):
            model_data = df_filtered[df_filtered["model"] == model]
            heatmap_data = model_data.pivot(index="agent_id", columns="user_id", values="score")

            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".3f",
                cmap="YlOrRd",
                cbar_kws={"label": "Score"},
                ax=axes[idx],
                vmin=0,
                vmax=1,
                annot_kws={"fontsize": 10},
            )

            axes[idx].set_title(f"Model: {model}\nExperiment: {experiment}", fontsize=14)
            axes[idx].set_xlabel("User ID", fontsize=12)
            axes[idx].set_ylabel("Agent ID", fontsize=12)

        print(f"Selected experiment: {experiment}")
        print(f"Number of agents selected: {n_users}")
        print(f"Number of users: {n_users}")

    else:
        # Multiple experiments: grid layout
        fig, axes = plt.subplots(
            len(selected_experiments), len(models), figsize=(10 * len(models), 8 * len(selected_experiments))
        )

        # Handle edge cases for axes indexing
        if len(selected_experiments) == 1 and len(models) == 1:
            axes = np.array([[axes]])
        elif len(selected_experiments) == 1:
            axes = axes.reshape(1, -1)
        elif len(models) == 1:
            axes = axes.reshape(-1, 1)

        for exp_idx, experiment in enumerate(selected_experiments):
            df_exp = df[df["experiment_name"] == experiment]

            # Get number of unique users and randomly select agents
            n_users = min(df_exp["user_id"].nunique(), max_users)
            selected_users = np.random.choice(df_exp["user_id"].unique(), size=n_users, replace=False)
            selected_agents = np.random.choice(df_exp["agent_id"].unique(), size=n_users, replace=False)
            df_exp = df_exp[df_exp["agent_id"].isin(selected_agents) & df_exp["user_id"].isin(selected_users)]

            for model_idx, model in enumerate(models):
                model_data = df_exp[df_exp["model"] == model]
                heatmap_data = model_data.pivot(index="agent_id", columns="user_id", values="score")

                sns.heatmap(
                    heatmap_data,
                    annot=True,
                    fmt=".3f",
                    cmap="YlOrRd",
                    cbar_kws={"label": "Score"},
                    ax=axes[exp_idx, model_idx],
                    vmin=0,
                    vmax=1,
                    annot_kws={"fontsize": 8},
                )

                # Set title (model name on top row only)
                if exp_idx == 0:
                    axes[exp_idx, model_idx].set_title(f"Model: {model}", fontsize=14, fontweight="bold")

                # Set ylabel (experiment name on first column only)
                if model_idx == 0:
                    axes[exp_idx, model_idx].set_ylabel(f"{experiment}\n\nAgent ID", fontsize=12)
                else:
                    axes[exp_idx, model_idx].set_ylabel("Agent ID", fontsize=10)

                axes[exp_idx, model_idx].set_xlabel("User ID", fontsize=10)

        print(f"Selected experiments: {selected_experiments}")

    num_trajectories = parameters.get("num_trajectories", "N/A")
    clustering_technique = "Close" if parameters.get("select_cluster_closest_to_num_trajectories", False) else "Rand"
    plt.suptitle(
        f"Score Heatmaps: Agents vs Users. Num Traj: {num_trajectories}, Clustering: {clustering_technique}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    print(f"Number of models: {len(models)}")
    plt.tight_layout()
    plt.savefig(Path(path) / f"score_heatmaps_{n_experiments}_experiments_{exp_hash}.png")
    plt.close()


def groupby_model(model_trajectories: list[re.Match[str]]):
    def get_group_name(match):
        return "_".join(match.groups("")[:2])

    mtraj = sorted(model_trajectories, key=get_group_name)
    prev_model = get_group_name(mtraj[0])
    group = []
    for i in range(len(mtraj)):
        traj = mtraj[i]
        current_model = get_group_name(traj)
        if current_model == prev_model:
            group.append(traj.string)
        else:
            yield group, prev_model
            prev_model = current_model
            group = [traj.string]

    if group:
        yield group, prev_model


def collect_experiment_results(
    exp_folder: str,
    experiment_name: str,
    evaluate_all_checkpoints: bool,
    num_nodes: int,
    node_embeddings: np.ndarray,
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
        try:
            latest_checkpoint = next(filter(lambda t: "aftertraining" in t, group))
        except StopIteration as e:
            warnings.warn(f"No aftertraining checkpoint found for model {group_name} in experiment {experiment_name}")
            raise e

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
                        constant_values=num_nodes,
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
                        constant_values=num_nodes,
                    )
                    for traj in agent_trajectories
                ]
            )

            wass_dist = wasserstein_distance.wasserstein_uniform(target_traj_padded, agent_traj_padded)
            wasserstein_results[experiment_name][group_name] = wass_dist

            agent_counter = Counter({n: 0 for n in range(num_nodes)})
            target_counter = Counter({n: 0 for n in range(num_nodes)})
            for i, traj in enumerate(agent_trajectories):
                # Use passed embeddings instead of graph.vs
                avg_embs = node_embeddings[traj].mean(axis=0).reshape(1, -1)
                for j, group in enumerate(target_group["trajectory_emb"]):
                    agent_counter.clear()
                    target_counter.clear()
                    avg_group = group.mean(axis=0).reshape(1, -1)
                    score = cosine_similarity(avg_embs, avg_group).item()
                    agent_counter.update(traj)
                    target_counter.update(target_group["trajectory_id"][j])
                    agent_entropy = entropy(list(agent_counter.values())).item()
                    target_entropy = entropy(list(target_counter.values())).item()
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


def process_experiment_directory(args_tuple):
    """Wrapper function for multiprocessing that unpacks arguments."""
    exp_folder, exp_dir, evaluate_all_checkpoints, num_nodes, target_group, node_embeddings = args_tuple
    try:
        return collect_experiment_results(
            os.path.join(exp_folder, exp_dir),
            exp_dir,
            evaluate_all_checkpoints,
            num_nodes,
            node_embeddings,
            target_group=target_group,
        )
    except:
        return None


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

    node_embeddings = np.array([v["embedding"] for v in graph.vs])
    process_args = [
        (args.exp_folder, exp_dir, args.evaluate_all_checkpoints, len(graph.vs), args.target_group, node_embeddings)
        for exp_dir in experiment_directories
    ]

    if args.njobs > 1:
        with mp.Pool(processes=args.njobs) as pool:
            results = list(
                filter(
                    bool,
                    tqdm(
                        pool.imap(process_experiment_directory, process_args),
                        total=len(experiment_directories),
                        desc="processing results...",
                    ),
                )
            )
    else:
        # Fallback to sequential processing
        results = []
        for exp_dir in tqdm(experiment_directories, desc="processing results..."):
            try:
                df_exp, wass_results = collect_experiment_results(
                    os.path.join(args.exp_folder, exp_dir),
                    exp_dir,
                    args.evaluate_all_checkpoints,
                    len(graph.vs),
                    node_embeddings,
                    target_group=args.target_group,
                )
                results.append((df_exp, wass_results))
            except:
                ...

    df = pd.DataFrame()
    wasserstein_dict = {}
    for df_exp, wass_results in results:
        df = pd.concat((df, df_exp), ignore_index=True)
        wasserstein_dict |= wass_results

    df["model"] = df["model"].replace(
        {"levenshtein_emb": "Cos-Levensht.", "levenshtein_nonemb": "Trad-Levensht.", "abid_zou_": "Abid&Zou"}
    )
    wass_df = pd.DataFrame(wasserstein_dict).T
    wass_df = wass_df.rename(
        columns={"levenshtein_emb": "Cos-Levensht.", "levenshtein_nonemb": "Trad-Levensht.", "abid_zou_": "Abid&Zou"}
    )

    path = Path(args.exp_folder).parent / "parameters.json"
    with open(path, "r") as f:
        params = json.load(f)

    experiment_hash = utils.get_params_hash_str(params)
    plot_path = "evaluation/plots"
    os.makedirs(plot_path, exist_ok=True)

    cluster_technique = "Close" if params["select_cluster_closest_to_num_trajectories"] else "Rand"
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="model", y="score")
    plt.title(
        f"Cosine Sim. Score Distribution. Total number of traj. {params['num_trajectories']}, Clustering: {cluster_technique}"
    )
    plt.xlabel("Model")
    plt.ylabel("Cosine similarity")
    plt.savefig(Path(plot_path) / f"cosine_simil_score_distribution_{experiment_hash}.png")
    plt.close()

    wass_melted = wass_df.reset_index().melt(
        id_vars="index",
        value_vars=["Cos-Levensht.", "Trad-Levensht.", "Abid&Zou"],
        var_name="model",
        value_name="wasserstein_distance",
    )

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=wass_melted, x="model", y="wasserstein_distance")
    plt.title(
        f"Wasserstein Distance Distribution by Model. Total number of traj. {params['num_trajectories']}, Clustering: {cluster_technique}"
    )
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

    abid_agent = df[df["model"] == "Abid&Zou"]["agent_entropy"]
    plot_data.extend([{"category": "Agent (Abid&Zou)", "entropy": val} for val in abid_agent])

    user_entropy = df["user_entropy"].unique()
    plot_data.extend([{"category": "User", "entropy": val} for val in user_entropy])

    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_df, x="category", y="entropy")
    plt.title(
        f"Entropy across experiments. Total number of traj. {params['num_trajectories']}, Clustering: {cluster_technique}"
    )
    plt.xlabel("Model")
    plt.ylabel("Entropy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(plot_path) / f"entropy_distribution_{experiment_hash}.png")
    plt.close()

    plot_score_heatmaps(df, plot_path, parameters=params, exp_hash=experiment_hash, n_experiments=1, random_seed=42)
    plot_score_heatmaps(df, plot_path, parameters=params, exp_hash=experiment_hash, n_experiments=4, random_seed=42)

    latex_table = (
        duckdb.sql(
            """
      SELECT
          model,
          concat(round(avg(score), 3), '+/-', round(stddev(score),3)) as 'Avg cos. sim.',
          concat(round(avg(agent_entropy), 3), '+/-', round(stddev(agent_entropy), 3)) as 'Avg agent entropy',
          concat(round(avg(user_entropy),3), '+/-', round(stddev(user_entropy),3)) as 'Avg user entropy'
      from df
      group by model
    """
        )
        .df()
        .to_latex()
    )
    with open(Path(plot_path) / f"cosine_similarity_evaluation_table_{experiment_hash}.tex", "w") as f:
        f.write(latex_table)
