import argparse
import itertools
import json
import multiprocessing as mp
import os
import pickle
import random
import re
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import duckdb
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy, mannwhitneyu, shapiro, ttest_rel, wilcoxon
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from rl_semantic_trajectories.distances import wasserstein_distance
from rl_semantic_trajectories.utils import utils

agent_traj_pattern = re.compile(r"(levenshtein|abid_zou)_?(\w+)?_trajectories_(.+)\.pkl")

random.seed(42)


def compute_random_walk_trajectories(
    graph: ig.Graph,
    start_vertices: list[int],
    walk_lengths: list[int],
) -> list[list[int]]:
    return [graph.random_walk(start, length) for start, length in zip(start_vertices, walk_lengths)]


def pad_trajectories(trajectories: list[list[int]], max_length: int, pad_value: int) -> np.ndarray:
    """Pad a list of trajectories to the same length."""
    return np.array(
        [np.pad(traj, (0, max_length - len(traj)), mode="constant", constant_values=pad_value) for traj in trajectories]
    )


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


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def perform_statistical_tests(df: pd.DataFrame, wass_df: pd.DataFrame, alpha: float = 0.05):
    """
    Perform statistical tests comparing models across different metrics.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with cosine similarity scores and entropy values
    wass_df : pd.DataFrame
        DataFrame with Wasserstein distances
    alpha : float
        Significance level (default: 0.05)

    Returns:
    --------
    dict : Dictionary containing test results for different metrics
    """
    results = {
        "cosine_similarity": [],
        "wasserstein_distance": [],
        "agent_entropy": [],
    }

    models = df["model"].unique()
    model_pairs = list(combinations(models, 2))
    n_comparisons = len(model_pairs)
    bonferroni_alpha = alpha / n_comparisons

    # Test 1: Cosine Similarity Comparisons
    print("\n" + "=" * 80)
    print("COSINE SIMILARITY TESTS")
    print("=" * 80)

    for model1, model2 in model_pairs:
        # Get paired scores (same agent_id and user_id)
        df1 = df[df["model"] == model1].set_index(["experiment_name", "agent_id", "user_id"])["score"]
        df2 = df[df["model"] == model2].set_index(["experiment_name", "agent_id", "user_id"])["score"]

        # Find common indices
        common_idx = df1.index.intersection(df2.index)
        scores1 = df1.loc[common_idx].values
        scores2 = df2.loc[common_idx].values

        # Data diagnostics
        print(f"\n{model1} vs {model2}:")
        print(f"  Sample size: {len(scores1)}")
        print(f"  {model1}: mean={np.mean(scores1):.4f}, std={np.std(scores1):.4f}, median={np.median(scores1):.4f}")
        print(f"  {model2}: mean={np.mean(scores2):.4f}, std={np.std(scores2):.4f}, median={np.median(scores2):.4f}")

        # Check for normality (Shapiro-Wilk test on differences)
        differences = scores1 - scores2
        if len(differences) <= 5000:  # Shapiro-Wilk has sample size limit
            _, normality_p = shapiro(differences)
            print(
                f"  Normality test (differences): p={normality_p:.4e} {'(normal)' if normality_p > 0.05 else '(non-normal)'}"
            )

        # Paired t-test
        t_stat, t_pval = ttest_rel(scores1, scores2)

        # Handle numerical underflow for p-values
        t_pval_str = f"{t_pval:.4e}" if t_pval > 1e-300 else "<1e-300"

        # Wilcoxon signed-rank test (non-parametric alternative)
        try:
            w_stat, w_pval = wilcoxon(scores1, scores2, zero_method="wilcox", alternative="two-sided")
            w_pval_str = f"{w_pval:.4e}" if w_pval > 1e-300 else "<1e-300"  # pyright: ignore[reportOperatorIssue]
        except ValueError as e:
            print(f"  Warning: Wilcoxon test failed: {e}")
            w_stat, w_pval = np.nan, np.nan
            w_pval_str = "N/A"

        # Effect size
        effect_size = cohens_d(scores1, scores2)

        results["cosine_similarity"].append(
            {
                "model1": model1,
                "model2": model2,
                "mean_diff": np.mean(scores1) - np.mean(scores2),
                "median_diff": np.median(scores1) - np.median(scores2),
                "t_statistic": t_stat,
                "t_pvalue": t_pval,
                "t_significant": t_pval < bonferroni_alpha,
                "wilcoxon_statistic": w_stat,
                "wilcoxon_pvalue": w_pval,
                "wilcoxon_significant": (
                    w_pval < bonferroni_alpha if not np.isnan(w_pval) else False
                ),  # pyright: ignore[reportOperatorIssue, reportCallIssue, reportArgumentType]
                "cohens_d": effect_size,
                "n_samples": len(scores1),
            }
        )

        print(f"  Mean difference: {np.mean(scores1) - np.mean(scores2):.4f}")
        print(f"  Median difference: {np.median(scores1) - np.median(scores2):.4f}")
        print(f"  Paired t-test: t={t_stat:.4f}, p={t_pval_str} {'*' if t_pval < bonferroni_alpha else ''}")
        print(
            f"  Wilcoxon test: W={w_stat:.4f}, p={w_pval_str} {'*' if w_pval < bonferroni_alpha else ''}"
        )  # pyright: ignore[reportOperatorIssue]
        print(f"  Cohen's d: {effect_size:.4f}")

    # Test 2: Wasserstein Distance Comparisons
    print("\n" + "=" * 80)
    print("WASSERSTEIN DISTANCE TESTS")
    print("=" * 80)

    wass_models = wass_df.columns
    wass_pairs = list(combinations(wass_models, 2))
    wass_bonferroni = alpha / len(wass_pairs)

    for model1, model2 in wass_pairs:
        dist1 = wass_df[model1].dropna().values
        dist2 = wass_df[model2].dropna().values

        print(f"\n{model1} vs {model2}:")
        print(f"  Sample size: {len(dist1)}")
        print(
            f"  {model1}: mean={np.mean(dist1):.4f}, std={np.std(dist1):.4f}"
        )  # pyright: ignore[reportCallIssue, reportArgumentType]
        print(
            f"  {model2}: mean={np.mean(dist2):.4f}, std={np.std(dist2):.4f}"
        )  # pyright: ignore[reportCallIssue, reportArgumentType]

        # Paired t-test
        t_stat, t_pval = ttest_rel(dist1, dist2)
        t_pval_str = f"{t_pval:.4e}" if t_pval > 1e-300 else "<1e-300"

        # Wilcoxon signed-rank test
        try:
            w_stat, w_pval = wilcoxon(dist1, dist2, zero_method="wilcox", alternative="two-sided")
            w_pval_str = f"{w_pval:.4e}" if w_pval > 1e-300 else "<1e-300"  # pyright: ignore[reportOperatorIssue]
        except ValueError as e:
            print(f"  Warning: Wilcoxon test failed: {e}")
            w_stat, w_pval = np.nan, np.nan
            w_pval_str = "N/A"

        # Effect size
        effect_size = cohens_d(dist1, dist2)

        results["wasserstein_distance"].append(
            {
                "model1": model1,
                "model2": model2,
                "mean_diff": np.mean(dist1) - np.mean(dist2),  # pyright: ignore[reportCallIssue, reportArgumentType]
                "median_diff": np.median(dist1)
                - np.median(dist2),  # pyright: ignore[reportCallIssue, reportArgumentType]
                "t_statistic": t_stat,
                "t_pvalue": t_pval,
                "t_significant": t_pval < wass_bonferroni,
                "wilcoxon_statistic": w_stat,
                "wilcoxon_pvalue": w_pval,
                "wilcoxon_significant": (
                    w_pval < wass_bonferroni if not np.isnan(w_pval) else False
                ),  # pyright: ignore[reportOperatorIssue, reportCallIssue, reportArgumentType]
                "cohens_d": effect_size,
                "n_samples": len(dist1),
            }
        )

        print(
            f"  Mean difference: {np.mean(dist1) - np.mean(dist2):.4f}"
        )  # pyright: ignore[reportCallIssue, reportArgumentType]
        print(f"  Paired t-test: t={t_stat:.4f}, p={t_pval_str} {'*' if t_pval < wass_bonferroni else ''}")
        print(
            f"  Wilcoxon test: W={w_stat:.4f}, p={w_pval_str} {'*' if w_pval < wass_bonferroni else ''}"
        )  # pyright: ignore[reportOperatorIssue]
        print(f"  Cohen's d: {effect_size:.4f}")

    # Test 3: Agent Entropy Comparisons
    print("\n" + "=" * 80)
    print("AGENT ENTROPY TESTS")
    print("=" * 80)

    for model1, model2 in model_pairs:
        entropy1 = df[df["model"] == model1]["agent_entropy"].values  # pyright: ignore[reportAttributeAccessIssue]
        entropy2 = df[df["model"] == model2]["agent_entropy"].values  # pyright: ignore[reportAttributeAccessIssue]

        print(f"\n{model1} vs {model2}:")
        print(f"  Sample sizes: {len(entropy1)}")
        print(f"  {model1}: mean={np.mean(entropy1):.4f}, std={np.std(entropy1):.4f}")
        print(f"  {model2}: mean={np.mean(entropy2):.4f}, std={np.std(entropy2):.4f}")

        # Mann-Whitney U test (independent samples)
        u_stat, u_pval = mannwhitneyu(entropy1, entropy2, alternative="two-sided")
        u_pval_str = f"{u_pval:.4e}" if u_pval > 1e-300 else "<1e-300"

        # Effect size
        effect_size = cohens_d(entropy1, entropy2)

        results["agent_entropy"].append(
            {
                "model1": model1,
                "model2": model2,
                "mean_diff": np.mean(entropy1) - np.mean(entropy2),
                "median_diff": np.median(entropy1) - np.median(entropy2),
                "mannwhitney_statistic": u_stat,
                "mannwhitney_pvalue": u_pval,
                "mannwhitney_significant": u_pval < bonferroni_alpha,
                "cohens_d": effect_size,
                "n_samples_1": len(entropy1),
                "n_samples_2": len(entropy2),
            }
        )

        print(f"  Mean difference: {np.mean(entropy1) - np.mean(entropy2):.4f}")
        print(f"  Mann-Whitney U: U={u_stat:.4f}, p={u_pval_str} {'*' if u_pval < bonferroni_alpha else ''}")
        print(f"  Cohen's d: {effect_size:.4f}")

    print(f"\n{'=' * 80}")
    print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.4e}")
    print(f"Number of comparisons: {n_comparisons}")
    print(f"{'=' * 80}\n")

    return results


def save_statistical_results(results: dict, plot_path: str, experiment_hash: str):
    """Save statistical test results to LaTeX tables."""

    # Cosine Similarity Results
    cos_df = pd.DataFrame(results["cosine_similarity"])
    cos_df = cos_df[["model1", "model2", "mean_diff", "t_pvalue", "wilcoxon_pvalue", "cohens_d", "t_significant"]]
    cos_df.columns = ["Model 1", "Model 2", "Mean Diff", "t-test p", "Wilcoxon p", "Cohen's d", "Significant"]
    cos_latex = cos_df.to_latex(index=False, float_format="%.4f")

    with open(Path(plot_path) / f"statistical_tests_cosine_{experiment_hash}.tex", "w") as f:
        f.write("% Cosine Similarity Statistical Tests\n")
        f.write(cos_latex)

    # Wasserstein Distance Results
    wass_df = pd.DataFrame(results["wasserstein_distance"])
    wass_df = wass_df[["model1", "model2", "mean_diff", "t_pvalue", "wilcoxon_pvalue", "cohens_d", "t_significant"]]
    wass_df.columns = ["Model 1", "Model 2", "Mean Diff", "t-test p", "Wilcoxon p", "Cohen's d", "Significant"]
    wass_latex = wass_df.to_latex(index=False, float_format="%.4f")

    with open(Path(plot_path) / f"statistical_tests_wasserstein_{experiment_hash}.tex", "w") as f:
        f.write("% Wasserstein Distance Statistical Tests\n")
        f.write(wass_latex)

    # Entropy Results
    entropy_df = pd.DataFrame(results["agent_entropy"])
    entropy_df = entropy_df[
        ["model1", "model2", "mean_diff", "mannwhitney_pvalue", "cohens_d", "mannwhitney_significant"]
    ]
    entropy_df.columns = ["Model 1", "Model 2", "Mean Diff", "Mann-Whitney p", "Cohen's d", "Significant"]
    entropy_latex = entropy_df.to_latex(index=False, float_format="%.4f")

    with open(Path(plot_path) / f"statistical_tests_entropy_{experiment_hash}.tex", "w") as f:
        f.write("% Agent Entropy Statistical Tests\n")
        f.write(entropy_latex)

    print(f"Statistical test results saved to {plot_path}")


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


def compute_trajectory_metrics(
    trajectories: list[list[int]],
    target_group: dict,
    node_embeddings: np.ndarray,
    num_nodes: int,
    experiment_name: str,
    model_name: str,
) -> list[dict]:
    """Compute cosine similarity and entropy metrics for a set of trajectories against a target group."""
    flat_results = []
    traj_counter = Counter({n: 0 for n in range(num_nodes)})
    target_counter = Counter({n: 0 for n in range(num_nodes)})

    for i, traj in enumerate(trajectories):
        traj_counter.clear()
        traj_counter.update(traj)
        traj_entropy = entropy(list(traj_counter.values())).item()
        avg_embs = node_embeddings[traj].mean(axis=0).reshape(1, -1)

        for j, group in enumerate(target_group["trajectory_emb"]):
            target_counter.clear()
            target_counter.update(target_group["trajectory_id"][j])
            avg_group = group.mean(axis=0).reshape(1, -1)
            score = cosine_similarity(avg_embs, avg_group).item()
            target_entropy = entropy(list(target_counter.values())).item()
            flat_results.append(
                {
                    "experiment_name": experiment_name,
                    "model": model_name,
                    "agent_id": f"agent_{i}",
                    "user_id": f"user_{j}",
                    "score": score,
                    "agent_entropy": traj_entropy,
                    "user_entropy": target_entropy,
                }
            )

    return flat_results


def collect_experiment_results(
    exp_folder: str,
    experiment_name: str,
    evaluate_all_checkpoints: bool,
    num_nodes: int,
    node_embeddings: np.ndarray,
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

    # Compute random walk baseline once — it only depends on the target group
    user_start_vertices = [traj[0] for traj in target_group["trajectory_id"]]
    user_start_vertices = random.choices(user_start_vertices, k=len(user_start_vertices))
    user_walk_lengths = [random.randint(1, 15) for _ in range(len(user_start_vertices))]
    random_walk_trajectories = compute_random_walk_trajectories(graph, user_start_vertices, user_walk_lengths)

    groups_it = groupby_model(model_trajectories)  # pyright: ignore[reportArgumentType]

    flat_results = []
    wasserstein_results = {experiment_name: {}}

    target_traj = target_group["trajectory_id"]

    for group, group_name in groups_it:
        try:
            latest_checkpoint = next(filter(lambda t: "aftertraining" in t, group))
        except StopIteration as e:
            warnings.warn(f"No aftertraining checkpoint found for model {group_name} in experiment {experiment_name}")
            raise e

        checkpoints_to_evaluate = group if evaluate_all_checkpoints else [latest_checkpoint]

        for model_file in checkpoints_to_evaluate:
            with open(os.path.join(exp_folder, model_file), "rb") as f:
                agent_trajectories = pickle.load(f)

            max_traj_length = max(itertools.chain(map(len, target_traj), map(len, agent_trajectories)))
            target_padded = pad_trajectories(target_traj, max_traj_length, num_nodes)
            agent_padded = pad_trajectories(agent_trajectories, max_traj_length, num_nodes)

            wasserstein_results[experiment_name][group_name] = wasserstein_distance.wasserstein_uniform(
                target_padded, agent_padded
            )
            flat_results.extend(
                compute_trajectory_metrics(
                    agent_trajectories, target_group, node_embeddings, num_nodes, experiment_name, group_name
                )
            )

    # Compute random walk baseline with same number of walks as agent trajectories
    n_random_walks = len(agent_trajectories)
    user_start_vertices = [traj[0] for traj in target_group["trajectory_id"]]
    rw_start_vertices = random.choices(user_start_vertices, k=n_random_walks)
    rw_walk_lengths = [random.randint(3, 15) for _ in range(n_random_walks)]
    random_walk_trajectories = compute_random_walk_trajectories(graph, rw_start_vertices, rw_walk_lengths)

    rw_max_len = max(itertools.chain(map(len, target_traj), map(len, random_walk_trajectories)))
    rw_target_padded = pad_trajectories(target_traj, rw_max_len, num_nodes)
    rw_traj_padded = pad_trajectories(random_walk_trajectories, rw_max_len, num_nodes)
    wasserstein_results[experiment_name]["random_walk"] = wasserstein_distance.wasserstein_uniform(
        rw_target_padded, rw_traj_padded
    )
    flat_results.extend(
        compute_trajectory_metrics(
            random_walk_trajectories, target_group, node_embeddings, num_nodes, experiment_name, "random_walk"
        )
    )

    # Fixed-length random walk baseline (15 steps) to isolate the effect of walk length
    rw_fixed_start_vertices = random.choices(user_start_vertices, k=n_random_walks)
    rw_fixed_walk_lengths = [15] * n_random_walks
    random_walk_fixed_trajectories = compute_random_walk_trajectories(
        graph, rw_fixed_start_vertices, rw_fixed_walk_lengths
    )

    rw_fixed_max_len = max(itertools.chain(map(len, target_traj), map(len, random_walk_fixed_trajectories)))
    rw_fixed_target_padded = pad_trajectories(target_traj, rw_fixed_max_len, num_nodes)
    rw_fixed_traj_padded = pad_trajectories(random_walk_fixed_trajectories, rw_fixed_max_len, num_nodes)
    wasserstein_results[experiment_name]["random_walk_fixed"] = wasserstein_distance.wasserstein_uniform(
        rw_fixed_target_padded, rw_fixed_traj_padded
    )
    flat_results.extend(
        compute_trajectory_metrics(
            random_walk_fixed_trajectories,
            target_group,  # pyright: ignore[reportArgumentType]
            node_embeddings,
            num_nodes,
            experiment_name,
            "random_walk_fixed",
        )
    )

    return pd.DataFrame(flat_results), wasserstein_results


def process_experiment_directory(args_tuple):
    """Wrapper function for multiprocessing that unpacks arguments."""
    exp_folder, exp_dir, evaluate_all_checkpoints, num_nodes, target_group, node_embeddings, graph = args_tuple
    try:
        return collect_experiment_results(
            os.path.join(exp_folder, exp_dir),
            exp_dir,
            evaluate_all_checkpoints,
            num_nodes,
            node_embeddings,
            graph,
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
        (
            args.exp_folder,
            exp_dir,
            args.evaluate_all_checkpoints,
            len(graph.vs),
            args.target_group,
            node_embeddings,
            graph,
        )
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
        results = []
        for exp_dir in tqdm(experiment_directories, desc="processing results..."):
            try:
                df_exp, wass_results = collect_experiment_results(
                    os.path.join(args.exp_folder, exp_dir),
                    exp_dir,
                    args.evaluate_all_checkpoints,
                    len(graph.vs),
                    node_embeddings,
                    graph,
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
        {
            "levenshtein_emb": "Cos-Levensht.",
            "levenshtein_nonemb": "Trad-Levensht.",
            "abid_zou_": "Abid&Zou",
            "random_walk": "Random Walk (rand. len)",
            "random_walk_fixed": "Random Walk (fixed 15)",
        }
    )
    wass_df = pd.DataFrame(wasserstein_dict).T
    wass_df = wass_df.rename(
        columns={
            "levenshtein_emb": "Cos-Levensht.",
            "levenshtein_nonemb": "Trad-Levensht.",
            "abid_zou_": "Abid&Zou",
            "random_walk": "Random Walk (rand. len)",
            "random_walk_fixed": "Random Walk (fixed 15)",
        }
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

    available_wass_models = [
        c
        for c in ["Cos-Levensht.", "Trad-Levensht.", "Abid&Zou", "Random Walk (rand. len)", "Random Walk (fixed 15)"]
        if c in wass_df.columns
    ]
    wass_melted = wass_df.reset_index().melt(
        id_vars="index",
        value_vars=available_wass_models,
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

    rw_agent = df[df["model"] == "Random Walk (rand. len)"]["agent_entropy"]
    plot_data.extend([{"category": "Agent (RW rand. len)", "entropy": val} for val in rw_agent])

    rw_fixed_agent = df[df["model"] == "Random Walk (fixed 15)"]["agent_entropy"]
    plot_data.extend([{"category": "Agent (RW fixed 15)", "entropy": val} for val in rw_fixed_agent])

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

    statistical_results = perform_statistical_tests(df, wass_df, alpha=0.05)
    save_statistical_results(statistical_results, plot_path, experiment_hash)

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
