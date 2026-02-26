import argparse
import json
import os
import pickle
import random
import re
from itertools import combinations
from pathlib import Path
from typing import Literal

import duckdb
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu, shapiro
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sklearn.metrics.pairwise import cosine_similarity
from torch.distributions import Distribution
from tqdm import tqdm

from rl_semantic_trajectories.baselines import bernhard_et_al_2016
from rl_semantic_trajectories.environment.rewards import DefaultReward
from rl_semantic_trajectories.environment.website_env import WebsiteEnvironment
from rl_semantic_trajectories.utils import utils

agent_traj_pattern = re.compile(r"(levenshtein|abid_zou)_?(\w+)?_trajectories_(.+)\.pkl")


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def perform_statistical_tests(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Perform statistical tests on next-click prediction results.

    Tests:
    - Cosine similarity: Mann-Whitney U + Cohen's d (all model pairs)
    - Next-click accuracy: chi-squared test on match/total counts (all model pairs)
    """
    results = {"cosine_similarity": [], "accuracy": []}

    models = df["model"].unique().tolist()
    model_pairs = list(combinations(models, 2))
    n_comparisons = len(model_pairs)
    bonferroni_alpha = alpha / n_comparisons

    # --- Cosine similarity ---
    print("\n" + "=" * 80)
    print("COSINE SIMILARITY TESTS (Mann-Whitney U)")
    print("=" * 80)

    for model1, model2 in model_pairs:
        scores1 = df[df["model"] == model1]["cossim"].values
        scores2 = df[df["model"] == model2]["cossim"].values

        print(f"\n{model1} vs {model2}:")
        print(f"  Sample sizes: {len(scores1)}, {len(scores2)}")
        print(f"  {model1}: mean={np.mean(scores1):.4f}, std={np.std(scores1):.4f}, median={np.median(scores1):.4f}")
        print(f"  {model2}: mean={np.mean(scores2):.4f}, std={np.std(scores2):.4f}, median={np.median(scores2):.4f}")

        # Normality check on a subsample
        sample_size = min(len(scores1), len(scores2), 5000)
        _, norm_p1 = shapiro(scores1[:sample_size])
        _, norm_p2 = shapiro(scores2[:sample_size])
        print(f"  Normality (Shapiro): {model1} p={norm_p1:.4e}, {model2} p={norm_p2:.4e}")

        u_stat, u_pval = mannwhitneyu(scores1, scores2, alternative="two-sided")
        u_pval_str = f"{u_pval:.4e}" if u_pval > 1e-300 else "<1e-300"
        effect = cohens_d(scores1, scores2)

        significant = u_pval < bonferroni_alpha
        print(f"  Mann-Whitney U: U={u_stat:.4f}, p={u_pval_str} {'*' if significant else ''}")
        print(f"  Cohen's d: {effect:.4f}")
        print(f"  Mean diff ({model1} - {model2}): {np.mean(scores1) - np.mean(scores2):.4f}")

        results["cosine_similarity"].append(
            {
                "model1": model1,
                "model2": model2,
                "mean_diff": np.mean(scores1) - np.mean(scores2),
                "median_diff": np.median(scores1) - np.median(scores2),
                "mannwhitney_statistic": u_stat,
                "mannwhitney_pvalue": u_pval,
                "significant": significant,
                "cohens_d": effect,
                "n_samples_1": len(scores1),
                "n_samples_2": len(scores2),
            }
        )

    # --- Next-click accuracy ---
    print("\n" + "=" * 80)
    print("NEXT-CLICK ACCURACY TESTS (Chi-squared)")
    print("=" * 80)

    match_counts = (
        df.assign(match=(df["prediction"] == df["truth"]).astype(int))
        .groupby("model")["match"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "matches", "count": "total"})
    )

    acc_bonferroni = alpha / n_comparisons
    for model1, model2 in model_pairs:
        m1 = match_counts.loc[model1]
        m2 = match_counts.loc[model2]
        # Contingency table: [[matches, non-matches], [matches, non-matches]]
        table = np.array([[m1["matches"], m1["total"] - m1["matches"]], [m2["matches"], m2["total"] - m2["matches"]]])
        chi2, p_val, dof, _ = chi2_contingency(table)
        p_val_str = f"{p_val:.4e}" if p_val > 1e-300 else "<1e-300"
        significant = p_val < acc_bonferroni
        acc1 = m1["matches"] / m1["total"]
        acc2 = m2["matches"] / m2["total"]

        print(f"\n{model1} vs {model2}:")
        print(f"  {model1}: {m1['matches']}/{m1['total']} = {acc1:.4f}")
        print(f"  {model2}: {m2['matches']}/{m2['total']} = {acc2:.4f}")
        print(f"  Chi-squared: chi2={chi2:.4f}, p={p_val_str} {'*' if significant else ''}")

        results["accuracy"].append(
            {
                "model1": model1,
                "model2": model2,
                "accuracy_1": acc1,
                "accuracy_2": acc2,
                "accuracy_diff": acc1 - acc2,
                "chi2_statistic": chi2,
                "chi2_pvalue": p_val,
                "significant": significant,
                "n_samples_1": int(m1["total"]),
                "n_samples_2": int(m2["total"]),
            }
        )

    print(f"\n{'=' * 80}")
    print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.4e} ({n_comparisons} comparisons)")
    print(f"{'=' * 80}\n")

    return results


def save_statistical_results(results: dict, results_path: str, experiment_hash: str):
    """Save statistical test results to LaTeX tables."""
    # Cosine similarity
    cos_df = pd.DataFrame(results["cosine_similarity"])
    cos_df = cos_df[["model1", "model2", "mean_diff", "mannwhitney_pvalue", "cohens_d", "significant"]]
    cos_df.columns = ["Model 1", "Model 2", "Mean Diff", "Mann-Whitney p", "Cohen's d", "Significant"]
    with open(Path(results_path) / f"statistical_tests_cossim_{experiment_hash}.tex", "w") as f:
        f.write("% Cosine Similarity Statistical Tests (Mann-Whitney U)\n")
        f.write(cos_df.to_latex(index=False, float_format="%.4f"))

    # Accuracy
    acc_df = pd.DataFrame(results["accuracy"])
    acc_df = acc_df[["model1", "model2", "accuracy_1", "accuracy_2", "accuracy_diff", "chi2_pvalue", "significant"]]
    acc_df.columns = ["Model 1", "Model 2", "Acc. 1", "Acc. 2", "Acc. Diff", "Chi2 p", "Significant"]
    with open(Path(results_path) / f"statistical_tests_accuracy_{experiment_hash}.tex", "w") as f:
        f.write("% Next-Click Accuracy Statistical Tests (Chi-squared)\n")
        f.write(acc_df.to_latex(index=False, float_format="%.4f"))

    print(f"Statistical test results saved to {results_path}")


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


def evaluate_nextclick_preds(
    exp_folder: str,
    graph: ig.Graph,
    target_group: Literal["all", "train", "test"] = "all",
    evaluate_all_checkpoints: bool = False,
):
    model_trajectories = os.listdir(exp_folder)
    model_trajectories = list(filter(lambda p: p is not None, map(agent_traj_pattern.match, model_trajectories)))

    train_target_group_fname = "trajectories_target_train.pkl"
    match target_group:
        case "all":
            target_group_fname = "trajectories_target.pkl"
        case "train":
            target_group_fname = train_target_group_fname
        case "test":
            target_group_fname = "trajectories_target_test.pkl"
        case _:
            raise ValueError(f"Unknown target group: {target_group}")

    with open(os.path.join(exp_folder, target_group_fname), "rb") as f:
        target_group = pickle.load(f)

    with open(os.path.join(exp_folder, train_target_group_fname), "rb") as f:
        train_target_group = pickle.load(f)

    # The class expects the actual number of nodes, so we remove the exit node
    landmark_mc = bernhard_et_al_2016.LandmarkMarkovChain(len(graph.vs) - 1)
    landmark_mc.build_markov_chain(train_target_group["trajectory_id"])

    predictions = []
    for traj in tqdm(target_group["trajectory_id"], desc="Evaluating LandmarkMC", leave=False):
        for j in range(1, len(traj) - 1):
            action_mc = landmark_mc.get_prediction(np.array([traj[j - 1]]))
            if action_mc[0] == -1:
                continue
            mc_pred_emb = graph.vs[action_mc[0]]["embedding"]

            target = int(traj[j])
            target_emb = graph.vs[target]["embedding"]
            mc_cossim = cosine_similarity([target_emb], [mc_pred_emb]).item()
            predictions.append(("markov_chain", int(action_mc[0]), target, j, mc_cossim))

    for traj in tqdm(target_group["trajectory_id"], desc="Evaluating Random Walk", leave=False):
        for j in range(1, len(traj) - 1):
            # Skip the same steps that are unpredictable for the MC/agent models
            action_mc = landmark_mc.get_prediction(np.array([traj[j - 1]]))
            if action_mc[0] == -1:
                continue
            current_node = int(traj[j - 1])
            neighbors = graph.neighbors(current_node, mode="out")
            if not neighbors:
                continue
            rw_action = random.choice(neighbors)
            target = int(traj[j])
            target_emb = graph.vs[target]["embedding"]
            rw_pred_emb = graph.vs[rw_action]["embedding"]
            rw_cossim = cosine_similarity([target_emb], [rw_pred_emb]).item()
            predictions.append(("random_walk", int(rw_action), target, j, rw_cossim))

    embeddings = np.array(graph.vs["embedding"])
    env = WebsiteEnvironment(
        graph,
        # We don't need starting locations here
        starting_locations=[0],
        max_steps=16,
        embedding_min_val=embeddings.min(),
        embedding_max_val=embeddings.max(),
        reward=DefaultReward(),
    )

    # Load models based on the new naming convention
    model_paths = {
        "levenshtein_emb": os.path.join(exp_folder, "model_with_cosine_levensth.zip"),
        "levenshtein_nonemb": os.path.join(exp_folder, "model_with_exact_levensth.zip"),
        "abid_zou": os.path.join(exp_folder, "model_with_abid_zou.zip"),
    }

    models = {}
    for model_name, model_path in model_paths.items():
        if os.path.exists(model_path):
            models[model_name] = MaskablePPO.load(model_path, env=env)

    if not models:
        raise FileNotFoundError("No model files found in the experiment folder.")

    # Group trajectories by model
    groups_it = groupby_model(model_trajectories)  # pyright: ignore[reportArgumentType]

    unpredictable = 0

    for group, group_name in groups_it:
        # Select checkpoint to evaluate
        try:
            latest_checkpoint = next(filter(lambda t: "aftertraining" in t, group))
        except StopIteration:
            continue

        if evaluate_all_checkpoints:
            checkpoints_to_evaluate = group
        else:
            checkpoints_to_evaluate = [latest_checkpoint]

        for checkpoint_file in checkpoints_to_evaluate:
            with open(os.path.join(exp_folder, checkpoint_file), "rb") as f:
                agent_trajectories = pickle.load(f)

            # Determine which model to use for predictions
            agent_model = None
            model_label = None

            if "levenshtein_emb" in group_name and "levenshtein_emb" in models:
                agent_model = models["levenshtein_emb"]
                model_label = "cosine_lev_agent"
            elif "levenshtein_nonemb" in group_name and "levenshtein_nonemb" in models:
                agent_model = models["levenshtein_nonemb"]
                model_label = "tradlev_agent"
            elif "abid_zou" in group_name and "abid_zou" in models:
                agent_model = models["abid_zou"]
                model_label = "abid_zou_agent"

            if agent_model is None:
                continue

            for traj in tqdm(
                target_group["trajectory_id"], desc=f"Evaluating {group_name}", leave=False
            ):  # pyright: ignore[reportCallIssue, reportArgumentType]
                for j in range(1, len(traj) - 1):
                    env.reset()
                    env.trajectory = traj[:j]
                    env.agent_location = env.trajectory[-1]
                    obs = env._get_obs()
                    action_masks = get_action_masks(env)
                    action_agent = int(agent_model.predict(obs, action_masks=action_masks)[0])

                    action_mc = landmark_mc.get_prediction(np.array([traj[j - 1]]))

                    if action_mc[0] == -1:
                        unpredictable += 1
                        continue

                    target = int(traj[j])  # target is `j`, since we start from `:j`, which does not include `j` itself
                    target_emb = graph.vs[target]["embedding"]
                    agent_pred_emb = graph.vs[action_agent]["embedding"]

                    agent_cossim = cosine_similarity([target_emb], [agent_pred_emb]).item()

                    predictions.append((model_label, int(action_agent), target, j, agent_cossim))

    return pd.DataFrame(predictions, columns=["model", "prediction", "truth", "past_steps", "cossim"]), unpredictable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate "next-click prediction" capabilities of learned models.')
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
        default="test",
    )
    # The line below is because of this issue during training https://github.com/DLR-RM/stable-baselines3/issues/1596
    # Feel free to remove if/when that issue is fixed (notice, github's issue is already closed at the time of writing, but not fixed)
    Distribution.set_default_validate_args(False)
    args = parser.parse_args()

    if not os.path.exists(args.exp_folder):
        print(f"{args.exp_folder} does not exist")
        exit(1)

    with open(os.path.join(args.exp_folder, "graph_with_embeddings.pkl"), "rb") as f:
        graph = pickle.load(f)

    experiment_directories = list(filter(lambda d: "experiment_" in d, os.listdir(args.exp_folder)))
    df = pd.DataFrame()

    unpredictable = 0
    for experiment_dir in tqdm(experiment_directories, desc="Collecting experiment data..."):
        temp_df, unp = evaluate_nextclick_preds(
            os.path.join(args.exp_folder, experiment_dir),
            graph,
            target_group=args.target_group,
            evaluate_all_checkpoints=args.evaluate_all_checkpoints,
        )
        temp_df["experiment_name"] = experiment_dir
        df = pd.concat((df, temp_df), ignore_index=True)
        unpredictable += unp

    # Rename models for consistency with the second script
    df["model"] = df["model"].replace(
        {
            "cosine_lev_agent": "Cos-Levensht.",
            "tradlev_agent": "Trad-Levensht.",
            "abid_zou_agent": "Abid&Zou",
            "markov_chain": "Landmark MC",
            "random_walk": "Random Walk",
        }
    )

    avg_results = duckdb.sql(
        "SELECT model, concat(avg(cossim), '+/-', stddev(cossim)) AS 'Avg. Cos. sim.' FROM df GROUP BY model order by model"
    ).df()
    avg_per_exp = duckdb.sql(
        "SELECT experiment_name, model, concat(avg(cossim), '+/-', stddev(cossim)) AS 'Avg. Cos. sim.' FROM df GROUP BY experiment_name, model order by experiment_name, model"
    ).df()
    print(f"Num unpredictable: {unpredictable}")

    next_click_pred = duckdb.sql(
        """
     SELECT model,
            COUNT(CASE WHEN prediction = truth THEN 1 END) AS matches,
            COUNT(*) AS total_predictions,
            ROUND(100.0 * COUNT(CASE WHEN prediction = truth THEN 1 END) / COUNT(*), 2) AS match_percentage
     FROM df
     GROUP BY model ORDER BY model;
     """
    ).df()

    next_click_pred_after_1 = duckdb.sql(
        """
     SELECT model,
            COUNT(CASE WHEN prediction = truth THEN 1 END) AS matches,
            COUNT(*) AS total_predictions,
            ROUND(100.0 * COUNT(CASE WHEN prediction = truth THEN 1 END) / COUNT(*), 2) AS match_percentage
     FROM df
     WHERE past_steps > 0
     GROUP BY model ORDER BY model;
     """
    ).df()

    next_click_pred_after_2 = duckdb.sql(
        """
     SELECT model,
            COUNT(CASE WHEN prediction = truth THEN 1 END) AS matches,
            COUNT(*) AS total_predictions,
            ROUND(100.0 * COUNT(CASE WHEN prediction = truth THEN 1 END) / COUNT(*), 2) AS match_percentage
     FROM df
     WHERE past_steps > 1
     GROUP BY model ORDER BY model;
     """
    ).df()

    next_click_pred_after_3 = duckdb.sql(
        """
     SELECT model,
            COUNT(CASE WHEN prediction = truth THEN 1 END) AS matches,
            COUNT(*) AS total_predictions,
            ROUND(100.0 * COUNT(CASE WHEN prediction = truth THEN 1 END) / COUNT(*), 2) AS match_percentage
     FROM df
     WHERE past_steps > 2
     GROUP BY model ORDER BY model;
     """
    ).df()

    path = Path(args.exp_folder).parent / "parameters.json"
    with open(path, "r") as f:
        params = json.load(f)
    experiment_hash = utils.get_params_hash_str(params)
    results_path = "evaluation/next_click"
    plot_path = "evaluation/plots"
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    with open(os.path.join(results_path, f"next_click_pred_{experiment_hash}.tex"), "w") as f:
        f.write(next_click_pred.to_latex(index=False))
    with open(os.path.join(results_path, f"next_click_pred_after_1_{experiment_hash}.tex"), "w") as f:
        f.write(next_click_pred_after_1.to_latex(index=False))
    with open(os.path.join(results_path, f"next_click_pred_after_2_{experiment_hash}.tex"), "w") as f:
        f.write(next_click_pred_after_2.to_latex(index=False))
    with open(os.path.join(results_path, f"next_click_pred_after_3_{experiment_hash}.tex"), "w") as f:
        f.write(next_click_pred_after_3.to_latex(index=False))
    with open(os.path.join(results_path, f"next_click_pred_avg_results_{experiment_hash}.tex"), "w") as f:
        f.write(avg_results.to_latex(index=False))

    cluster_technique = "Close" if params["select_cluster_closest_to_num_trajectories"] else "Rand"
    model_order = ["Abid&Zou", "Cos-Levensht.", "Trad-Levensht.", "Landmark MC", "Random Walk"]

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="model", y="cossim", order=model_order)
    plt.title(
        f"Next-Click Prediction Cosine Similarity by Model. Total number of traj. {params['num_trajectories']}, Clustering: {cluster_technique}"
    )
    plt.xlabel("Model")
    plt.ylabel("Cosine similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, f"next_click_pred_cossim_boxplot_{experiment_hash}.png"))

    statistical_results = perform_statistical_tests(df)
    save_statistical_results(statistical_results, results_path, experiment_hash)
