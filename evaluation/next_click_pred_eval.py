import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import Literal

import duckdb
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sklearn.metrics.pairwise import cosine_similarity
from torch.distributions import Distribution
from tqdm import tqdm

from evaluating_trajectories.baselines import bernhard_et_al_2016
from evaluating_trajectories.environment.rewards import DefaultReward
from evaluating_trajectories.environment.website_env import WebsiteEnvironment
from evaluating_trajectories.utils import utils

agent_traj_pattern = re.compile(r"(levenshtein|abid_zou)_?(\w+)?_trajectories_(.+)\.pkl")


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
    groups_it = groupby_model(model_trajectories)

    predictions = []
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

            for traj in tqdm(target_group["trajectory_id"], desc=f"Evaluating {group_name}", leave=False):
                for j in range(len(traj) - 1):
                    env.reset()
                    env.trajectory = traj[: j + 1]
                    env.agent_location = env.trajectory[-1]
                    obs = env._get_obs()
                    action_masks = get_action_masks(env)
                    action_agent = int(agent_model.predict(obs, action_masks=action_masks)[0])

                    action_mc = landmark_mc.get_prediction(np.array([traj[j]]))

                    if action_mc[0] == -1:
                        unpredictable += 1
                        continue

                    target = int(traj[j + 1])
                    target_emb = graph.vs[target]["embedding"]
                    agent_pred_emb = graph.vs[action_agent]["embedding"]
                    mc_pred_emb = graph.vs[action_mc[0]]["embedding"]

                    agent_cossim = cosine_similarity([target_emb], [agent_pred_emb]).item()
                    mc_cossim = cosine_similarity([target_emb], [mc_pred_emb]).item()

                    predictions.append((model_label, int(action_agent), target, j, agent_cossim))
                    predictions.append(("markov_chain", int(action_mc[0]), target, j, mc_cossim))

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
    model_order = ["Abid&Zou", "Cos-Levensht.", "Trad-Levensht.", "Landmark MC"]

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="model", y="cossim", order=model_order)
    plt.title(
        f"Next-Click Prediction Cosine Similarity by Model. Total number of traj. {params['num_trajectories']}, Clustering: {cluster_technique}"
    )
    plt.xlabel("Model")
    plt.ylabel("Cosine similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, f"next_click_pred_cossim_boxplot_{experiment_hash}.png"))
