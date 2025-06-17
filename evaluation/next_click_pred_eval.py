import argparse
import os
import pickle
import re
from typing import Literal

import duckdb
import igraph as ig
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sklearn.metrics.pairwise import cosine_similarity
from torch.distributions import Distribution
from tqdm import tqdm

from evaluating_trajectories.baselines import bernhard_et_al_2016
from evaluating_trajectories.environment.rewards import DefaultReward
from evaluating_trajectories.environment.website_env import WebsiteEnvironment

agent_traj_pattern = re.compile(r"levenshtein_(\w+)_trajectories_(.+)\.pkl")


def evaluate_nextclick_preds(
    exp_folder: str,
    graph: ig.Graph,
    target_group: Literal["all", "train", "test"] = "all",
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

    if not os.path.exists(os.path.join(exp_folder, "model_with_cosine_levensth.zip")) or not os.path.exists(
        os.path.join(exp_folder, "model_with_exact_levensth.zip")
    ):
        raise FileNotFoundError("The model files are not found in the experiment folder.")

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
    agent_cos_model = MaskablePPO.load(os.path.join(exp_folder, "model_with_cosine_levensth.zip"), env=env)
    agent_tradlev_model = MaskablePPO.load(os.path.join(exp_folder, "model_with_exact_levensth.zip"), env=env)

    predictions = []
    unpredictable = 0
    for traj in tqdm(target_group["trajectory_id"]):
        for j in range(len(traj) - 1):
            env.reset()
            env.trajectory = traj[: j + 1]
            env.agent_location = env.trajectory[-1]
            obs = env._get_obs()
            action_masks = get_action_masks(env)
            action_cos = int(agent_cos_model.predict(obs, action_masks=action_masks)[0])
            action_trad = int(agent_tradlev_model.predict(obs, action_masks=action_masks)[0])

            action_mc = landmark_mc.get_prediction(np.array([traj[j]]))

            if action_mc[0] == -1:
                unpredictable += 1
                continue

            target = int(traj[j + 1])
            target_emb = graph.vs[target]["embedding"]
            agentcos_pred_emb = graph.vs[action_cos]["embedding"]
            agenttrad_pred_emb = graph.vs[action_trad]["embedding"]
            mc_pred_emb = graph.vs[action_mc[0]]["embedding"]

            agentcos_cossim = cosine_similarity([target_emb], [agentcos_pred_emb]).item()
            agenttrad_cossim = cosine_similarity([target_emb], [agenttrad_pred_emb]).item()
            mc_cossim = cosine_similarity([target_emb], [mc_pred_emb]).item()

            predictions.append(("cosine_lev_agent", int(action_cos), target, j, agentcos_cossim))
            predictions.append(("tradlev_agent", int(action_trad), target, j, agenttrad_cossim))
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
    # The line below is because of this issue during training https://github.com/DLR-RM/stable-baselines3/issues/1596
    # Feel free to remove if/when that issue is fixed (notice, github's issue is already closed at the time of writing, but not fixed)
    Distribution.set_default_validate_args(False)
    args = parser.parse_args()
    if not os.path.exists(args.exp_folder):
        print(f"{args.exp_folder} does not exist")
        exit(1)

    with open(os.path.join(args.exp_folder, "graph_with_embeddings.pkl"), "rb") as f:
        graph = pickle.load(f)

    experiment_directories = filter(lambda d: "experiment_" in d, os.listdir(args.exp_folder))
    df = pd.DataFrame()

    unpredictable = 0
    for experiment_dir in tqdm(experiment_directories, desc="Collecting experiment data..."):
        temp_df, unp = evaluate_nextclick_preds(
            os.path.join(args.exp_folder, experiment_dir),
            graph,
            target_group="test",
        )
        temp_df["experiment_name"] = experiment_dir
        df = pd.concat((df, temp_df), ignore_index=True)
        unpredictable += unp

    avg_results = duckdb.sql("SELECT model, avg(cossim) FROM df GROUP BY model order by model")
    avg_per_exp = duckdb.sql(
        "SELECT experiment_name, model, avg(cossim) FROM df GROUP BY experiment_name, model order by experiment_name, model"
    )
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
    )

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
    )

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
    )

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
    )
