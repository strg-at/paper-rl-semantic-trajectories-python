import argparse
import os
import pickle
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
            yield group
            yield [m.string for m in mtraj[i:]]
            break


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
    args = parser.parse_args()

    if not os.path.exists(args.exp_folder):
        print(f"{args.exp_folder} does not exist")
        exit(1)

    experiments = os.listdir(args.exp_folder)
    with open(os.path.join(args.exp_folder, "graph_with_embeddings.pkl"), "rb") as f:
        graph = pickle.load(f)

    with open(os.path.join(args.exp_folder, "trajectories_target.pkl"), "rb") as f:
        target_group = pickle.load(f)

    model_trajectories = os.listdir(args.exp_folder)
    model_trajectories = list(filter(lambda p: p is not None, map(agent_traj_pattern.match, model_trajectories)))

    groups_it = groupby_model(model_trajectories)

    groups_results = []
    for group in groups_it:
        latest_checkpoint = next(filter(lambda t: "aftertraining" in t, group))

        if args.evaluate_all_checkpoints:
            checkpoints_to_evaluate = group
        else:
            checkpoints_to_evaluate = [latest_checkpoint]

        for model_file in checkpoints_to_evaluate:
            with open(os.path.join(args.exp_folder, model_file), "rb") as f:
                agent_trajectories = pickle.load(f)
            results = {}
            for i, traj in enumerate(agent_trajectories):
                avg_embs = np.array(graph.vs[traj]["embedding"]).mean(axis=0).reshape(1, -1)
                for j, group in enumerate(target_group["trajectory_emb"]):
                    avg_group = group.mean(axis=0).reshape(1, -1)
                    user_dict = results.setdefault(f"agent_{i}", {})
                    user_dict[f"user_{j}"] = cosine_similarity(avg_embs, avg_group).item()
            groups_results.append(results)
