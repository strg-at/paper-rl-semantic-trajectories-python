import glob
import json
import pickle
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

from evaluating_trajectories.evaluation.wasserstein_distance import wasserstein_uniform

np.set_printoptions(threshold=sys.maxsize)

experiment_name = "single_traj"

hashes = [
    "eb38b9ab",
    "19993abf",
    "70d1eb80",
    "aa67979e",
    "fe7d3abb",
    "64199be3",
    "81644dcd",
]


def get_label(params: dict):
    return f"$\\alpha$={params['alpha']}"


for hash in hashes:
    output_folder = f"experiments/{experiment_name}/trajectories_{hash}"

    with open(f"{output_folder}/parameters.json", "r") as f:
        params = json.load(f)
    label = get_label(params)  # type:ignore
    print(label)

    trajectories_expert = {}

    pattern = re.compile(
        f"({output_folder}/run_([0-9\\.]+))/trajectories_([0-9]+)\\.pkl"
    )

    distances = {}
    print_traj = None
    for path in glob.glob(f"{output_folder}/run_*/trajectories_*.pkl"):
        match = pattern.fullmatch(path)
        if match is None:
            print(path)
            continue
        run_id = match.groups()[1]
        if run_id not in trajectories_expert:
            with open(f"{match.groups()[0]}/trajectories_expert.pkl", "rb") as f:
                trajectories = np.stack(pickle.load(f), axis=0)
            print_traj = trajectories
            trajectories_expert[run_id] = trajectories.reshape(
                trajectories.shape[0], -1
            )

        with open(path, "rb") as f:
            trajectories_iqlearn = np.stack(pickle.load(f), axis=0)
        # print(trajectories_iqlearn)
        print(f"expert: {print_traj[0, :, 0]}")
        print(trajectories_iqlearn[0, :, 0])
        trajectories_iqlearn = trajectories_iqlearn.reshape(
            trajectories_iqlearn.shape[0], -1
        )
        wasserstein = wasserstein_uniform(
            trajectories_expert[run_id], trajectories_iqlearn
        )
        id = int(match.groups()[2])
        print(f"{run_id}: {id}: {wasserstein}")
        if id not in distances:
            distances[id] = []
        distances[id].append(wasserstein)

    xs = list(distances.keys())
    xs.sort()
    ys = [np.mean(distances[key]) for key in xs]
    plt.plot(xs, ys, label=label)

plt.title("Wasserstein Distance over Training Steps")
plt.xlabel("Training Steps")
plt.ylabel("Wasserstein Distance")
plt.legend()
plt.show()
