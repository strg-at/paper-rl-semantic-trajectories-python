import glob
import json
import pickle
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from numba.typed import List

from evaluating_trajectories.evaluation.levenshtein_distance import (
    euclid_sim,
    mean_levenshtein_distance,
)
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


def convert_to_list(trajectories):
    return List(
        List(trajectory[:, i] for i in range(trajectory.shape[1]))
        for trajectory in trajectories
    )


fig, axs = plt.subplots(2, 1, figsize=(12, 6))

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

    wasserstein_distances = {}
    levenshtein_distances = {}
    print_traj = None
    for path in glob.glob(f"{output_folder}/run_*/trajectories_*.pkl"):
        match = pattern.fullmatch(path)
        if match is None:
            print(path)
            continue
        run_id = match.groups()[1]
        if run_id not in trajectories_expert:
            with open(f"{match.groups()[0]}/trajectories_expert.pkl", "rb") as f:
                trajectories_expert[run_id] = np.stack(pickle.load(f), axis=0)

        with open(path, "rb") as f:
            trajectories_iqlearn = np.stack(pickle.load(f), axis=0)
        # print(trajectories_iqlearn)
        print(f"expert: {trajectories_expert[run_id][0, :, 0]}")
        print(trajectories_iqlearn[0, :, 0])
        wasserstein = wasserstein_uniform(
            trajectories_expert[run_id].reshape(
                trajectories_expert[run_id].shape[0], -1
            ),
            trajectories_iqlearn.reshape(trajectories_iqlearn.shape[0], -1),
        )
        levenshtein = mean_levenshtein_distance(
            convert_to_list(trajectories_expert[run_id]),
            convert_to_list(trajectories_iqlearn),
            euclid_sim,
        )
        id = int(match.groups()[2])
        print(f"{run_id}: {id}: {wasserstein}")
        if id not in wasserstein_distances:
            wasserstein_distances[id] = []
        wasserstein_distances[id].append(wasserstein)
        if id not in levenshtein_distances:
            levenshtein_distances[id] = []
        levenshtein_distances[id].append(levenshtein)

    xs = list(wasserstein_distances.keys())
    xs.sort()
    ys = [np.mean(wasserstein_distances[key]) for key in xs]
    axs[0].plot(xs, ys, label=label)
    xs = list(levenshtein_distances.keys())
    xs.sort()
    ys = [np.mean(levenshtein_distances[key]) for key in xs]
    axs[1].plot(xs, ys, label=label)

axs[0].set_title("Wasserstein Distance over Training Steps")
axs[0].set_xlabel("Training Steps")
axs[0].set_ylabel("Wasserstein Distance")
axs[0].legend()

axs[1].set_title("Levenshtein Distance over Training Steps")
axs[1].set_xlabel("Training Steps")
axs[1].set_ylabel("Levenshtein Distance")
axs[1].legend()
plt.show()
