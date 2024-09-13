import glob
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np

from evaluating_trajectories.evaluation.wasserstein_distance import wasserstein_uniform

experiment_name = "single_traj"
hash = "e1e6d044"
output_folder = f"experiments/{experiment_name}/trajectories_{hash}"

trajectories_expert = {}

pattern = re.compile(f"({output_folder}/run_([0-9\\.]+))/trajectories_([0-9]+)\\.pkl")

distances = {}
for path in glob.glob(f"{output_folder}/run_*/trajectories_*.pkl"):
    match = pattern.fullmatch(path)
    if match is None:
        print(path)
        continue
    run_id = match.groups()[1]
    if run_id not in trajectories_expert:
        with open(f"{match.groups()[0]}/trajectories_expert.pkl", "rb") as f:
            trajectories = np.stack(pickle.load(f), axis=0)
        trajectories_expert[run_id] = trajectories.reshape(trajectories.shape[0], -1)

    with open(path, "rb") as f:
        trajectories_iqlearn = np.stack(pickle.load(f), axis=0)
    # print(trajectories_iqlearn)
    trajectories_iqlearn = trajectories_iqlearn.reshape(
        trajectories_iqlearn.shape[0], -1
    )
    wasserstein = wasserstein_uniform(trajectories_expert[run_id], trajectories_iqlearn)
    id = int(match.groups()[2])
    print(f"{run_id}: {id}: {wasserstein}")
    if id not in distances:
        distances[id] = []
    distances[id].append(wasserstein)

xs = list(distances.keys())
ys = [np.mean(distances[key]) for key in xs]
plt.plot(xs, ys)
plt.show()
