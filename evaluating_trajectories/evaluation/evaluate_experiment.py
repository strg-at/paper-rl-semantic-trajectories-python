import glob
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np

from evaluating_trajectories.evaluation.wasserstein_distance import wasserstein_uniform

trajectories_folder = "trajectories"

with open(f"{trajectories_folder}/trajectories_expert.pkl", "rb") as f:
    trajectories_expert = np.stack(pickle.load(f), axis=0)
trajectories_expert = trajectories_expert.reshape(trajectories_expert.shape[0], -1)

pattern = re.compile(f"{trajectories_folder}/trajectories_([0-9]+)\\.pkl")

distances = {}
for path in glob.glob(f"{trajectories_folder}/trajectories_*.pkl"):
    match = pattern.fullmatch(path)
    if match is None:
        print(path)
        continue
    with open(path, "rb") as f:
        trajectories_iqlearn = np.stack(pickle.load(f), axis=0)
    # print(trajectories_iqlearn)
    trajectories_iqlearn = trajectories_iqlearn.reshape(
        trajectories_iqlearn.shape[0], -1
    )
    wasserstein = wasserstein_uniform(trajectories_expert, trajectories_iqlearn)
    id = int(match.groups()[0])
    print(f"{id}: {wasserstein}")
    distances[id] = wasserstein

xs = list(distances.keys())
ys = [distances[key] for key in xs]
plt.plot(xs, ys)
plt.show()
