import pickle

import matplotlib.pyplot as plt
import numpy as np
import ot


def wasserstein(x, y, weights_x, weights_y) -> float:
    M = ot.dist(x, y, metric="cosine")
    return ot.emd2(weights_x, weights_y, M, check_marginals=False)


def wasserstein_uniform(x, y) -> float:
    samples_x = x.shape[0]
    samples_y = y.shape[0]
    weights_x = np.ones((samples_x,)) / samples_x
    weights_y = np.ones((samples_y,)) / samples_y
    return wasserstein(x, y, weights_x, weights_y)


def wasserstein_from_trajectories(trajectories_x, trajectories_y, gamma):
    samples_x, weights_x = samples_weights_from_trajectories(trajectories_x, gamma)
    samples_y, weights_y = samples_weights_from_trajectories(trajectories_y, gamma)
    return wasserstein(samples_x, samples_y, weights_x, weights_y)


def samples_weights_from_trajectories(trajectories, gamma):
    samples = []
    weights = []
    for trajectory in trajectories:
        for i, state in enumerate(trajectory):
            samples.append(state)
            weights.append(gamma**i)
    samples = np.stack(samples, axis=0)
    samples = samples.reshape((samples.shape[0], -1))
    weights = np.array(weights)
    weights /= len(trajectories)
    return samples, weights


if __name__ == "__main__":
    trajectories_x = [
        [np.array([0.1, 0.2]), np.array([0.2, 0.1])],
        [np.array([0.2, 0.1]), np.array([0.1, 0.2])],
    ]
    trajectories_y = [
        [np.array([0.9, 0.2]), np.array([0.2, 0.1])],
        [np.array([0.2, 0.1]), np.array([0.1, 0.2])],
    ]
    gamma = 0.99
    print(wasserstein_from_trajectories(trajectories_x, trajectories_y, gamma))
