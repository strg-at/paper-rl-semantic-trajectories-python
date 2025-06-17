import collections
import itertools as itt
from typing import Final

import numpy as np
import numpy.typing as npt
from tqdm import tqdm


def sliding_window(iterable, n):
    "Collect data into overlapping fixed-length chunks or blocks."
    # sliding_window('ABCDEFG', 4) → ABCD BCDE CDEF DEFG
    iterator = iter(iterable)
    window = collections.deque(itt.islice(iterator, n - 1), maxlen=n)
    for x in iterator:
        window.append(x)
        yield tuple(window)


class LandmarkMarkovChain:
    markov_chain: npt.NDArray[np.uint32]

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.exit: Final[int] = num_nodes
        self.entry: Final[int] = num_nodes + 1

    def build_markov_chain(self, trajectories: list[npt.NDArray[np.integer]]):
        # +2 since we need enter and exit node
        self.markov_chain = np.zeros((self.num_nodes + 2, self.num_nodes + 2), dtype=np.uint32)
        for traj in tqdm(trajectories):
            self.markov_chain[self.entry, traj[0]] += 1
            slides = np.array(list(sliding_window(traj, 2)))

            self.markov_chain[slides[:, 0], slides[:, 1]] += 1
            self.markov_chain[traj[-1], self.exit] += 1

    def sample_trajectories(self, starting_nodes: npt.NDArray[np.integer]) -> list[int]:
        assert (
            hasattr(self, "markov_chain") and len(self.markov_chain.nonzero()[0]) > 0
        ), "Markov chain must be built before sampling trajectories."
        sampled_trajectories = []
        rng = np.random.default_rng()
        for start_node in enumerate(starting_nodes):
            next_node = start_node
            trajectory = []
            while next_node != self.exit:
                trajectory.append(next_node)
                transitions = self.markov_chain[next_node] / self.markov_chain[next_node].sum()
                next_node = rng.choice(np.arange(len(self.markov_chain)), size=1, p=transitions)[0]
            sampled_trajectories.append(trajectory)
        return sampled_trajectories

    def make_predictions(self, target_trajectories: list[npt.NDArray[np.integer]]):
        num_correct, num_wrong, num_unpred = 0, 0, 0
        all_predictions = []
        for traj in tqdm(target_trajectories, desc="Markov chain: Making predictions..."):
            predictions = self.get_prediction(traj)
            traj = np.append(traj, self.exit)
            all_predictions.append(predictions)
            num_correct += (predictions == traj[1:]).sum()
            num_unpred += (predictions == -1).sum()
            num_wrong += (predictions[predictions != -1] != traj[1:][predictions != -1]).sum()
        return all_predictions, num_correct, num_wrong, num_unpred

    def get_prediction(self, current_states: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
        unpredictable = np.where(self.markov_chain[current_states].sum(axis=1) == 0)[0]
        predictions = self.markov_chain[current_states].argmax(axis=1)
        predictions[unpredictable] = -1
        return predictions
