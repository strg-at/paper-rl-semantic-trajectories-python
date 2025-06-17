from abc import ABC, abstractmethod
from typing import Literal, Sequence, override

import numpy as np
import numpy.typing as npt

from evaluating_trajectories.distances.levenshtein_distance import (
    DistanceFn,
    cos_dist,
    levenshtein_distance,
)


class RewardClass(ABC):
    @abstractmethod
    def compute_reward(self, trajectory: npt.NDArray[np.float32]) -> float: ...

    @abstractmethod
    def reset(self): ...


class DefaultReward(RewardClass):
    def compute_reward(self, trajectory: npt.NDArray[np.float32]) -> float:
        return 0

    def reset(self): ...


class LevenshteinReward(RewardClass):
    def __init__(
        self,
        group_trajectories: Sequence[npt.ArrayLike],
        reduction: Literal["min", "max", "avg"] = "avg",
        distance: DistanceFn = cos_dist,
        strategy: Literal["plain", "diff", "shift"] = "plain",
        penalty: int = 2,
    ):
        """
        The levenshtein distance is computed between the input trajectory and each trajectory in the group.
        We reduce this list of distances according to the ``reduction`` parameter.
        The strategy parameter affects how the reward is returned:
          - plain: the reduced reward is returned as-is;
          - diff: this class will keep track of the best reward achieved. The reward is then returned as the difference between the previous best and the current reward;
          - shift: this shifts the levenshtein score from [0,1] to [-1, 1], by centering the score at 0.5 ()
        """
        self.group_trajectories = group_trajectories
        assert reduction in ["min", "max", "avg"], f"Invalid reduction method: {reduction}"
        self.reduction = reduction
        self.distance = distance
        self.strategy = strategy
        self.penalty = penalty
        self.best_reward = -1

    @override
    def compute_reward(self, trajectory: npt.NDArray[np.float32] | Sequence[int]) -> float:
        distances = np.zeros((len(self.group_trajectories),))
        if isinstance(trajectory[0], int) or isinstance(trajectory[0], np.integer):
            # sometimes we might receive a list of mixed types: python ints and numpy ints.
            # Numba really doesn't like this, so we make sure that types are consistent
            trajectory = list(map(int, trajectory))
        for i, user_traj in enumerate(self.group_trajectories):
            distances[i] = 1 - levenshtein_distance(user_traj, trajectory, self.distance, penalty=self.penalty)[0]  # type: ignore

        match self.reduction:
            case "min":
                reward = np.min(distances).item()
            case "max":
                reward = np.max(distances).item()
            case "avg":
                reward = np.mean(distances).item()
            case _:
                raise ValueError(f"Invalid reduction method: {self.reduction}")

        if self.strategy == "diff":
            reward = reward - self.best_reward
            self.best_reward = max(reward, self.best_reward)
        elif self.strategy == "shift":
            reward = (reward - 0.5) * 2
            self.best_reward = max(reward, self.best_reward)
        elif self.strategy == "plain":
            self.best_reward = max(reward, self.best_reward)

        return reward

    @override
    def reset(self):
        self.best_reward = -1
