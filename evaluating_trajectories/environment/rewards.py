from abc import ABC, abstractmethod
from typing import Literal, Sequence
import numpy.typing as npt
import numpy as np

from evaluating_trajectories.distances.levenshtein_distance import (
    levenshtein_distance,
    cos_dist,
    DistanceFn,
)


class RewardClass(ABC):
    @abstractmethod
    def compute_reward(self, trajectory: npt.NDArray[np.float32]) -> float: ...


class DefaultReward(RewardClass):
    def compute_reward(self, trajectory: npt.NDArray[np.float32]) -> float:
        return 0


class LevenshteinReward(RewardClass):
    def __init__(
        self,
        group_trajectories: Sequence[npt.ArrayLike],
        reduction: Literal["min", "max", "avg"] = "avg",
        distance: DistanceFn = cos_dist,
        penalty: int = 2,
    ):
        self.group_trajectories = group_trajectories
        assert reduction in ["min", "max", "avg"], f"Invalid reduction method: {reduction}"
        self.reduction = reduction
        self.distance = distance
        self.penalty = penalty

    def compute_reward(self, trajectory: npt.NDArray[np.float32] | Sequence[int]) -> float:
        distances = np.zeros((len(self.group_trajectories),))
        if isinstance(trajectory[0], int) or isinstance(trajectory[0], np.int_):
            # sometimes we might receive a list of mixed types: python ints and numpy ints.
            # Numba really doesn't like this, so we make sure that types are consistent
            trajectory = list(map(int, trajectory))
        for i, user_traj in enumerate(self.group_trajectories):
            distances[i] = 1 - levenshtein_distance(user_traj, trajectory, self.distance, penalty=self.penalty)[0]  # type: ignore

        match self.reduction:
            case "min":
                return np.min(distances).item()
            case "max":
                return np.max(distances).item()
            case "avg":
                return np.mean(distances).item()
            case _:
                raise ValueError(f"Invalid reduction method: {self.reduction}")
