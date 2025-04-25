from abc import ABC, abstractmethod
from typing import Literal
import numpy.typing as npt
import numpy as np

from evaluating_trajectories.distances.levenshtein_distance import levenshtein_distance, cos_dist


class RewardClass(ABC):
    @abstractmethod
    def compute_reward(self, trajectory: npt.NDArray[np.float32]) -> float: ...


class DefaultReward(RewardClass):
    def compute_reward(self, trajectory: npt.NDArray[np.float32]) -> float:
        return 0


class LevenshteinReward(RewardClass):
    def __init__(self, group_trajectories: list[list[int]], reduction: Literal["min", "max", "avg"] = "avg"):
        self.group_trajectories = group_trajectories
        assert reduction in ["min", "max", "avg"], f"Invalid reduction method: {reduction}"
        self.reduction = reduction

    def compute_reward(self, trajectory: npt.NDArray[np.float32]) -> float:
        trajectory = [obs for obs in trajectory]  # type: ignore
        distances = np.zeros((len(self.group_trajectories),))
        for i, user_traj in enumerate(self.group_trajectories):
            distances[i] = 1 - levenshtein_distance(user_traj, trajectory, cos_dist)[0]  # type: ignore

        match self.reduction:
            case "min":
                return np.min(distances).item()
            case "max":
                return np.max(distances).item()
            case "avg":
                return np.mean(distances).item()
            case _:
                raise ValueError(f"Invalid reduction method: {self.reduction}")
