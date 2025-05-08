import gymnasium as gym
import igraph as ig
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from .rewards import DefaultReward, RewardClass


class WebsiteEnvironment(gym.Env):
    """
    Environment to deal with graph extraction from Website data.
    """

    def __init__(
        self,
        graph: ig.Graph,
        starting_locations: list[int],
        max_steps: int,
        embedding_min_val: float,
        embedding_max_val: float,
        reward: RewardClass | None = None,
        render_mode=None,
    ):
        self.render_mode = render_mode
        assert render_mode is None, "Rendering is not currently supported by this environment!"
        self.graph = graph
        self.reward = reward if reward is not None else DefaultReward()

        self.observation_space = spaces.Box(
            low=embedding_min_val,
            high=embedding_max_val,
            shape=(max_steps, graph.vs[0]["embedding"].shape[-1]),
            seed=42,
        )

        self.max_steps = max_steps
        # Actions we can take are:
        # - navigate to a node (we'll mask nodes we can't travel to)
        # - exit
        self.exit_action = len(self.graph.vs) - 1
        self.starting_location = starting_locations
        self.agent_location = np.random.choice(self.starting_location)

        self.trajectory = []
        self.action_space = spaces.Discrete(self.exit_action + 1, seed=42)

    def reset(self, seed=None) -> tuple[npt.NDArray, dict]:  # type:ignore
        """
        Reset method subclassed from ``gymnasium``.
        :return: same as ``gymnasium`` reset.
        """
        super().reset(seed=seed)
        self.agent_location = int(np.random.choice(self.starting_location))
        self.trajectory = [self.agent_location]
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: int) -> tuple[npt.NDArray, float, bool, bool, dict]:
        """
        Step method subclassed from ``gymnasium``. Action should be the node_id the agent wants to travel to.

        :param action: valid ``node_id``. This node should also be reachable from the current location.
        :return: same as ``gymnasium`` reset.
        """

        terminated = action == self.exit_action
        self.trajectory.append(action)
        self.agent_location = action

        if len(self.trajectory) == self.max_steps:
            return self._get_obs(), 0, True, False, self._get_info()

        if terminated:
            observation = self._get_obs()
            return (
                observation,
                self.reward.compute_reward(observation),
                terminated,
                False,
                self._get_info(),
            )

        observation = self._get_obs()
        info = self._get_info()
        return (
            observation,
            self.reward.compute_reward(observation),
            terminated,
            False,
            info,
        )

    def valid_actions(self) -> list[int]:
        """
        Returns the list of valid actions given the current location of the agent on the graph.
        """
        return [self.exit_action] + self.neighbors()

    def map_action_ids_to_embeddings(self, action_ids: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """
        Converts a given list of action ids to a numpy array containing the embeddings of the given actions
        """
        embs = self.graph.vs[action_ids]["embedding"]
        return np.array(embs)

    def neighbors(self) -> list[int]:
        """
        Returns the list of neighbors reachable from the current agent location with at most 1 step.
        """
        return self.graph.neighbors(self.agent_location)

    def action_masks(self) -> npt.NDArray[np.bool_]:
        """
        Implements the ``action_masks`` interface from sb3. See `here <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/envs/invalid_actions_env.py>`_ and `here <https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html>`_
        """
        mask = np.zeros(self.action_space.n, dtype=bool)  # type:ignore
        mask[self.valid_actions()] = True
        return mask

    def render(self): ...

    def _get_obs(self) -> npt.NDArray[np.float32]:
        """
        Returns the observation for the agent. Information (e.g., embeddings) on nodes that
        are at distant > 1 from current location are masked.

        :return: observation for the agent.
        """
        # mask out embeddings of nodes that cannot be reached in one step
        padded_trajectory = np.full(self.max_steps, len(self.graph.vs) - 1, dtype=np.int32)
        padded_trajectory[: len(self.trajectory)] = self.trajectory

        return self.map_action_ids_to_embeddings(padded_trajectory)

    def _get_info(self) -> dict:
        return {}
