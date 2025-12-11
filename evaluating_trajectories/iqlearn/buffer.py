from typing import Any, Dict, List, NamedTuple, Optional, Union

import gymnasium
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize


class MaskedReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    action_masks: th.Tensor
    next_action_masks: th.Tensor


class MaskableReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        assert n_envs == 1, "Currently only non-vectore environments are supported"
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        assert type(action_space) == gymnasium.spaces.Discrete, "Masking is only supported with discrete Action Spaces"
        self.action_masks: np.ndarray = np.zeros((buffer_size, action_space.n))

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        action_mask: np.ndarray | None = None,
    ) -> None:
        if action_mask is not None:
            self.action_masks[self.pos] = action_mask
        else:
            self.action_masks[self.pos] = 1

        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(  # type: ignore
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] | None = None
    ) -> MaskedReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :],
                env,
            )
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.action_masks[batch_inds],
            self.action_masks[(batch_inds + 1) % self.buffer_size],
        )
        return MaskedReplayBufferSamples(*tuple(map(self.to_torch, data)))  # type: ignore
