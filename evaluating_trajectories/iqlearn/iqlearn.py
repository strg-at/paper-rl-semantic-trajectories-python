# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.nn.utils.rnn import pad_sequence
from torch.types import Number
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.rich import tqdm

from evaluating_trajectories.environment.website_env import WebsiteEnvironment
from evaluating_trajectories.iqlearn.buffer import MaskableReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = "cpu"
    """device to be used"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: object = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    buffer_size: int = int(1e4)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    use_targets: bool = False
    """Whether or not to use target nets"""
    tau: float = 0.0005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    learning_starts: int = 0
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: dict = field(default_factory=lambda: defaultdict(None, {0: 0.6}))
    """Entropy regularization coefficient given as a dict for timetabling. Has to contain key 0, alpha will be set in every key step."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""
    auto_target_entropy: bool = False
    """whether or not to choose the target entropy automatically"""
    target_entropy: float = -2.0
    """The target entropy when not chosen automatically"""


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1),
        )

    def forward(self, x):
        x = self.network(x)
        return x


class IQLearn:
    def __init__(
        self,
        env: WebsiteEnvironment,
        phi: Callable[[torch.Tensor], torch.Tensor] | None = None,
        regularizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
        online_size: int = 0,
        max_batch=256,
        sac_args: Args | dict[str, Any] | None = None,
    ):
        if sac_args is None:
            self.args = Args()
        elif type(sac_args) == dict:
            self.args = Args(**sac_args)
        elif type(sac_args) == Args:
            self.args = sac_args

        self.embeddings = torch.tensor(env.embeddings).to(self.args.device)
        print(torch.cuda.memory_allocated(0) / 1000000000)
        self.online_size = online_size
        self.demonstration_buffer: MaskableReplayBuffer | None = None
        self.max_batch = max_batch

        if phi is None:
            self.phi = lambda x: x
        else:
            self.phi = phi
        if regularizer is None:
            self.regularizer = lambda x: x**2 / 40
        else:
            self.regularizer = regularizer

        run_name = f"{env.spec.id if env.spec is not None else ''}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"
        if self.args.track:
            import wandb

            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                sync_tensorboard=True,
                config=vars(self.args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        self.writer = SummaryWriter(f"website_log/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % (
                "\n".join(
                    [f"|{key}|{value}|" for key, value in vars(self.args).items()]
                )
            ),
        )

        # TRY NOT TO MODIFY: seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

        self.env = env
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)  # type: ignore

        assert isinstance(
            self.env.observation_space, gym.spaces.Box
        ), "only continuous observation space is supported"

        input_dim = (
            np.prod(self.env.observation_space.shape) + self.env.embeddings.shape[1]
        )
        self.qf1 = SoftQNetwork(input_dim).to(self.args.device)
        self.qf2 = SoftQNetwork(input_dim).to(self.args.device)
        if self.args.use_targets:
            self.qf1_target = SoftQNetwork(input_dim).to(self.args.device)
            self.qf2_target = SoftQNetwork(input_dim).to(self.args.device)
            self.qf1_target.load_state_dict(self.qf1.state_dict())
            self.qf2_target.load_state_dict(self.qf2.state_dict())
        else:
            self.qf1_target = self.qf1
            self.qf2_target = self.qf2
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.args.q_lr
        )

        # Automatic entropy tuning
        if self.args.autotune:
            raise NotImplementedError("Autotuning currently not supported")
            if self.args.auto_target_entropy:
                self.target_entropy = -torch.prod(
                    torch.Tensor(self.env.action_space.shape).to(self.args.device)
                ).item()
            else:
                self.target_entropy = self.args.target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.args.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.args.q_lr)
        else:
            self.alpha = self.args.alpha[0]

        self.env.observation_space.dtype = np.float32  # type: ignore
        self.rb = ReplayBuffer(
            self.args.buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.args.device,
            handle_timeout_termination=False,
        )
        self.start_time = time.time()

        self.global_step = 0

    def compute_action(
        self, observation, actions
    ) -> tuple[Number, Number, torch.Tensor]:
        observation = observation.flatten()
        observation_repeated = observation.unsqueeze(0).repeat(len(actions), 1)
        obs_act_pairs = torch.hstack((observation_repeated, actions))
        qf1_a_values = self.qf1(obs_act_pairs)
        qf2_a_values = self.qf2(obs_act_pairs)
        q_values = torch.minimum(qf1_a_values, qf2_a_values).squeeze(1)
        probabilities = F.softmax(q_values, 0)
        sampled_action = torch.distributions.Categorical(probabilities).sample()
        deterministic_action = torch.argmax(q_values)
        return sampled_action.item(), deterministic_action.item(), probabilities

    def compute_values(
        self,
        observations: list[torch.Tensor],
        possible_actions_per_observation: list[torch.Tensor],
        use_target=False,
        add_entropy=True,
        log_entropy: bool = False,
    ):
        qf1 = self.qf1_target if use_target else self.qf1
        qf2 = self.qf2_target if use_target else self.qf2
        obs_act_pairs = []
        for observation, possible_actions in zip(
            observations, possible_actions_per_observation
        ):
            observation = observation.flatten()
            observation_repeated = observation.unsqueeze(0).repeat(
                len(possible_actions), 1
            )
            obs_act_pair = torch.hstack((observation_repeated, possible_actions))
            obs_act_pairs.append(obs_act_pair)
        obs_act_pairs = torch.vstack(obs_act_pairs)
        batch_list = [
            obs_act_pairs[i * self.max_batch : (i + 1) * self.max_batch]
            for i in range(obs_act_pairs.shape[0] // self.max_batch + 1)
        ]
        qf1_outputs = [qf1(batch) for batch in batch_list]
        qf2_outputs = [qf2(batch) for batch in batch_list]
        qf_outputs = [
            torch.minimum(qf1_output, qf2_output)
            for qf1_output, qf2_output in zip(qf1_outputs, qf2_outputs)
        ]
        qf_concat = torch.vstack(qf_outputs).squeeze()
        if use_target:
            qf_concat_policy = qf_concat
        else:
            qf1_outputs = [self.qf1_target(batch) for batch in batch_list]
            qf2_outputs = [self.qf1_target(batch) for batch in batch_list]
            qf_outputs = [
                torch.minimum(qf1_output, qf2_output)
                for qf1_output, qf2_output in zip(qf1_outputs, qf2_outputs)
            ]
            qf_concat_policy = torch.vstack(qf_outputs).squeeze()

        possible_action_lengths = torch.tensor(
            [len(a) for a in possible_actions_per_observation], device=self.args.device
        )
        cumsum_actions = torch.cat(
            (
                torch.tensor([0], device=self.args.device),
                torch.cumsum(
                    possible_action_lengths,
                    dim=0,
                ),
            ),
        )
        qf_split = [
            qf_concat_policy[cumsum_actions[i] : cumsum_actions[i + 1]]
            for i in range(len(cumsum_actions) - 1)
        ]
        padded_qfs = pad_sequence(
            qf_split, batch_first=True, padding_value=torch.finfo(torch.float32).min
        )
        probabilities = torch.softmax(padded_qfs, dim=1)
        entropy = (-probabilities * (torch.log(probabilities + 1e-20))).sum(dim=1)
        # samples = torch.distributions.Categorical(probs=probabilities).sample()
        # # print(samples[-1])
        # # print(cumsum_actions[:-1][-1])
        # # print(qf_split[-1])
        # # print(probabilities[-1])
        # samples = torch.minimum(
        #     samples, possible_action_lengths - 1
        # )  # impossible actions are still getting selected veeeeery infrequently, so make sure no error is thrown
        # qf_indices = cumsum_actions[:-1] + samples
        # print(qf_indices)
        if log_entropy:
            self.writer.add_scalar(
                "charts/entropy",
                entropy.mean().detach().cpu().item(),
                self.global_step,
            )
        soft_q_values = torch.hstack(
            [
                torch.sum(
                    qf_concat_policy[cumsum_actions[i] : cumsum_actions[i + 1]]
                    * probabilities[i][: possible_action_lengths[i]]
                )
                for i in range(len(cumsum_actions) - 1)
            ]
        )
        soft_q_values += self.alpha * entropy
        return soft_q_values

    def set_demonstration_buffer(self, demonstration_buffer: MaskableReplayBuffer):
        self.demonstration_buffer = demonstration_buffer

    def learn(self, timesteps: int):
        self.qf1.train()
        self.qf2.train()
        # ALGO LOGIC: put action logic here
        assert (
            self.demonstration_buffer is not None
        ), "Demonstration Buffer has to be set!"
        if self.online_size > 0:
            obs, _ = self.env.reset(seed=self.args.seed)

        for _ in tqdm(range(timesteps)):
            self.global_step += 1
            if self.global_step in self.args.alpha:
                self.alpha = self.args.alpha[self.global_step]
            if self.online_size > 0:
                if self.global_step < self.args.learning_starts:
                    action = np.array(self.env.action_space.sample())
                else:
                    action, _, _ = self.actor.get_action(  # type: ignore
                        torch.Tensor(obs).unsqueeze(0).to(self.args.device)  # type: ignore
                    )
                    action = action.detach().cpu().numpy()[0]

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, termination, _, info = self.env.step(action)  # type: ignore

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                if "final_info" in info:
                    for info in info["final_info"]:
                        # print(
                        #     f"self.global_step={self.global_step}, episodic_return={info['episode']['r']}"
                        # )
                        self.writer.add_scalar(
                            "charts/episodic_return",
                            info["episode"]["r"],
                            self.global_step,
                        )
                        self.writer.add_scalar(
                            "charts/episodic_length",
                            info["episode"]["l"],
                            self.global_step,
                        )
                        break

                # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
                self.rb.add(obs, next_obs, action, reward, termination, info)  # type: ignore

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs

            # ALGO LOGIC: training.
            if self.global_step > self.args.learning_starts:
                data = self.demonstration_buffer.sample(self.args.batch_size)

                loss, demonstration_loss, mixed_loss, regularizer_loss = (
                    self.update_critic(data)
                )
                # if self.global_step % self.args.policy_frequency == 0:  # TD 3 Delayed update support
                #     actor_loss, alpha_loss = self.update_policy(data)
                actor_loss = torch.tensor(0)
                alpha_loss = torch.tensor(0)

                # update the target networks
                if (
                    self.global_step % self.args.target_network_frequency == 0
                    and self.args.use_targets
                ):
                    for param, target_param in zip(
                        self.qf1.parameters(), self.qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.args.tau * param.data
                            + (1 - self.args.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.qf2.parameters(), self.qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.args.tau * param.data
                            + (1 - self.args.tau) * target_param.data
                        )

                if self.global_step % 1 == 0:
                    self.writer.add_scalar(
                        "losses/critic_loss",
                        loss.item(),
                        self.global_step,
                    )
                    self.writer.add_scalar(
                        "losses/demonstration_loss",
                        demonstration_loss.item(),
                        self.global_step,
                    )
                    self.writer.add_scalar(
                        "losses/mixed_loss", mixed_loss.item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "losses/regularizer_loss",
                        regularizer_loss.item(),
                        self.global_step,
                    )
                    self.writer.add_scalar("losses/actor_loss", actor_loss.item(), self.global_step)  # type: ignore
                    self.writer.add_scalar("losses/alpha", self.alpha, self.global_step)
                    # print(
                    #     "SPS:", int(self.global_step / (time.time() - self.start_time))
                    # )
                    self.writer.add_scalar(
                        "charts/SPS",
                        int(self.global_step / (time.time() - self.start_time)),
                        self.global_step,
                    )
                    if self.args.autotune:
                        self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), self.global_step)  # type: ignore

    def get_values(
        self,
        observations,
        actions,
        use_target=False,
    ):
        qf1 = self.qf1_target if use_target else self.qf1
        qf2 = self.qf2_target if use_target else self.qf2
        observations = observations.flatten(start_dim=1, end_dim=2)
        obs_actions = torch.concatenate((observations, actions), dim=1)
        qf1_a_values = qf1(obs_actions).view(-1)
        qf2_a_values = qf2(obs_actions).view(-1)
        qf_values = torch.min(qf1_a_values, qf2_a_values)
        return qf_values

    def update_critic(self, data, live_data=None):
        actions = self.embeddings[data.actions.squeeze()]
        possible_actions = [
            self.embeddings[action_mask.type(torch.bool)]
            for action_mask in data.action_masks
        ]
        next_possible_actions = [
            self.embeddings[action_mask.type(torch.bool)]
            for action_mask in data.next_action_masks
        ]
        demonstration_loss = self.get_values(data.observations, actions) - (
            1 - data.dones
        ) * self.args.gamma * self.compute_values(
            data.next_observations, next_possible_actions, True  # type: ignore
        )
        mixed_loss = self.compute_values(
            data.observations, possible_actions, add_entropy=False, log_entropy=True  # type: ignore
        ) - (1 - data.dones) * self.args.gamma * self.compute_values(
            data.next_observations, next_possible_actions, True  # type:ignore
        )
        if live_data is not None:
            raise NotImplementedError()
            live_loss = (
                self.get_values(live_data.observations)
                - (1 - live_data.dones)
                * self.args.gamma
                * self.get_values(live_data.next_observations).detach()
            )
        else:
            live_loss = []  # hack so live_loss has len()

        data_normalizer = (len(mixed_loss) + len(live_loss)) / (len(demonstration_loss))

        regularizer_loss = self.regularizer(mixed_loss).mean()
        if live_data is not None:
            regularizer_loss += self.regularizer(live_loss).mean()  # type: ignore
        regularizer_loss += self.regularizer(demonstration_loss).mean()

        demonstration_loss = self.phi(demonstration_loss).mean()
        mixed_loss = (
            mixed_loss.mean() + (0 if live_data is None else live_loss.mean())  # type: ignore
        ) / data_normalizer
        regularizer_loss = regularizer_loss.mean()

        loss = demonstration_loss - mixed_loss - regularizer_loss
        loss = -loss  # maximize

        # optimize the model
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        return loss, demonstration_loss, mixed_loss, regularizer_loss

    def update_policy(self, data):
        actor_loss = torch.tensor(0.0)
        alpha_loss = torch.tensor(0.0)
        for _ in range(
            self.args.policy_frequency
        ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
            pi, log_pi, _ = self.actor.get_action(data.observations)
            qf1_pi = self.qf1(data.observations, pi)
            qf2_pi = self.qf2(data.observations, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.args.autotune:
                with torch.no_grad():
                    _, log_pi, _ = self.actor.get_action(data.observations)
                alpha_loss = (
                    -self.log_alpha.exp() * (log_pi + self.target_entropy)
                ).mean()

                self.a_optimizer.zero_grad()
                alpha_loss.backward()
                self.a_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
        return actor_loss, alpha_loss

    def predict(self, obs: torch.Tensor | np.ndarray, deterministic: bool = False):
        self.qf1.eval()
        self.qf2.eval()
        if type(obs) == np.ndarray:
            obs = torch.tensor(obs, dtype=torch.float32, device=self.args.device)
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)  # type: ignore
        possible_actions = self.env.map_action_ids_to_embeddings(
            self.env.valid_actions()
        )
        possible_actions = torch.tensor(possible_actions, device=self.args.device)
        action, mean, probabilities = self.compute_action(obs, possible_actions)
        prediction = mean if deterministic else action

        return self.env.valid_actions()[prediction], probabilities  # type: ignore

    def close(self):
        self.env.close()
        self.writer.close()
