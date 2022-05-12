# Library Imports
import numpy as np
import os

from gym import Env

import torch
from torch.nn import functional as F

from replay_buffers.PER import PrioritizedReplayBuffer
from replay_buffers.utils import LinearSchedule

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Critic(torch.nn.Module):
    """Defines a Critic Deep Learning Network"""

    def __init__(self, input_dim: int, beta: float, density: int = 512, name: str = 'critic'):
        super(Critic, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name

        # Architecture
        self.H1 = torch.nn.Linear(input_dim, density, dtype=torch.float32)
        self.H2 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.drop = torch.nn.Dropout(p=0.1)
        self.H3 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.H4 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.Q = torch.nn.Linear(density, 1, dtype=torch.float32)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = device
        self.to(device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        value = torch.hstack((state, action))
        value = F.relu(self.H1(value))
        value = F.relu(self.H2(value))
        value = self.drop(value)
        value = F.relu(self.H3(value))
        value = F.relu(self.H4(value))
        value = self.Q(value)
        return value

    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join(path, self.checkpoint) + '.pth')

    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, self.checkpoint) + '.pth'))


class Actor(torch.nn.Module):
    """Defines a Actor Deep Learning Network"""

    def __init__(self, input_dim: int, n_actions: int, alpha: float, density: int = 512, name='actor'):
        super(Actor, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name

        # Architecture
        self.H1 = torch.nn.Linear(input_dim, density, dtype=torch.float32)
        self.H2 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.drop = torch.nn.Dropout(p=0.1)
        self.H3 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.H4 = torch.nn.Linear(density, density, dtype=torch.float32)
        self.mu = torch.nn.Linear(density, n_actions, dtype=torch.float32)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        value = F.relu(self.H1(state))
        value = F.relu(self.H2(value))
        value = self.drop(value)
        value = F.relu(self.H3(value))
        value = F.relu(self.H4(value))
        action = torch.tanh(self.mu(value))
        return action

    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join(path, self.checkpoint) + '.pth')

    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, self.checkpoint) + '.pth'))


class Agent:
    def __init__(self,
                 env: Env,
                 training: bool = True,
                 alpha=1e-4,
                 beta=1e-3,
                 gamma=0.99,
                 tau=0.005,
                 batch_size: int = 256,
                 noise: float = 0.01,
                 n_games: int = 250):

        self.gamma = torch.tensor(gamma, dtype=torch.float32, device=device)
        self.tau = tau
        self.n_games = n_games
        self.env = env

        self.max_size: int = (env._max_episode_steps * n_games * 2)
        self.memory = PrioritizedReplayBuffer(self.max_size, 0.6)
        self.beta_scheduler = LinearSchedule(n_games, 0.4, 0.9)
        self.batch_size = batch_size

        # Read dimension of states
        OBS = env.observation_space.sample()
        state, actual_goal, desired_desgoal = OBS.values()
        state_vector = np.concatenate((state, actual_goal, desired_desgoal))

        self.obs_shape: int = state_vector.shape[0]
        self.n_actions: int = env.action_space.shape[0]

        self.max_action = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)
        self.min_action = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)

        self.actor = Actor(self.obs_shape, self.n_actions, alpha, name='actor')
        self.critic = Critic(self.obs_shape + self.n_actions, beta, name='critic')
        self.target_actor = Actor(self.obs_shape, self.n_actions, alpha, name='target_actor')
        self.target_critic = Critic(self.obs_shape + self.n_actions, beta, name='target_critic')

        self.is_training = training
        self.noise = noise

        self.per_step: int = 0

        self._update_networks()

    def _update_networks(self, tau: float = 1.0) -> None:

        for critic_weights, target_critic_weights in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_critic_weights.data.copy_(tau * critic_weights.data + (1.0 - tau) * target_critic_weights.data)

        for actor_weights, target_actor_weights in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_actor_weights.data.copy_(tau * actor_weights.data + (1.0 - tau) * target_actor_weights.data)

    def _add_exploration_noise(self, action: torch.Tensor) -> torch.Tensor:
        noise = np.random.uniform(0.0, self.noise, action.shape)
        action += torch.as_tensor(noise, dtype=torch.float32, device=device)

        return action

    def _action_scaling(self, action: torch.Tensor) -> torch.Tensor:
        neural_min = -1.0 * torch.ones_like(action)
        neural_max = 1.0 * torch.ones_like(action)

        env_min = self.min_action * torch.ones_like(action)
        env_max = self.max_action * torch.ones_like(action)

        return ((action - neural_min) / (neural_max - neural_min)) * (env_max - env_min) + env_min

    def choose_action(self, observation: np.ndarray) -> np.ndarray:
        self.actor.eval()

        state = torch.as_tensor(observation, dtype=torch.float32, device=device)
        action = self.actor.forward(state)

        if self.is_training:
            action = self._add_exploration_noise(action)

        action = self._action_scaling(action)

        return action.detach().cpu().numpy()

    def save_models(self, path) -> None:
        self.actor.save_model(path)
        self.target_actor.save_model(path)
        self.critic.save_model(path)
        self.target_critic.save_model(path)

    def load_models(self, path) -> None:
        self.actor.load_model(path)
        self.target_actor.load_model(path)
        self.critic.load_model(path)
        self.target_critic.load_model(path)

    def optimize(self):
        if len(self.memory._storage) < self.batch_size:
            return

        beta = self.beta_scheduler.value(self.per_step)
        state, action, reward, new_state, done, weights, indices = self.memory.sample(self.batch_size, beta)

        state = torch.as_tensor(np.vstack(state), dtype=torch.float32, device=device)
        action = torch.as_tensor(np.vstack(action), dtype=torch.float32, device=device)
        done = torch.as_tensor(np.vstack(1 - done), dtype=torch.float32, device=device)
        reward = torch.as_tensor(np.vstack(reward), dtype=torch.float32, device=device)
        new_state = torch.as_tensor(np.vstack(new_state), dtype=torch.float32, device=device)
        weights = torch.as_tensor(np.hstack(weights), dtype=torch.float32, device=device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.train()
        self.actor.train()

        Q_target = self.target_critic.forward(new_state, self.target_actor.forward(new_state))
        Y = reward + (done * self.gamma * Q_target)
        Q = self.critic.forward(state, action)
        TD_errors = torch.sub(Y, Q).squeeze(dim=-1)

        weighted_TD_Errors = TD_errors * torch.sqrt(weights)
        critic_loss = F.huber_loss(weighted_TD_Errors, torch.zeros_like(weighted_TD_Errors))
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Compute & Update Actor losses
        actor_loss = torch.mean(-1.0 * self.critic.forward(state, self.actor(state)))
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        td_errors: np.ndarray = TD_errors.detach().cpu().numpy()
        new_priorities = np.abs(td_errors) + 1e-6
        self.memory.update_priorities(indices, new_priorities)

        self._update_networks(self.tau)
