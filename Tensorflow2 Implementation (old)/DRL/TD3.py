# Library Imports
import numpy as np
import tensorflow as tf


class ReplayBuffer:
    """Defines the Buffer dataset from which the agent learns"""
    def __init__(self, max_size, input_shape, dim_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, dim_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        _states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, _states, dones


class Critic(tf.keras.Model):
    """Defines a Critic Deep Learning Network"""
    def __init__(self, density=512, name='critic'):
        super(Critic, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name + '.h5'

        self.H1 = tf.keras.layers.Dense(density, activation='relu')
        self.H2 = tf.keras.layers.Dense(density, activation='relu')
        self.drop = tf.keras.layers.Dropout(0.2)
        self.H3 = tf.keras.layers.Dense(density, activation='relu')
        self.H4 = tf.keras.layers.Dense(density, activation='relu')
        self.Q = tf.keras.layers.Dense(1, activation=None)

    @tf.function()
    def call(self, state, action):
        action = self.H1(tf.concat([state, action], axis=1))
        action = self.H2(action)
        action = self.drop(action)
        action = self.H3(action)
        action = self.H4(action)
        Q = self.Q(action)
        return Q


class Actor(tf.keras.Model):
    """Defines a Actor Deep Learning Network"""
    def __init__(self, n_actions, density=512, name='actor'):
        super(Actor, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name + '.h5'

        self.H1 = tf.keras.layers.Dense(density, activation='relu')
        self.H2 = tf.keras.layers.Dense(density, activation='relu')
        self.drop = tf.keras.layers.Dropout(0.2)
        self.H3 = tf.keras.layers.Dense(density, activation='relu')
        self.H4 = tf.keras.layers.Dense(density, activation='relu')
        self.mu = tf.keras.layers.Dense(n_actions, activation='tanh')

    @tf.function()
    def call(self, state):
        state = self.H1(state)
        state = self.H2(state)
        state = self.drop(state)
        state = self.H3(state)
        state = self.H4(state)
        mu = self.mu(state)
        return mu


class Agent:
    """Defines a RL Agent based on Actor-Critc method"""
    def __init__(self, env, alpha=0.0001, beta=0.001,
                 gamma=0.99, max_size=250000, tau=0.005,
                 batch_size=64, noise=0.1):

        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.n_actions = env.action_space.shape[0]
        self.obs_shape = env.observation_space['observation'].shape[0] + \
                         env.observation_space['achieved_goal'].shape[0] + \
                         env.observation_space['desired_goal'].shape[0]
        self.memory = ReplayBuffer(max_size, self.obs_shape, self.n_actions)

        self.batch_size = batch_size
        self.noise = noise
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low

        self.actor = Actor(self.n_actions, name='actor')
        self.critic = Critic(name='critic')
        self.target_actor = Actor(self.n_actions, name='target_actor')
        self.target_critic = Critic(name='target_critic')

        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.update_networks(tau=1)

    def update_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

    def store(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self, path):
        self.actor.save_weights(path + self.actor.checkpoint)
        self.critic.save_weights(path + self.critic.checkpoint)
        self.target_actor.save_weights(path + self.target_actor.checkpoint)
        self.target_critic.save_weights(path + self.target_critic.checkpoint)

    def load_models(self, path):
        self.actor.load_weights(path + self.actor.checkpoint)
        self.critic.load_weights(path + self.critic.checkpoint)
        self.target_actor.load_weights(path + self.target_actor.checkpoint)
        self.target_critic.load_weights(path + self.target_critic.checkpoint)

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.noise)
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        return actions[0]

    def optimize(self, steps):

        for i in range(steps):
            if self.memory.mem_cntr < self.batch_size:
                return

            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

            with tf.GradientTape() as tape:
                target_actions = self.target_actor(new_state)
                critic_value_ = tf.squeeze(self.target_critic(new_state, target_actions), 1)
                critic_value = tf.squeeze(self.critic(state, action), 1)
                target = reward + self.gamma * critic_value_ * (1.0 - done)
                critic_loss = tf.keras.losses.MSE(target, critic_value)

            critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

            with tf.GradientTape() as tape:
                new_policy_actions = self.actor(state)
                actor_loss = -self.critic(state, new_policy_actions)
                actor_loss = tf.math.reduce_mean(actor_loss)

            actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))
            self.update_networks()
