# Library Imports
import numpy as np
import tensorflow as tf

# DDPG Reinforcement Learning
class ReplayBuffer:
    """Defines the Buffer dataset from which the agent learns"""
    def __init__(self, max_size, input_shape, dim_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.goal_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, dim_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        
    def store_transition(self, state, goal, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.goal_memory[index] = goal
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr +=1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        goal = self.goal_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, goal, actions, rewards, new_states, dones
    
class Critic(tf.keras.Model):
    """Defines a Critic Deep Learning Network"""
    def __init__(self, dim_actions, H1_dim=128, H2_dim=128, name='critic'):
        super(Critic, self).__init__()
        self.H1_dim = H1_dim
        self.H2_dim = H2_dim
        self.dim_actions = dim_actions
        self.model_name = name
        self.checkpoint = self.model_name+'.h5'
        self.H1 = tf.keras.layers.Dense(self.H1_dim, activation='relu')
        self.H2 = tf.keras.layers.Dense(self.H2_dim, activation='relu') 
        self.Q = tf.keras.layers.Dense(1, activation=None)
        
    def call(self, state, goal, action):
        action = self.H1(tf.concat([state, goal, action], axis=1))
        action = self.H2(action)
        Q = self.Q(action)
        return Q
    
class Actor(tf.keras.Model):
    """Defines a Actor Deep Learning Network"""
    def __init__(self, dim_actions, H1_dim=128, H2_dim=128, name='actor'):
        super(Actor, self).__init__()
        self.H1_dim = H1_dim
        self.H2_dim = H2_dim
        self.dim_actions = dim_actions
        self.model_name = name
        self.checkpoint = self.model_name+'.h5'
        self.H1 = tf.keras.layers.Dense(self.H1_dim, activation='relu')
        self.H2 = tf.keras.layers.Dense(self.H2_dim, activation='relu') 
        self.mu = tf.keras.layers.Dense(self.dim_actions, activation='tanh')
        
    def call(self, state, goal):
        action_prob = self.H1(tf.concat([state, goal], axis=1))
        action_prob = self.H2(action_prob)
        mu = self.mu(action_prob)
        return mu
    
class Agent:
    """Defines a RL Agent based on Actor-Critc method"""
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None,
                 gamma=0.99, n_actions=4, max_size=1000000, tau=0.005,
                 H1=125, H2=125, batch_size=128, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.memory = ReplayBuffer(max_size, input_dims, self.n_actions)
        self.batch_size = batch_size
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        
        self.actor  = Actor(self.n_actions, name='actor')
        self.critic = Critic(self.n_actions, name='critic')
        self.target_actor  = Actor(self.n_actions, name='target_actor')
        self.target_critic = Critic(self.n_actions, name='target_critic')
        
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.update_networks(tau=1)
        
    def update_networks(self, tau=None):
        if tau is None:
            tau = self.tau
            
        weights=[]
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)
        
        weights=[]
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)
        
    def recall(self, state, goal, action, reward, new_state, done):
        self.memory.store_transition(state, goal, action, reward, new_state, done)
        
    def save_models(self):
        self.actor.save_weights(self.actor.checkpoint)
        self.critic.save_weights(self.critic.checkpoint)
        self.target_actor.save_weights(self.target_actor.checkpoint)
        self.target_critic.save_weights(self.target_critic.checkpoint)
        
    def load_models(self):
        self.actor.load_weights(self.actor.checkpoint)
        self.critic.load_weights(self.critic.checkpoint)
        self.target_actor.load_weights(self.target_actor.checkpoint)
        self.target_critic.load_weights(self.target_critic.checkpoint)
        
    def choose_action(self, state, goal):
        evaluate=False
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        goal = tf.convert_to_tensor([goal], dtype=tf.float32)
        actions = self.actor(state, goal)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.noise)
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        return actions[0]
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, goal, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        goals = tf.convert_to_tensor(goal, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_, goals)
            critic_value_ = tf.squeeze(self.target_critic(states_, goals, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, goals, actions), 1)
            target = reward + self.gamma*critic_value_*(1-done)
            critic_loss = tf.keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states, goals)
            actor_loss = -self.critic(states, goals, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))
        self.update_networks()