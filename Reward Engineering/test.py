import gym
from ddpg import Agent
import tensorflow as tf
import numpy as np

# Load the environment and Agent
env = gym.make('FetchReach-v1')

ip_shape =  env.observation_space['desired_goal'].shape[0]
action_shape = env.action_space.shape[0]  
agent = Agent(input_dims= ip_shape, env=env, n_actions= action_shape)

state = env.reset().values()
pure_states, curr_pos, goal_pos = state

agent.choose_action(curr_pos, goal_pos)
agent.actor.load_weights('actor.h5')

score_history = []
n_games = 10

for i in range(n_games):
    state = env.reset().values()
    pure_states, curr_pos, goal_pos = state
    init_pos = curr_pos
    done = False
    score = 0
    
    while not done:
        env.render()    
        
        action = agent.choose_action(curr_pos, goal_pos)
        next_state, reward, done, info =  env.step(action)
        next_pure_states, next_curr_pos, goal_pos = next_state.values()
        
        score += reward
        curr_pos = next_curr_pos 
        if done:
            break
           
    score_history.append(score)
    print(f'Episode: {i} \t Avg. Episodic Reward: {score:.4f}')
env.close()