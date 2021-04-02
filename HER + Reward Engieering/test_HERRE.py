import gym
import sys
sys.path.append('HER + Reward Engieering')
from ddpg import Agent
import tensorflow as tf
import numpy as np

def test(env, n_games):
    # Initiate the agent
    desiredgoal_shape =  env.observation_space['desired_goal'].shape[0]
    achievedgoal_shape =  env.observation_space['achieved_goal'].shape[0]
    observation_shape =  env.observation_space['observation'].shape[0]
    action_shape = env.action_space.shape[0]
    state_shape = observation_shape + desiredgoal_shape + achievedgoal_shape
    agent = Agent(input_dims= state_shape, env=env, n_actions= action_shape)
    
    state = env.reset().values()
    obs, curr_pos, goal_pos = state
    agent.choose_action(np.concatenate((obs, curr_pos, goal_pos)))
    agent.actor.load_weights('HER + Reward Engieering/actor.h5')

    # Play 10 games
    n_games = n_games
    for i in range(n_games):
        state = env.reset().values()
        obs, curr_pos, goal_pos = state
        done = False
        score = 0
        
        while not done:
            env.render()    
            
            action = agent.choose_action(np.concatenate((obs, curr_pos, goal_pos)))
            next_state, reward, done, info =  env.step(action)
            next_obs, next_curr_pos, goal_pos = next_state.values()
            
            curr_pos = next_curr_pos
            obs = next_obs 
            score += reward
            if done:
                break

        print(f'Episode: {i} \t Avg. Episodic Reward: {score:.4f}')
    env.close()

if __name__ == '__main__':
    # Load the environment
    env = gym.make('FetchReach-v1')
    test(env, 10)


    
