import gym
import sys
sys.path.append('Reward Engineering')
from ddpg import Agent
import tensorflow as tf
import numpy as np

def test(env, n_games):
    # Initiate the agent
    ip_shape =  env.observation_space['desired_goal'].shape[0]
    action_shape = env.action_space.shape[0]  
    agent = Agent(input_dims= ip_shape, env=env, n_actions= action_shape)
    state = env.reset().values()
    pure_states, curr_pos, goal_pos = state
    agent.choose_action(curr_pos, goal_pos)

    # Load the trained agent weights
    agent.actor.load_weights('Reward Engineering/actor.h5')

    # Play 10 games
    n_games = n_games
    for i in range(n_games):
        state = env.reset().values()
        pure_states, curr_pos, goal_pos = state
        init_pos = curr_pos
        done = False
        score = []
        
        while not done:
            env.render()    
            
            action = agent.choose_action(curr_pos, goal_pos)
            next_state, reward, done, info =  env.step(action)
            next_pure_states, next_curr_pos, goal_pos = next_state.values()
            
            score.append(reward)
            curr_pos = next_curr_pos 
            if done:
                break

        print(f'Episode: {i} \t Avg. Episodic Reward: {np.array(score).sum():.4f}')
    env.close()

if __name__ == '__main__':
    # Load the environment
    env = gym.make('FetchReach-v1')
    test(env, 10)


    
