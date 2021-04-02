import gym
import numpy as np
from ddpg import Agent

def reward_engg(goal_pos, current_pos, initial_pos):
    max_dist = np.linalg.norm(goal_pos - initial_pos)
    min_dist = 0
    curr_dist = np.linalg.norm(goal_pos - current_pos)
    value = (max_dist - curr_dist)/(max_dist - min_dist)
    reward = (value ) * (1 - (-1)) + (-1)
    return reward

if __name__ == "__main__":
    # Load the environment
    env = gym.make('FetchReach-v1')
    desiredgoal_shape =  env.observation_space['desired_goal'].shape[0]
    achievedgoal_shape =  env.observation_space['achieved_goal'].shape[0]
    observation_shape =  env.observation_space['observation'].shape[0]
    action_shape = env.action_space.shape[0]
    state_shape = observation_shape + achievedgoal_shape + desiredgoal_shape
    
    agent = Agent(input_dims= state_shape, env=env, n_actions= action_shape)
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    n_games = 5000
    
    for i in range(n_games):
        obs_log = []
        curr_pos_log = []
        next_obs_log=[]
        next_curr_pos_log=[]

        state = env.reset().values()
        obs, curr_pos, goal_pos = state
        init_pos = curr_pos
        score = 0
        
        HER_goal = init_pos
        
        for j in range(1, env._max_episode_steps+1):
            
            action = agent.choose_action(np.concatenate((obs, curr_pos, goal_pos)))
            next_state, _, done, info =  env.step(action)
            next_obs, next_curr_pos, goal_pos = next_state.values()
            reward = reward_engg(goal_pos, curr_pos, init_pos)
            
            # Add Normal Experience to memory
            exp = np.concatenate((obs, curr_pos, goal_pos))
            next_exp = np.concatenate((next_obs, next_curr_pos, goal_pos))
            agent.remember(exp.reshape((state_shape,)), action, reward, next_exp.reshape((state_shape,)), done)
            
            # Log the history of the states
            curr_pos_log.append(curr_pos)
            next_curr_pos_log.append(next_curr_pos)
            obs_log.append(obs)
            next_obs_log.append(next_obs)
            
            # Update the States        
            curr_pos = next_curr_pos
            obs = next_obs 
            score += reward

            if done:
                # HER Goal Initialization
                HER_goal = next_curr_pos
        
        # Hindsight Experience Replay        
        for k in range(j):      
            exp = np.concatenate((obs_log[k], curr_pos_log[k], HER_goal))
            next_exp = np.concatenate((next_obs_log[k], next_curr_pos_log[k], HER_goal))
            HER_reward = reward_engg(HER_goal, curr_pos_log[k], init_pos) 
            agent.remember(exp.reshape((state_shape,)), action, HER_reward, next_exp.reshape((state_shape,)), True)
        
        for l in range(64): # Another Hyperparameter
            agent.learn() 
         
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(f'Episode: {i} \t Avg. Episodic Reward: {score:.4f}')  
        np.save('score_history', score_history, allow_pickle=False)