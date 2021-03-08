import gym
import numpy as np
from ddpg import Agent

if __name__ == "__main__":
    # Load the environment
    env = gym.make('FetchReach-v1')
    desiredgoal_shape =  env.observation_space['desired_goal'].shape[0]
    achievedgoal_shape =  env.observation_space['achieved_goal'].shape[0]
    observation_shape =  env.observation_space['observation'].shape[0]
    action_shape = env.action_space.shape[0]
    state_shape = observation_shape + desiredgoal_shape
    
    agent = Agent(input_dims= state_shape, env=env, n_actions= action_shape)
    best_score = env.reward_range[0]
    load_checkpoint = False
    n_games = 5000
    score_history = []
    

    for i in range(n_games):
        state = env.reset().values()
        obs, curr_pos, goal_pos = state
        init_pos = curr_pos
        score = 0

        for j in range(env._max_episode_steps):
            
            action = agent.choose_action(np.concatenate((obs, goal_pos)))
            next_state, reward, done, info =  env.step(action)
            next_obs, next_curr_pos, goal_pos = next_state.values()
                
            # Add Normal Experience to memory
            if not done:
                exp = np.concatenate((obs, goal_pos))
                next_exp = np.concatenate((next_obs, goal_pos))
                
                agent.remember(exp.reshape((state_shape,)), action, reward, next_exp.reshape((state_shape,)), done)
                agent.learn()

          
            # Add 'Hindsight' Experience to memory with Future Goal Orientation
            if done:
                HER_exp = np.concatenate((obs, curr_pos))
                HER_next_exp = np.concatenate((next_obs, curr_pos))
            
                HER_goal = curr_pos.copy()
                HER_reward = env.compute_reward(curr_pos, HER_goal, info)
                
                agent.remember(HER_exp.reshape((state_shape,)), action, HER_reward, HER_next_exp.reshape((state_shape,)), done)
                agent.learn()
                    
            curr_pos = next_curr_pos
            obs = next_obs 
            score += reward

             
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
                
        print(f'Episode: {i} \t Sum Eps. Reward: {score:.2f}')  
        np.save('score_history', score_history, allow_pickle=False)