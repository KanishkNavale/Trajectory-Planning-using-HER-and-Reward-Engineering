import gym
import numpy as np
from ddpg import Agent

def reward_engg(goal_pos, current_pos, initial_pos):
    max_dist = np.linalg.norm(goal_pos - initial_pos)
    min_dist = 0.0001
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
    state_shape = observation_shape + desiredgoal_shape
    
    agent = Agent(input_dims= state_shape, env=env, n_actions= action_shape)
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    n_games = 5000
    
    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            state = env.reset().values()
            obs, curr_pos, goal_pos = state
            init_pos = curr_pos
            action = env.action_space.sample()
            
            next_state, reward, done, info =  env.step(action)
            next_obs, next_curr_pos, goal_pos = next_state.values()
            
            # Add Normal Experience to memory & Dense Reward
            exp = np.concatenate((obs, goal_pos))
            next_exp = np.concatenate((next_obs, goal_pos))
            dense_reward = reward_engg(goal_pos, curr_pos, init_pos)
            agent.remember(exp.reshape((state_shape,)), action, dense_reward, next_exp.reshape((state_shape,)), done)
            
            # Add 'Hindsight' Experience to memory & Dense Reward
            exp = np.concatenate((obs, curr_pos))
            next_exp = np.concatenate((next_obs, next_curr_pos))
            fake_goal = curr_pos.copy()
            dense_reward = reward_engg(fake_goal, curr_pos, init_pos)
            agent.remember(exp.reshape((state_shape,)), action, dense_reward, next_exp.reshape((state_shape,)), True)
            n_steps += 1
            
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

    for i in range(n_games):
        state = env.reset().values()
        obs, curr_pos, goal_pos = state
        init_pos = curr_pos
        score = 0

        for j in range(1, env._max_episode_steps+1):
            
            action = agent.choose_action(np.concatenate((obs, goal_pos)))
            next_state, reward, done, info =  env.step(action)
            next_obs, next_curr_pos, goal_pos = next_state.values()
            
            # Add Normal Experience to memory & Dense Reward
            exp = np.concatenate((obs, goal_pos))
            next_exp = np.concatenate((next_obs, goal_pos))
            dense_reward = reward_engg(goal_pos, curr_pos, init_pos)
            agent.remember(exp.reshape((state_shape,)), action, dense_reward, next_exp.reshape((state_shape,)), done)
            
            # Add 'Hindsight' Experience to memory & Dense Reward
            exp = np.concatenate((obs, curr_pos))
            next_exp = np.concatenate((next_obs, next_curr_pos))
            fake_goal = curr_pos.copy()
            fake_reward = reward_engg(fake_goal, curr_pos, init_pos)
            agent.remember(exp.reshape((state_shape,)), action, fake_reward, next_exp.reshape((state_shape,)), True)
            
            if not load_checkpoint:
                agent.learn()
                
            curr_pos = next_curr_pos
            obs = next_obs 
            score += dense_reward
              
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print(f'Episode: {i} \t Avg. Episodic Reward: {score:.4f}')  
        np.save('score_history', score_history, allow_pickle=False)