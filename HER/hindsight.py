import gym
import numpy as np
from ddpg import Agent

if __name__ == "__main__":
    # Load the environment
    env = gym.make('FetchReach-v1')
    ip_shape =  env.observation_space['desired_goal'].shape[0]
    action_shape = env.action_space.shape[0]
    
    agent = Agent(input_dims= ip_shape, env=env, n_actions= action_shape)
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    n_games = 5000
    
    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            state = env.reset().values()
            pure_states, curr_pos, goal_pos = state
            init_pos = curr_pos
            action = env.action_space.sample()
            
            next_state, reward, done, info =  env.step(action)
            next_pure_states, next_curr_pos, goal_pos = next_state.values()
            
            # Add Normal Experience to memory
            agent.recall(curr_pos, goal_pos, action, reward, next_curr_pos, done)
            
            # Add 'Hindsight' Experience to memory
            fake_goal = curr_pos.copy()
            fake_reward = env.compute_reward(curr_pos, fake_goal, info)
            agent.recall(curr_pos, goal_pos, action, reward, next_curr_pos, True)
            n_steps += 1
            
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

    for i in range(n_games):
        state = env.reset().values()
        pure_states, curr_pos, goal_pos = state
        init_pos = curr_pos
        done = False
        score = 0

        for j in range(1, env._max_episode_steps+1):
            if j % n_games == 0:
                env.render()
            
            action = agent.choose_action(curr_pos, goal_pos)
            next_state, reward, done, info =  env.step(action)
            next_pure_states, next_curr_pos, goal_pos = next_state.values()
            score += reward
            
            # Add Normal Experience to memory
            agent.recall(curr_pos, goal_pos, action, reward, next_curr_pos, done)
            
            # Add 'Hindsight' Experience to memory
            fake_goal = curr_pos.copy()
            fake_reward = env.compute_reward(curr_pos, fake_goal, info)
            agent.recall(curr_pos, goal_pos, action, reward, next_curr_pos, True)
            
            if not load_checkpoint:
                agent.learn()
            curr_pos = next_curr_pos 
            
        env.close()     
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print(f'Episode: {i} \t Avg. Episodic Reward: {score:.4f}')  
        np.save('score_history', score_history, allow_pickle=False)