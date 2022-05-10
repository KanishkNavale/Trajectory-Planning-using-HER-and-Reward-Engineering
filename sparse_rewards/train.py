# Library Imports
import gym
import numpy as np
import copy
import os
import sys
sys.path.append('DRL')
from TD3 import Agent


# HER Augmentation
def her_augmentation(agent, OBSs, actions, new_OBSs):
    """
    Desciption,
        1. HER Augmentation

    Args:
        agent ([class]): Instance of TD3 agent.
        OBSs ([type=np.float32, shape=(*, episode length, OBS.shape[0])]): list of OBS.
        actions ([type=np.float32,  shape=(*, episode length, action.shape[0])]]): list of actions.
        new_OBSs ([type=np.float32, shape=(*, episode length, new_OBS.shape[0])]): list of new_OBS.
    """
    # Hyperparameter for Future Goal Sampling
    k = 8

    # Augment the replay buffer
    T = len(actions)
    for index in range(T):
        for _ in range(k):
            # Always fetch index of upcoming episode transitions
            future = np.random.randint(index, T)

            # Unpack the buffers using the future index
            _, future_actgoal, _ = new_OBSs[future].values()
            HER_goal = copy.deepcopy(future_actgoal)

            # Compute HER Reward
            reward = agent.env.compute_reward(HER_goal, future_actgoal, 1.0)

            # Repack augmented episode transitions
            obs, _, _ = OBSs[future].values()
            state = np.concatenate((obs, future_actgoal, HER_goal))

            next_obs, _, _ = new_OBSs[future].values()
            next_state = np.concatenate((next_obs, future_actgoal, HER_goal))

            action = actions[future]

            # Add augmented episode transitions to agent's memory
            agent.memory.store_transition(state, action, reward, next_state, True)


# Main script pointer
if __name__ == "__main__":
    # Init. datapath
    data_path = os.getcwd() + '/HER/data/'

    # Load the environment
    env = gym.make('FetchReach-v1')

    # Init. Agent
    agent = Agent(env)

    # Init. Training
    best_score = env.reward_range[0]
    score_history = []
    avg_history = []
    n_games = 2500

    for i in range(n_games):
        score = 0
        done = False

        # Exp. Buffers
        OBSs = []
        actions = []
        next_OBSs = []

        # Initial Reset of Environment
        OBS = env.reset()

        tick = 0
        while not done:
            # Unpack the observation
            state, curr_actgoal, curr_desgoal = OBS.values()
            obs = np.concatenate((state, curr_actgoal, curr_desgoal))

            # Choose agent based action & make a transition
            action = agent.choose_action(obs)
            next_OBS, reward, done, info = env.step(action)

            next_state, next_actgoal, next_desgoal = next_OBS.values()
            next_obs = np.concatenate((next_state, next_actgoal, next_desgoal))

            agent.memory.store_transition(np.concatenate((state, curr_actgoal, curr_desgoal)),
                                          action,
                                          reward,
                                          np.concatenate((next_state, next_actgoal, next_desgoal)),
                                          done)

            # Exp. Buffers
            OBSs.append(OBS)
            actions.append(actions)
            next_OBSs.append(next_OBS)

            OBS = copy.deepcopy(next_OBS)
            score += reward
            tick += 1

            if tick == env._max_episode_steps:
                break

        # Init. HER augmentation
        her_augmentation(agent, OBSs, actions, next_OBSs)

        # Optimize the agent
        agent.optimize(64)

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_history.append(avg_score)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models(data_path)
            print(f'Episode:{i} \t ACC. Rewards: {score:3.2f} \t AVG. Rewards: {avg_score:3.2f} \t *** MODEL SAVED! ***')
        else:
            print(f'Episode:{i} \t ACC. Rewards: {score:3.2f} \t AVG. Rewards: {avg_score:3.2f}')

        # Save the score log
        np.save(data_path + 'score_history', score_history, allow_pickle=False)
        np.save(data_path + 'avg_history', avg_history, allow_pickle=False)
