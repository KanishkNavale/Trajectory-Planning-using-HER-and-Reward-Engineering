# Library Imports
import gym
import numpy as np
import copy
import os
import sys
sys.path.append('DRL')
from TD3 import Agent


# Init. datapath
data_path = os.getcwd() + '/HER/data/'

# Load the environment
env = gym.make('FetchReach-v1')
OBS = env.reset()

# Init. Agent
agent = Agent(env)

for _ in range(10):
    state, curr_actgoal, curr_desgoal = OBS.values()
    obs = np.concatenate((state, curr_actgoal, curr_desgoal))

    # Choose agent based action & make a transition
    action = agent.choose_action(obs)
    next_OBS, reward, done, info = env.step(action)

    next_state, next_actgoal, next_desgoal = next_OBS.values()
    next_obs = np.concatenate((next_state, next_actgoal, next_desgoal))

    OBS = copy.deepcopy(next_OBS)
agent.optimize(1)

agent.actor.load_weights(data_path + 'actor.h5')

# Init. Training
n_games = 100
test_score = []
score = 0

for i in range(n_games):
    done = False

    # Initial Reset of Environment
    OBS = env.reset()
    _, init_actgoal, _ = OBS.values()

    while not done:
        # Render
        # env.render()

        # Unpack the observation
        state, curr_actgoal, curr_desgoal = OBS.values()
        obs = np.concatenate((state, curr_actgoal, curr_desgoal))

        # Choose agent based action & make a transition
        action = agent.choose_action(obs)
        next_OBS, reward, done, info = env.step(action)

        next_state, next_actgoal, next_desgoal = next_OBS.values()
        next_obs = np.concatenate((next_state, next_actgoal, next_desgoal))

        OBS = copy.deepcopy(next_OBS)
        score += reward

    print(f'Episode:{i} \t Summed Rewards: {score:3.2f}')

    test_score.append(score)
    np.save(data_path + 'test_score', test_score, allow_pickle=False)
env.close()
