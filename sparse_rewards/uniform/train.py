from typing import Dict, List
import os
import json

import numpy as np
import gym

from torch.utils.tensorboard import SummaryWriter

from algorithms.DDPG import Agent
from algorithms.augmentations import her_augmentation

# Init. tensorboard summary writer
tb = SummaryWriter(log_dir=os.path.abspath('sparse_rewards/uniform/data/tensorboard'))


if __name__ == '__main__':

    # Init. Environment
    env = gym.make('FetchReach-v1')
    env.reset()

    # Init. Datapath
    data_path = os.path.abspath('sparse_rewards/uniform/data')

    # Init. Training
    n_games: int = 2500
    best_score = -np.inf
    score_history: List[float] = [] * n_games
    avg_history: List[float] = [] * n_games
    logging_info: List[Dict[str, float]] = [] * n_games

    # Init. Agent
    agent = Agent(env=env, n_games=n_games)

    for i in range(n_games):
        done: bool = False
        score: float = 0.0

        states: List[Dict[str, np.ndarray]] = []
        actions: List[np.ndarray] = []
        next_states: List[Dict[str, np.ndarray]] = []

        # Initial Reset of Environment
        OBS: Dict[str, np.array] = env.reset()
        next_OBS: Dict[str, np.array]

        while not done:
            # Unpack the observation
            state, curr_actgoal, curr_desgoal = OBS.values()
            obs = np.concatenate((state, curr_actgoal, curr_desgoal))

            # Choose agent based action & make a transition
            action = agent.choose_action(obs)
            next_OBS, reward, done, info = env.step(action)

            next_state, next_actgoal, next_desgoal = next_OBS.values()
            next_obs = np.concatenate((next_state, next_actgoal, next_desgoal))

            agent.memory.add(obs, action, reward, next_obs, done)
            agent.optimize()

            states.append(OBS)
            next_states.append(next_OBS)
            actions.append(action)

            OBS = next_OBS
            score += reward

        her_augmentation(agent, states, actions, next_states)

        score_history.append(score)
        avg_score: float = np.mean(score_history[-100:])
        avg_history.append(avg_score)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models(data_path)
            print(f'Episode:{i}'
                  f'\t ACC. Rewards: {score:3.2f}'
                  f'\t AVG. Rewards: {avg_score:3.2f}'
                  f'\t *** MODEL SAVED! ***')
        else:
            print(f'Episode:{i}'
                  f'\t ACC. Rewards: {score:3.2f}'
                  f'\t AVG. Rewards: {avg_score:3.2f}')

        episode_info = {
            'Episode': i,
            'Total Episodes': n_games,
            'Epidosic Summed Rewards': score,
            'Moving Mean of Episodic Rewards': avg_score
        }

        logging_info.append(episode_info)

        # Add info. to tensorboard
        tb.add_scalars('training_rewards',
                       {'Epidosic Summed Rewards': score,
                        'Moving Mean of Episodic Rewards': avg_score}, i)

        # Dump .json
        with open(os.path.join(data_path, 'training_info.json'), 'w', encoding='utf8') as file:
            json.dump(logging_info, file, indent=4, ensure_ascii=False)

    # Close tensorboard writer
    tb.close()
