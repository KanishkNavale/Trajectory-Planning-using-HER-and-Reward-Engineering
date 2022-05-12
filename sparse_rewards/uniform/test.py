from typing import Dict, List
import os
import json

from tqdm import tqdm
import numpy as np
import gym

from algorithms.DDPG import Agent


if __name__ == '__main__':

    # Init. Environment
    env = gym.make('FetchReach-v1')
    env.reset()

    # Init. Datapath
    data_path = os.path.abspath('sparse_rewards/uniform/data')

    # Init. Testing
    n_games = 10
    test_data: List[Dict[str, np.ndarray]] = [] * n_games

    # Init. Agent
    agent = Agent(env=env, n_games=n_games, training=False)
    agent.load_models(data_path)

    for i in tqdm(range(n_games), desc=f'Testing', total=n_games):
        score_history: List[np.float32] = [] * n_games

        for _ in tqdm(range(n_games), desc=f'Testing', total=n_games):
            done: bool = False
            score: float = 0.0

            # Initial Reset of Environment
            OBS: Dict[str, np.array] = env.reset()

            while not done:

                # Unpack the observation
                state, curr_actgoal, curr_desgoal = OBS.values()
                obs = np.concatenate((state, curr_actgoal, curr_desgoal))

                # Choose agent based action & make a transition
                action = agent.choose_action(obs)
                next_OBS, reward, done, info = env.step(action)

                OBS = next_OBS
                score += reward

            score_history.append(score)

        print(f'Test Analysis:\n'
              f'Mean:{np.mean(score_history)}\n'
              f'Variance:{np.std(score_history)}')

        test_data.append({'Test Score': score_history})

    # Dump .json
    with open(os.path.join(data_path, 'testing_info.json'), 'w', encoding='utf8') as file:
        json.dump(test_data, file, indent=4, ensure_ascii=False)
