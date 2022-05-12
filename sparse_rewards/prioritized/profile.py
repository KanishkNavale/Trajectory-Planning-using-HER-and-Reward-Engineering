import os
import numpy as np
import json

import gym
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Init. path
    data_path = os.path.abspath('sparse_rewards/prioritized/data')

    with open(os.path.join(data_path, 'training_info.json')) as f:
        train_data = json.load(f)

    with open(os.path.join(data_path, 'testing_info.json')) as f:
        test_data = json.load(f)

    # Load all the data frames
    score = [data["Epidosic Summed Rewards"] for data in train_data]
    average = [data["Moving Mean of Episodic Rewards"] for data in train_data]
    test = [data["Test Score"] for data in test_data]

    # Generate graphs
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    axes[0].plot(score, alpha=0.5, label='Episodic summation')
    axes[0].plot(average, label='Moving mean of 100 episodes')
    axes[0].grid(True)
    axes[0].set_xlabel('Training Episodes')
    axes[0].set_ylabel('Rewards')
    axes[0].legend(loc='best')
    axes[0].set_title('Training Profile')

    axes[1].boxplot(test)
    axes[1].grid(True)
    axes[1].set_xlabel('Test Run')
    axes[1].set_title('Testing Profile')

    fig.tight_layout()
    plt.savefig(os.path.join(data_path, 'HER + PDDPG with Dense Rewards.png'))
