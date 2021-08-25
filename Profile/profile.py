# Library Import
import numpy as np
import os
import matplotlib.pyplot as plt

# Init. path
path = os.getcwd()

# Load all the data frames
acc_trainher = np.load(path + '/HER/data/score_history.npy', allow_pickle=False)
avg_trainher = np.load(path + '/HER/data/avg_history.npy', allow_pickle=False)
sum_testher = np.load(path + '/HER/data/test_score.npy', allow_pickle=False)

acc_trainre = np.load(path + '/Reward Engineering/data/score_history.npy', allow_pickle=False)
avg_trainre = np.load(path + '/Reward Engineering/data/avg_history.npy', allow_pickle=False)
sum_testre = np.load(path + '/Reward Engineering/data/test_score.npy', allow_pickle=False)

# Generate graphs
plt.figure(1)
plt.plot(acc_trainher, alpha=0.25, label='ACC. Rewards')
plt.plot(avg_trainher, label='AVG. Rewards')
plt.grid(True)
plt.xlabel('Training Episodes')
plt.ylabel('Rewards')
plt.legend(loc='best')
plt.title('HER Training Profile')
plt.savefig(path + '/Profile/data/' + 'HER Training Profile.png')

plt.figure(2)
plt.plot(acc_trainre, alpha=0.25, label='ACC. Rewards')
plt.plot(avg_trainre, label='AVG. Rewards')
plt.grid(True)
plt.xlabel('Training Episodes')
plt.ylabel('Rewards')
plt.legend(loc='best')
plt.title('Dense Reward Engg. Training Profile')
plt.savefig(path + '/Profile/data/' + 'RE Training Profile.png')

plt.figure(3)
plt.plot(sum_testre, label='HER Sum. Rewards')
plt.plot(sum_testher, label='RE Sum. Rewards')
plt.grid(True)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend(loc='best')
plt.title('Testing Profiles')
plt.savefig(path + '/Profile/data/' + 'Testing Profile.png')

# Analysis
print(f'HER based TD3 Agent Mean Test Score: {sum_testher[-1]/len(sum_testher)}')
print(f'RE based TD3 Agent Mean Test Score: {sum_testre[-1]/len(sum_testre)}')
print(f'Test Performance Ratio (HER Agent | RE Agent)=({(sum_testher[-1]/len(sum_testher))/(sum_testre[-1]/len(sum_testre))} | 1)')
