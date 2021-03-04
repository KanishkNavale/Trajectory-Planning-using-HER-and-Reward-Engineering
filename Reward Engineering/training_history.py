import numpy as np
import matplotlib.pyplot as plt

def plot_results(data, smoothing):
    y = []
    sum = []
    smoothing = int(smoothing)
    for i, point in enumerate(data):
        sum.append(point)
        if len(sum) == smoothing:
            mean = np.array(sum).mean()
            for i in range(len(sum)):
                y.append(mean)
            sum = []
    if len(sum) > 0:
        mean = np.array(sum).mean()
        for i in range(len(sum)):
            y.append(mean)
    
    assert len(data) == len(y)
    plt.plot(data, c='red', alpha=0.25)
    plt.plot(y, c='red')
    plt.xlabel('Episodes')
    plt.ylabel('Avg. Episodic Reward')
    plt.savefig("Avg_Rewards.png")
    
if __name__ == '__main__':
    plot_results(np.load('score_history.npy'), 500)