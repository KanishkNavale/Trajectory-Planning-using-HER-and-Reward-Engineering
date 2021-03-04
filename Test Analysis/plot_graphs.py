import numpy as np
import matplotlib.pyplot as plt

def plot_graph(data, smoothing, color, title):
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
    
    plt.figure()
    plt.plot(data, c=color, alpha=0.25, label='Accumulated Rewards')
    plt.plot(y, c=color)
    plt.xlabel('Episodes')
    plt.ylabel('Accumulate Rewards')
    plt.grid(True)
    plt.legend(loc='best')
    plt.title(title)
    plt.savefig('Test Analysis/'+title+'.png')

if __name__ == '__main__':
      
    # Import all the training score_history
    her = np.load('HER/score_history.npy')
    re = np.load('Reward Engineering/score_history.npy')
    her_re = np.load ('HER + Reward Engieering/score_history.npy')

    # Plot the data
    her = plot_graph(her, 10, 'red', 'HER: Cummulated Rewards vs Episodes')
    re = plot_graph(re, 10, 'blue', 'Reward Engg: Cummulated Rewards vs Episodes')
    her_re = plot_graph(her_re, 10, 'green', 'HER + Reward Engg: Cummulated Rewards vs Episodes')
    
    
    
    
     