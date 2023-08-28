'''
This code is mainly based off https://github.com/RoyElkabetz/DQN_with_PyTorch_and_Gym
'''

import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, epsilons, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label='1')
    ax2 = fig.add_subplot(111, label='2', frame_on=False)

    ax.plot(x, epsilons, color='C0')
    ax.set_xlabel('Training Steps', color='C0')
    ax.set_ylabel('Epsilon', color='C0')
    ax.tick_params(axis='x', color='C0')
    ax.tick_params(axis='y', color='C0')

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[np.max([0, t - 100]):(t + 1)])

    ax2.scatter(x, running_avg, color='C1')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Average Reward', color='C1')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', color='C1')

    plt.savefig(filename)
