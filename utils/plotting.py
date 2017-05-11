import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3

def plot_visited_states(visits, num_episodes):
    n_thousand_episode = int(np.floor(num_episodes / 1000))
    eps = list(range(1, n_thousand_episode + 1))

    plt.figure(figsize=(10,5))
    for i_state in range(2, 6):
        state_label = "State %d" % (i_state + 1)
        plt.plot(eps, visits[:, i_state]/1000, label=state_label)

    plt.legend()
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.1, 1.2)
    plt.xlim(1, 12)
    plt.title("Number of visits (for States 3 to 6) averaged over 1000 episodes")
    plt.grid(True)
