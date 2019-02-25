"""
Module used for plotting
Code from:
https://github.com/dennybritz/reinforcement-learning/blob/master/lib/plotting.py
"""
import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("Mountain \"Cost To Go\" Function")
    fig.colorbar(surf)
    plt.show()


def plot_value_function_blackjack(state_values, title):
    """
    Plots the value function of a Blackjack solution as a surface plot.
    :param state_values: state values of the Blackjack game.
    :param title: title of the plot
    """
    x_states = [s[0] for s in state_values.keys()]
    y_states = [s[1] for s in state_values.keys()]
    min_x = np.min(x_states)
    max_x = np.max(x_states)
    min_y = np.min(y_states)
    max_y = np.max(y_states)

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    z_no_ace = np.apply_along_axis(lambda _: state_values[(_[0], _[1], False)],
                                   2, np.dstack([x_grid, y_grid]))
    z_ace = np.apply_along_axis(lambda _: state_values[(_[0], _[1], True)],
                                2, np.dstack([x_grid, y_grid]))

    def plot_surface(x, y, z, plot_title):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(x, y, z, rstride=1, cstride=1,
                        cmap='viridis')
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(plot_title)
        plt.show()

    plot_surface(x_grid, y_grid, z_no_ace, "{} (No Usable Ace)".format(title))
    plot_surface(x_grid, y_grid, z_ace, "{} (Usable Ace)".format(title))


def plot_episode_stats(stats, smoothing_window=10, no_show=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if no_show:
        plt.close()
    else:
        plt.show()

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(
        smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if no_show:
        plt.close()
    else:
        plt.show()

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if no_show:
        plt.close()
    else:
        plt.show()

    return fig1, fig2, fig3
