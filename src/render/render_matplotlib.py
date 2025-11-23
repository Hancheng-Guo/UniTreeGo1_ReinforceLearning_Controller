import math
import threading
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def init_plt_render():
    fig = None
    plt_data = dict()
    plt_axes = None
    plt_n_line = 5
    plt_line = [[] for _ in range(plt_n_line)]
    plt_max_col = 3
    plt.ion()

    def plt_render(**kwargs):
        nonlocal fig, plt_data, plt_axes, plt_n_line, plt_line, plt_max_col
        if not fig:
            cols = min(plt_max_col, len(kwargs))
            rows = math.ceil(len(kwargs) / cols)
            fig, plt_axes = plt.subplots(rows, cols)
            plt_axes = plt_axes.flatten()
            for i, (key, value) in enumerate(kwargs.items()):
                plt_axes[i].relim()
                plt_axes[i].autoscale_view()
                plt_axes[i].set_title(key)
        if not plt_line[0]:
            for i, (key, value) in enumerate(kwargs.items()):
                plt_data[key] = list()
                line, = plt_axes[i].plot([], []) 
                plt_line[0].append(line)
        for i, (key, value) in enumerate(kwargs.items()):
            plt_data[key].append(value)
            plt_line[0][i].set_data(range(len(plt_data[key])), plt_data[key]) 
            plt_line[0][i].set_alpha(1)
            plt_axes[i].relim()
            plt_axes[i].autoscale_view()
        plt.pause(0.001)
        return 
    
    def plt_render_newline():
        nonlocal fig, plt_data, plt_axes, plt_n_line, plt_line, plt_max_col
        plt_data = dict()
        for line in plt_line[len(plt_line) - 1]:
            line.remove()
        for i in range(len(plt_line) - 1, 0, -1):
            plt_line[i] = plt_line[i - 1]
            for line in plt_line[i]:
                line.set_alpha(line.get_alpha() - 1 / plt_n_line)
        plt_line[0] = list()
        return 
    
    return plt_render, plt_render_newline


