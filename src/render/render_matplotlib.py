import math
import threading
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def init_plt_render():
    fig = None
    plt_data = dict()
    plt_axes = None
    plt_line = list()
    plt.ion()

    def plt_render(**kwargs):
        nonlocal fig, plt_data, plt_axes, plt_line
        if not fig:
            cols = 3
            rows = math.ceil(len(kwargs) / cols)
            fig, plt_axes = plt.subplots(rows, cols)
            plt_axes = plt_axes.flatten()
        if not plt_line:
            for i, (key, value) in enumerate(kwargs.items()):
                plt_data[key] = list()
                line, = plt_axes[i].plot([], []) 
                plt_line.append(line)
        for i, (key, value) in enumerate(kwargs.items()):
            plt_data[key].append(value)
            plt_line[i].remove()
            line, = plt_axes[i].plot(range(len(plt_data[key])), plt_data[key]) 
            plt_line[i] = line
        plt.pause(0.001)
        return 
    
    def plt_render_newline():
        nonlocal fig, plt_data, plt_axes, plt_line
        plt_data = dict()
        plt_line = list()
        return 
    
    return plt_render, plt_render_newline


