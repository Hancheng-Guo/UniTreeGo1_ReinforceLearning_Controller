import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from src.config.config import CONFIG

def init_plt_render(plt_clr=False):
    fig = None
    plt_data = dict()
    plt_axes = None
    plt_n_line = 1 if plt_clr else 5
    plt_line = [[] for _ in range(plt_n_line)]
    plt_max_col = 3
    plt.ion()

    def plt_select_kwargs(state, info):
        selected_kwargs = {
            "x_velocity": {"value": info["x_velocity"], "needs_unwrap": False},
            "y_velocity": {"value": info["y_velocity"], "needs_unwrap": False},
            "z_position": {"value": state[2], "needs_unwrap": False},
            "x_roll": {"value": info["roll"], "needs_unwrap": True},
            "y_pitch": {"value": info["pitch"], "needs_unwrap": True},
            "z_yaw": {"value": info["yaw"], "needs_unwrap": True},
            "reward_forward": {"value": info["reward_forward"], "needs_unwrap": False},
            "costs": {"value": info["reward_ctrl"] + info["reward_contact"], "needs_unwrap": False},
            "reward_total": {"value": info["reward_total"], "needs_unwrap": False},
            "foot_fz_max": {"value": max(info["foot_fz"]), "needs_unwrap": False},
            "foot_fz_mean": {"value": sum(info["foot_fz"]) / len(info["foot_fz"]), "needs_unwrap": False},
            "ori_contact_cost": {"value": info["clip_contact_forces_squared_sum"], "needs_unwrap": False},
            }
        return selected_kwargs
    
    def plt_update_lim(ax):
        ax.relim()
        ax.set_xlim(max(ax.dataLim.x0, ax.dataLim.x1 - CONFIG["demo"]["plt_x_range"]),
                    max(ax.dataLim.x0 + CONFIG["demo"]["plt_x_range"], ax.dataLim.x1))
        if ax.dataLim.y1 - ax.dataLim.y0 < 0.5:
            y_padding = (0.5 - (ax.dataLim.y1 - ax.dataLim.y0)) / 2
        else:
            y_padding = 0
        ax.set_ylim(ax.dataLim.y0 - y_padding, ax.dataLim.y1 + y_padding)

    def plt_newfig(selected_kwargs):
        nonlocal fig, plt_data, plt_axes, plt_n_line, plt_line, plt_max_col
        cols = min(plt_max_col, len(selected_kwargs))
        rows = math.ceil(len(selected_kwargs) / cols)
        fig, plt_axes = plt.subplots(rows, cols, figsize=(1.8*cols, 1.8*rows)) # default figsize=(6.4, 4.8)
        plt_axes = plt_axes.flatten()
        for i, (key, _) in enumerate(selected_kwargs.items()):
            # plt_update_lim(plt_axes[i])
            plt_axes[i].set_title(key)

    def plt_newline(selected_kwargs):
        nonlocal fig, plt_data, plt_axes, plt_n_line, plt_line, plt_max_col
        for i, (key, _) in enumerate(selected_kwargs.items()):
            plt_data[key] = list()
            line, = plt_axes[i].plot([], []) 
            # plt_update_lim(plt_axes[i])
            plt_line[0].append(line)

    def plt_plot(selected_kwargs):
        nonlocal fig, plt_data, plt_axes, plt_n_line, plt_line, plt_max_col
        for i, (key, value) in enumerate(selected_kwargs.items()):
            plt_data[key].append(value["value"])
            line_data = np.unwrap(plt_data[key]) if value["needs_unwrap"] else plt_data[key]
            plt_line[0][i].set_data(range(len(line_data)), line_data) 
            plt_update_lim(plt_axes[i])
        plt.tight_layout()
        plt.pause(0.00001)

    def plt_render(state, info):
        nonlocal fig, plt_data, plt_axes, plt_n_line, plt_line, plt_max_col
        selected_kwargs = plt_select_kwargs(state, info)
        if not fig: plt_newfig(selected_kwargs)
        if not plt_line[0]: plt_newline(selected_kwargs)
        plt_plot(selected_kwargs)

    def plt_endline():
        nonlocal fig, plt_data, plt_axes, plt_n_line, plt_line, plt_max_col
        plt_data = dict()
        for line in plt_line[0]:
            line.set_alpha(1)
        for line in plt_line[len(plt_line) - 1]:
            line.remove()
        for i in range(len(plt_line) - 1, 0, -1):
            plt_line[i] = plt_line[i - 1]
            for line in plt_line[i]:
                line.set_alpha(line.get_alpha() - 1 / plt_n_line)
        plt_line[0] = list()
    
    return plt_render, plt_endline


