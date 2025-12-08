import math
import matplotlib
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from src.utils.noop import noop
from src.config.config import CONFIG


def init_plt_render(render_mode):
    if render_mode in {"human", "rgb_array", "depth_array", "rgbd_tuple"}:
        plt_render, plt_endline = _plt_render(render_mode)
    else:
        plt_render = noop
        plt_endline = noop
    return plt_render, plt_endline

def _plt_render(render_mode):
    fig = None
    plt_data = dict()
    plt_axes = None
    plt_n_lines = CONFIG["demo"]["plt_n_lines"]
    plt_line = [[] for _ in range(plt_n_lines)]
    plt_max_col = 4
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.titlesize"] = 8.4
    plt.rcParams["axes.labelsize"] = 8.4
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    if render_mode == "human":
        plt.ion()

    def plt_select_kwargs(state, info):
        selected_kwargs = {
            "x_velocity": {"value": info["x_velocity"], "needs_unwrap": False},
            # "y_velocity": {"value": info["y_velocity"], "needs_unwrap": False},
            "y_pitch": {"value": info["pitch"], "needs_unwrap": True},
            "z_position": {"value": state[2], "needs_unwrap": False},
            "state": {"value": info["state"], "needs_unwrap": False},
            "reward": {"value": info["reward_total"], "needs_unwrap": False},
            "reward_forward": {"value": info["reward_forward"], "needs_unwrap": False},
            "reward_posture": {"value": info["reward_posture"], "needs_unwrap": False},
            "reward_state": {"value": info["reward_state"], "needs_unwrap": False},
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
        nonlocal fig, plt_data, plt_axes, plt_n_lines, plt_line, plt_max_col
        cols = min(plt_max_col, len(selected_kwargs))
        rows = math.ceil(len(selected_kwargs) / cols)
        fig, plt_axes = plt.subplots(rows, cols, figsize=(1.25*cols, 1.25*rows)) # default figsize=(6.4, 4.8)
        plt_axes = plt_axes.flatten()
        for i, (key, _) in enumerate(selected_kwargs.items()):
            plt_axes[i].set_title(key)

    def plt_newline(selected_kwargs):
        nonlocal fig, plt_data, plt_axes, plt_n_lines, plt_line, plt_max_col
        for i, (key, _) in enumerate(selected_kwargs.items()):
            plt_data[key] = list()
            line, = plt_axes[i].plot([], []) 
            plt_line[0].append(line)

    def plt_plot(selected_kwargs):
        nonlocal fig, plt_data, plt_axes, plt_n_lines, plt_line, plt_max_col
        for i, (key, value) in enumerate(selected_kwargs.items()):
            plt_data[key].append(value["value"])
            line_data = np.unwrap(plt_data[key]) if value["needs_unwrap"] else plt_data[key]
            plt_line[0][i].set_data(range(len(line_data)), line_data) 
            plt_update_lim(plt_axes[i])
        plt.tight_layout()
        if render_mode == "human":
            plt.pause(0.00001)

    def plt_render(state, info):
        nonlocal fig, plt_data, plt_axes, plt_n_lines, plt_line, plt_max_col
        selected_kwargs = plt_select_kwargs(state, info)
        if not fig: plt_newfig(selected_kwargs)
        if not plt_line[0]: plt_newline(selected_kwargs)
        plt_plot(selected_kwargs)
        fig.canvas.draw()
        plt_img = np.asarray(fig.canvas.renderer.buffer_rgba()).astype(np.uint8)
        return Image.fromarray(plt_img, mode="RGBA")

    def plt_endline():
        nonlocal fig, plt_data, plt_axes, plt_n_lines, plt_line, plt_max_col
        plt_data = dict()
        for line in plt_line[0]:
            line.set_alpha(1)
        for line in plt_line[len(plt_line) - 1]:
            line.remove()
        for i in range(len(plt_line) - 1, 0, -1):
            plt_line[i] = plt_line[i - 1]
            for line in plt_line[i]:
                line.set_alpha(line.get_alpha() - 1 / plt_n_lines)
        plt_line[0] = list()
    
    return plt_render, plt_endline


