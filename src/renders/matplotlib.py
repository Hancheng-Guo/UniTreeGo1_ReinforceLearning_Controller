import math
import matplotlib
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from src.utils.noop import noop
from src.config.config import CONFIG


def plt_select_kwargs(state, info):
    selected_kwargs = {
        "x_velocity": {"value": info.get("x_velocity"), "needs_unwrap": False},
        "y_pitch": {"value": info.get("pitch"), "needs_unwrap": True},
        "z_position": {"value": state[2:3], "needs_unwrap": False},
        "state": {"value": info.get("state"), "needs_unwrap": False},
        "reward": {"value": info.get("reward_total"), "needs_unwrap": False},
        "reward_forward": {"value": info.get("reward_forward"), "needs_unwrap": False},
        "reward_posture": {"value": info.get("reward_posture"), "needs_unwrap": False},
        "reward_state": {"value": info.get("reward_state"), "needs_unwrap": False},
        }
    return selected_kwargs


class PltRenderer():
    def __init__(self,
                 render_mode,
                 plt_max_col=4,
                 font_size=7,
                 title_size=8.4,
                 label_size=8.4,
                 xlabel_size=7,
                 ylabel_size=7,
                 ):
        # set plt
        self.render_mode = render_mode
        plt.rcParams["font.size"] = font_size
        plt.rcParams["axes.titlesize"] = title_size
        plt.rcParams["axes.labelsize"] = label_size
        plt.rcParams["xtick.labelsize"] = xlabel_size
        plt.rcParams["ytick.labelsize"] = ylabel_size
        if render_mode == "human":
            plt.ion()
        # set_fig
        selected_kwargs = plt_select_kwargs([],{})
        self.subfigs = len(selected_kwargs)
        self.cols = min(plt_max_col, self.subfigs)
        self.rows = math.ceil(self.subfigs / self.cols)
        self.fig, self.plt_axes = plt.subplots(self.rows, self.cols, figsize=(1.25*self.cols, 1.25*self.rows)) # default figsize=(6.4, 4.8)
        self.plt_axes = self.plt_axes.flatten()
        for i, (key, _) in enumerate(selected_kwargs.items()):
            self.plt_axes[i].set_title(key)
        self.plt_line = [[] for _ in range(CONFIG["demo"]["plt_n_lines"])]
        self.plt_data = dict()
        for i, (key, _) in enumerate(selected_kwargs.items()):
            self.plt_data[key] = list()
            line, = self.plt_axes[i].plot([], []) 
            self.plt_line[0].append(line)
        
    def __call__(self, state, info, *args, **kwds):
        selected_kwargs = plt_select_kwargs(state, info)
        self._plot(selected_kwargs)
        self.fig.canvas.draw()
        plt_img = np.asarray(self.fig.canvas.renderer.buffer_rgba()).astype(np.uint8)
        return Image.fromarray(plt_img, mode="RGBA")
    
    def reset(self):
        self.plt_data = dict()
        for line in self.plt_line[0]:
            line.set_alpha(1)
        for line in self.plt_line[len(self.plt_line) - 1]:
            line.remove()
        for i in range(len(self.plt_line) - 1, 0, -1):
            self.plt_line[i] = self.plt_line[i - 1]
            for line in self.plt_line[i]:
                line.set_alpha(line.get_alpha() - 1 / self.plt_n_lines)
        self.plt_line[0] = list()

    def _plot(self, selected_kwargs):
        for i, (key, value) in enumerate(selected_kwargs.items()):
            self.plt_data[key].append(value["value"])
            line_data = np.unwrap(self.plt_data[key]) if value["needs_unwrap"] else self.plt_data[key]
            self.plt_line[0][i].set_data(range(len(line_data)), line_data) 
            self._update_lim(self.plt_axes[i])
        plt.tight_layout()
        if self.render_mode == "human":
            plt.pause(0.00001)

    def _update_lim(self, ax):
        ax.relim()
        ax.set_xlim(max(ax.dataLim.x0, ax.dataLim.x1 - CONFIG["demo"]["plt_x_range"]),
                    max(ax.dataLim.x0 + CONFIG["demo"]["plt_x_range"], ax.dataLim.x1))
        if ax.dataLim.y1 - ax.dataLim.y0 < 0.5:
            y_padding = (0.5 - (ax.dataLim.y1 - ax.dataLim.y0)) / 2
        else:
            y_padding = 0
        ax.set_ylim(ax.dataLim.y0 - y_padding, ax.dataLim.y1 + y_padding)
