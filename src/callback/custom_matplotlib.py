import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plt_select_kwargs(state, info):
    selected_kwargs = {
        "x_velocity": {"value": info.get("x_velocity"), "needs_unwrap": False},
        "x_velocity_target": {"value": info.get("x_velocity_target"), "needs_unwrap": False},
        "y_velocity": {"value": info.get("y_velocity"), "needs_unwrap": False},
        "y_velocity_target": {"value": info.get("y_velocity_target"), "needs_unwrap": False},
        "x_velocity_reward": {"value": info.get("x_velocity_reward"), "needs_unwrap": False},
        "y_velocity_reward": {"value": info.get("y_velocity_reward"), "needs_unwrap": False},
        "z_velocity_reward": {"value": info.get("z_velocity_reward"), "needs_unwrap": False},
        "z_position_reward": {"value": info.get("z_position_reward"), "needs_unwrap": False},
        "xy_angular_velocity_reward": {"value": info.get("xy_angular_velocity_reward"), "needs_unwrap": False},
        "xy_angular_reward": {"value": info.get("xy_angular_reward"), "needs_unwrap": False},
        "gait_loop_reward": {"value": info.get("gait_loop_reward"), "needs_unwrap": False},
        }
    return selected_kwargs


class CustomMatPlotLibCallback():
    def __init__(self,
                 render_mode,
                 plt_max_col=4,
                 font_size=7,
                 title_size=8.4,
                 label_size=8.4,
                 xlabel_size=7,
                 ylabel_size: float = 7.,
                 plt_n_lines: int = 1,
                 plt_x_range: int = 200,
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
        # set fig and axes
        selected_kwargs = plt_select_kwargs([],{})
        self.subfigs = len(selected_kwargs)
        self.cols = min(plt_max_col, self.subfigs)
        self.rows = math.ceil(self.subfigs / self.cols)
        self.fig, self.plt_axes = plt.subplots(self.rows, self.cols, figsize=(1.25*self.cols, 1.25*self.rows)) # default figsize=(6.4, 4.8)
        self.plt_axes = self.plt_axes.flatten()
        # set plt_line and plt_data
        self.plt_n_lines = plt_n_lines
        self.plt_x_range = plt_x_range
        for i, (key, _) in enumerate(selected_kwargs.items()):
            self.plt_axes[i].set_title(key)
        self.plt_line = [[] for _ in range(self.plt_n_lines)]
        self._init()

        
    def _on_training_start(self, env, *args, **kwargs):
        self.env = env
        self.env.plt_img = None

    def _on_episode_start(self, *args, **kwargs):
        if self.render_mode in {"human", "rgb_array", "depth_array", "rgbd_tuple"}:
            self.reset()


    def _on_step(self, state, info, *args, **kwargs):
        if self.render_mode in {"human", "rgb_array", "depth_array", "rgbd_tuple"}:
            selected_kwargs = plt_select_kwargs(state, info)
            self._plot(selected_kwargs)
            self.fig.canvas.draw()
            plt_img = np.asarray(self.fig.canvas.renderer.buffer_rgba()).astype(np.uint8)
            self.env.plt_img = Image.fromarray(plt_img, mode="RGBA")


    def _on_episode_end(self, *args, **kwargs):
        if self.render_mode in {"human", "rgb_array", "depth_array", "rgbd_tuple"}:
            self.reset()

    
    def reset(self):
        for line in self.plt_line[0]:
            line.set_alpha(1)
        for line in self.plt_line[len(self.plt_line) - 1]:
            line.remove()
        for i in range(len(self.plt_line) - 1, 0, -1):
            self.plt_line[i] = self.plt_line[i - 1]
            for line in self.plt_line[i]:
                line.set_alpha(line.get_alpha() - 1 / len(self.plt_line))
        self._init()

    def _init(self):
        self.plt_data = dict()
        self.plt_line[0] = list()
        for i, (key, _) in enumerate(plt_select_kwargs([],{}).items()):
            self.plt_data[key] = list()
            line, = self.plt_axes[i].plot([], []) 
            self.plt_line[0].append(line)


    def _plot(self, selected_kwargs):
        for i, (key, value) in enumerate(selected_kwargs.items()):
            self.plt_data[key].append(value["value"])
            line_data = np.unwrap(self.plt_data[key]) if value["needs_unwrap"] else self.plt_data[key]
            self.plt_line[0][i].set_data(range(len(line_data)), line_data)
            if len(line_data):
                self._update_lim(self.plt_axes[i])
        plt.tight_layout()
        if self.render_mode == "human":
            plt.pause(0.00001)

    def _update_lim(self, ax):
        try:
            ax.relim()
            ax.set_xlim(max(ax.dataLim.x0, ax.dataLim.x1 - self.plt_x_range),
                        max(ax.dataLim.x0 + self.plt_x_range, ax.dataLim.x1))
            if ax.dataLim.y1 - ax.dataLim.y0 < 0.5:
                y_padding = (0.5 - (ax.dataLim.y1 - ax.dataLim.y0)) / 2
            else:
                y_padding = 0
            ax.set_ylim(ax.dataLim.y0 - y_padding, ax.dataLim.y1 + y_padding)
        except:
            pass
