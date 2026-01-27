import os
import shutil
import json
import os
import re
import numpy as np
from anytree import Node, RenderTree, find
from src.callback.common.iter_base_callback import IterBaseCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import FloatSchedule, ConstantSchedule
from src.callback.stage_schedule import HierarchicalStageScheduleCallback
from src.config.base import save_config
from stable_baselines3.ppo.ppo import PPO


def from_dict(json_dict, parent=None):
    node = Node(json_dict["name"], parent=parent, note=json_dict["note"])
    for child in json_dict["children"]:
        from_dict(child, node)
    return node


def to_dict(node):
    json_dict = {
        "name": node.name,
        "note": node.note,
        "children": [to_dict(c) for c in node.children],
        }
    return json_dict


def to_txt(root, txt_path="", checkpoints_path=""):
    with open(txt_path, "w", encoding="utf-8") as f:
        for pre, fill, node in RenderTree(root):
            pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(\d+)$')
            m = pattern.match(node.name)
            if m:
                zip_exist = os.path.isfile(
                    os.path.join(checkpoints_path, m.group(1), f"mdl_{node.name}.zip"))
                pkl_exist = os.path.isfile(
                    os.path.join(checkpoints_path, m.group(1), f"env_{node.name}.pkl"))
                marker = "" if (zip_exist and pkl_exist) else "*"
            else:
                marker = "*"
            print(f"{pre}{node.name}{marker}\t({node.note})", file=f)
        print("\n\n\nNote: Models marked with * have been deleted.", file=f)


def update_checkpoints_tree(child, parent="root", note="", checkpoints_path="."):
    json_path = os.path.join(checkpoints_path, "checkpoint_tree.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    root = from_dict(data)

    node = find(root, lambda n: n.name == parent)
    if node:
        Node(child, parent=node, note=note)
    else:
        Node(child, parent=root, note=note)

    json_dict = to_dict(root)
    json.dump(json_dict, open(json_path, "w"), ensure_ascii=False, indent=2)

    txt_path = os.path.join(checkpoints_path, "checkpoint_tree.txt")
    to_txt(root, txt_path=txt_path, checkpoints_path=checkpoints_path)


class CustomCheckpoint():
    def __init__(self, verbose: int = 2, **kwargs):
        self.verbose = verbose

    def _save_model(self, model, path):
        model.save(path)
        if self.verbose >= 2:
            print(f"Saving model to {path}")

    def _save_vec_norm_env(self, env, path):
        if env is not None:
            env.save(path)
            if self.verbose >= 2:
                print(f"Saving vecnormalized env to {path}")

    def _save_checkpoint_tree(self, child_name, parent_name, note, checkpoints_path):
        update_checkpoints_tree(child=child_name,
                                parent=parent_name,
                                note=note,
                                checkpoints_path=checkpoints_path)
        return child_name

    def _save_config(self, config, path):
        save_config(config, path)
        if self.verbose >= 2:
            print(f"Saving config to {path}")

    def _save_pyfile(self, origin_path, target_path):
        shutil.copy2(origin_path, target_path)
        if self.verbose >= 2:
            print(f"Saving origin py.file of customize env to {target_path}")

    def _save_stage(self, stage, path):
        np.save(path, stage)
        if self.verbose >= 2:
            print(f"Saving training stage to {path}")


class FlatCheckpointCallback(BaseCallback, CustomCheckpoint):
    def __init__(self,
                 save_name: str,
                 save_dir: str,
                 note: str = "",
                 config: dict = {},
                 base_name: str = None,
                 save_freq: int = 200000,
                 env_py_path: str = None,
                 checkpoint_tree_file_path: str = None,
                 checkpoints_path: str = ".",
                 save_vecnormalize: bool = True,
                 verbose: int = 2,
                 **kwargs):
        super().__init__(verbose)
        self.save_name = save_name
        self.save_dir = save_dir
        self.base_name = base_name
        self.note = note
        self.config = config
        self.save_freq = save_freq
        self.env_py_path = env_py_path
        self.checkpoint_tree_file_path = checkpoint_tree_file_path
        self.checkpoints_path = checkpoints_path
        self.save_vecnormalize = save_vecnormalize
        self.save_count = 1
        self.last_save_stamp = None

    def _on_training_start(self, **kwargs) -> bool:
        self.save_freq = (-self.save_freq % self.model.n_envs) + self.save_freq
        return True
        
    def _on_step(self, **kwargs) -> bool:
        if (self.n_calls * self.model.n_envs) % self.save_freq == 0:
            self._save_checkpoint(self.n_calls * self.model.n_envs)
        return True

    def _on_training_end(self, **kwargs) -> bool:
        self._save_checkpoint(self.n_calls * self.model.n_envs)
        return True
    
    @property
    def _counted_save_name(self) -> str:
        return f"{self.save_name}_{self.save_count}"

    def _save_checkpoint(self, current_stamp) -> bool:
        if self.last_save_stamp == current_stamp:
            return True
        self._on_saving_start()

        self._save_model(self.model,
            os.path.join(self.save_dir, f"mdl_{self._counted_save_name}.zip"))
        self._save_vec_norm_env(self.model.get_vec_normalize_env(),
            os.path.join(self.save_dir, f"env_{self._counted_save_name}.pkl"))
        self.base_name = self._save_checkpoint_tree(child_name=self._counted_save_name,
                                                    parent_name=self.base_name,
                                                    note=self.note,
                                                    checkpoints_path=self.checkpoints_path)
        self._save_config(self.config,
            os.path.join(self.save_dir, f"cfg_{self._counted_save_name}.yaml"))
        self._save_pyfile(self.env_py_path,
            os.path.join(self.save_dir, f"bkp_{self._counted_save_name}.py"))
        self._save_stage(self.model.env.venv.envs[0].env.env.env.env.stage,
            os.path.join(self.save_dir, f"cst_{self._counted_save_name}.npy"))

        self._on_saving_end()
        self.last_save_stamp = current_stamp
        return True
    

    def _on_saving_start(self) -> None:
        self.lr_schedule_tmp = self.model.lr_schedule
        self.lr_tmp = self.model.lr_schedule(self.model._current_progress_remaining)
        self.model.learning_rate = self.lr_tmp
        self.model.lr_schedule = FloatSchedule(ConstantSchedule(self.lr_tmp))


    def _on_saving_end(self) -> None:
        self.model.lr_schedule = self.lr_schedule_tmp
        self.note = ""
        self.save_count += 1


class HierarchicalCheckpointCallback(IterBaseCallback, CustomCheckpoint):
    def __init__(self,
                 loco_model: PPO,
                 mode_model: PPO,
                 save_name: str,
                 save_dir: str,
                 note: str = "",
                 config: dict = {},
                 base_name: str = None,
                 save_freq_iterations: int = 10,
                 loco_env_py_path: str = None,
                 mode_env_py_path: str = None,
                 checkpoint_tree_file_path: str = None,
                 checkpoints_path: str = ".",
                 save_vecnormalize: bool = True,
                 verbose: int = 2,
                 **kwargs):
        super().__init__(verbose)
        self.loco_model = loco_model
        self.mode_model = mode_model
        self.save_name = save_name
        self.save_dir = save_dir
        self.base_name = base_name
        self.note = note
        self.config = config
        self.save_freq_iterations = save_freq_iterations
        self.loco_env_py_path = loco_env_py_path
        self.mode_env_py_path = mode_env_py_path
        self.checkpoint_tree_file_path = checkpoint_tree_file_path
        self.checkpoints_path = checkpoints_path
        self.save_vecnormalize = save_vecnormalize
        self.save_count = 1
        self.last_save_calls = 0
        self.n_interations = 0

    def _on_training_start(self, **kwargs) -> bool:
        return True
        
    def _on_step(self, **kwargs) -> bool:
        return True

    def _on_training_end(self, **kwargs) -> bool:
        self._save_checkpoint()
        return True
    
    def _on_iteration_end(self, model, **kwargs) -> bool:
        if model == self.mode_model:
            self.n_interations += 1
            if self.n_interations % self.save_freq_iterations == 0:
                self._save_checkpoint()
        return True
    
    @property
    def _counted_save_name(self) -> str:
        return f"{self.save_name}_{self.save_count}"

    def _save_checkpoint(self) -> bool:
        if self.last_save_calls == self.n_calls:
            return True
        self._on_saving_start()

        self._save_model(self.loco_model,
            os.path.join(self.save_dir, f"mdl_loco_{self._counted_save_name}.zip"))
        self._save_vec_norm_env(self.loco_model.get_vec_normalize_env(),
            os.path.join(self.save_dir, f"env_loco_{self._counted_save_name}.pkl"))
        self._save_pyfile(self.loco_env_py_path,
            os.path.join(self.save_dir, f"bkp_loco_{self._counted_save_name}.py"))
        
        self._save_model(self.mode_model,
            os.path.join(self.save_dir, f"mdl_mode_{self._counted_save_name}.zip"))
        self._save_vec_norm_env(self.mode_model.get_vec_normalize_env(),
            os.path.join(self.save_dir, f"env_mode_{self._counted_save_name}.pkl"))
        self._save_pyfile(self.mode_env_py_path,
            os.path.join(self.save_dir, f"bkp_mode_{self._counted_save_name}.py"))
        
        self._save_config(self.config,
            os.path.join(self.save_dir, f"cfg_{self._counted_save_name}.yaml"))
        self._save_stage(self.stage,
            os.path.join(self.save_dir, f"cst_{self._counted_save_name}.npy"))
        
        self.base_name = self._save_checkpoint_tree(child_name=self._counted_save_name,
                                                    parent_name=self.base_name,
                                                    note=self.note,
                                                    checkpoints_path=self.checkpoints_path)

        self._on_saving_end()
        self.last_save_calls = self.n_calls
        return True
    

    def _on_saving_start(self) -> None:
        self.loco_lr_schedule_tmp = self.loco_model.lr_schedule
        self.loco_lr_tmp = self.loco_model.lr_schedule(self.loco_model._current_progress_remaining)
        self.loco_model.learning_rate = self.loco_lr_tmp
        self.loco_model.lr_schedule = FloatSchedule(ConstantSchedule(self.loco_lr_tmp))
        self.loco_logger_tmp = self.loco_model._logger
        self.loco_model._logger = None

        self.mode_lr_schedule_tmp = self.mode_model.lr_schedule
        self.mode_lr_tmp = self.mode_model.lr_schedule(self.mode_model._current_progress_remaining)
        self.mode_model.learning_rate = self.mode_lr_tmp
        self.mode_model.lr_schedule = FloatSchedule(ConstantSchedule(self.mode_lr_tmp))
        self.mode_logger_tmp = self.mode_model._logger
        self.mode_model._logger = None

        


    def _on_saving_end(self) -> None:
        self.loco_model.lr_schedule = self.loco_lr_schedule_tmp
        self.loco_model._logger = self.loco_logger_tmp
        self.mode_model.lr_schedule = self.mode_lr_schedule_tmp
        self.mode_model._logger = self.mode_logger_tmp
        self.note = ""
        self.save_count += 1

    @property
    def stage(self) -> float:
        for cb in self.locals["com_callback"]:
            if isinstance(cb, HierarchicalStageScheduleCallback):
                return cb.stage
        return 0.