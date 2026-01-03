import os
import shutil
import json
import os
import re
import numpy as np
from anytree import Node, RenderTree, find
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import FloatSchedule, ConstantSchedule
from src.config.base import save_config


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


def update_checkpoints_tree(child, parent="root", note="",
                            file_path="", checkpoints_path="."):
    json_path = f"{file_path}.json"
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
    to_txt(root, txt_path=f"{file_path}.txt", checkpoints_path=checkpoints_path)


class CustomCheckpointCallback(BaseCallback):
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
        self.last_save_step = None

    def _on_training_start(self, **kwargs) -> bool:
        self.save_freq = (-self.save_freq % self.model.n_envs) + self.save_freq
        return True

    @property
    def _counted_save_name(self) -> str:
        return f"{self.save_name}_{self.save_count}"

    def _save_checkpoint(self) -> bool:
        if self.last_save_step == self.n_calls * self.model.n_envs:
            return True
        lr_schedule_tmp = self.model.lr_schedule
        lr_tmp = self.model.lr_schedule(self.model._current_progress_remaining)
        self.model.learning_rate = lr_tmp
        self.model.lr_schedule = FloatSchedule(ConstantSchedule(lr_tmp))
        print()

        # save model
        model_path = os.path.join(self.save_dir, f"mdl_{self._counted_save_name}.zip")
        self.model.save(model_path)
        if self.verbose >= 2:
            print(f"Saving model to {model_path}")

        # save vecnormalized env
        env_path = os.path.join(self.save_dir, f"env_{self._counted_save_name}.pkl")
        if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
            self.model.get_vec_normalize_env().save(env_path)
            if self.verbose >= 2:
                print(f"Saving vecnormalized env to {env_path}")

        # update checkpoints tree
        update_checkpoints_tree(child=self._counted_save_name,
                                parent=self.base_name,
                                note=self.note,
                                file_path=self.checkpoint_tree_file_path,
                                checkpoints_path=self.checkpoints_path)
        self.base_name = self._counted_save_name

        # save config
        config_path = os.path.join(self.save_dir, f"cfg_{self._counted_save_name}.yaml")
        save_config(self.config, config_path)
        if self.verbose >= 2:
            print(f"Saving config to {config_path}")

        # save origin py.file of customize env
        backup_path = os.path.join(self.save_dir, f"bkp_{self._counted_save_name}.py")
        shutil.copy2(self.env_py_path, backup_path)
        if self.verbose >= 2:
            print(f"Saving origin py.file of customize env to {backup_path}")

        # save training stage
        stage = self.model.env.venv.envs[0].env.env.env.env.stage
        stage_path = os.path.join(self.save_dir, f"cst_{self._counted_save_name}.npy")
        np.save(stage_path, stage)
        if self.verbose >= 2:
            print(f"Saving training stage to {stage_path}")

        print()
        self.last_save_step = self.n_calls * self.model.n_envs
        self.model.lr_schedule = lr_schedule_tmp
        self.note = ""
        self.save_count += 1
        return True
        
    def _on_step(self, **kwargs) -> bool:
        if (self.n_calls * self.model.n_envs) % self.save_freq == 0:
            self._save_checkpoint()
        return True

    def _on_training_end(self, **kwargs) -> bool:
        self._save_checkpoint()
        return True