import yaml


with open("./src/config/config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)


from src.config.common.get import get_CONFIG
from src.config.common.save import save_CONFIG
from src.config.common.update import update_CONFIG


__all__ = [
    "get_CONFIG",
    "save_CONFIG",
    "update_CONFIG",
    ]