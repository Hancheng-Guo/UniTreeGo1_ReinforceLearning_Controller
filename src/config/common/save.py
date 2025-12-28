import yaml
from src.config.base import CONFIG

def save_CONFIG(config_yaml_path):
    with open(config_yaml_path, "w") as f:
        yaml.safe_dump(CONFIG, f, sort_keys=False, allow_unicode=True)