import yaml

with open("./src/config/config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

def update_CONFIG(config_yaml_path):
    with open(config_yaml_path, "r") as f:
        config_target = yaml.safe_load(f)
    for key, value in config_target.items():
        CONFIG[key] = value

def save_CONFIG(config_yaml_path):
    with open(config_yaml_path, "w") as f:
        yaml.safe_dump(CONFIG, f, sort_keys=False, allow_unicode=True)
