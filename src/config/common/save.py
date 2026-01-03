import yaml

def save_config(config, config_yaml_path):
    with open(config_yaml_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)