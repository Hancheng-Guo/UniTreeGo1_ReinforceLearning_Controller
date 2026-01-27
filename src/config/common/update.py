import yaml


def update_config(config, config_yaml_path):
    with open(config_yaml_path, "r") as f:
        config_target = yaml.safe_load(f)
    for key, value in config_target.items():
        config[key] = value