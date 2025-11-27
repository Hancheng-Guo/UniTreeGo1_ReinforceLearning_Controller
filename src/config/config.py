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

def get_CONFIG(cfg=CONFIG, field=None, try_keys=[]):
    if field is not None:
        try_cfg = cfg.get(field, {})
        optional_cfg = {}
        for try_key in try_keys:
            if try_key in try_cfg:
                try_value = try_cfg.get(try_key)
                optional_cfg[try_key] = try_value
        return optional_cfg
    return {}