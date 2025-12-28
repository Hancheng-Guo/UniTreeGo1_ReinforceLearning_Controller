def get_CONFIG(config, field=None, try_keys=[]):
    if field is not None:
        try_cfg = config.get(field, {})
        optional_cfg = {}
        for try_key in try_keys:
            if try_key in try_cfg:
                try_value = try_cfg.get(try_key)
                if isinstance(try_value, (list, dict, set)):
                    optional_cfg[try_key] = try_value.copy()
                else:
                    optional_cfg[try_key] = try_value
        return optional_cfg
    return {}