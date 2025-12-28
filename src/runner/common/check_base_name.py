import os
import re


def check_base_name(base_name, config):
    if base_name:
        pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(\d+)$')
        match = pattern.match(base_name)
        if match:
            base_dir = os.path.join(config["path"]["output"], match.group(1))
            return base_name, base_dir
        else:
            pattern = re.compile(r'^(mdl|env)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(\d+)\.(zip|pkl)$')
            index = []
            for filename in os.listdir(os.path.join(config["path"]["output"], base_name)):
                match = pattern.match(filename)
                if match:
                    index.append(int(match.group(3)))
            index.sort(reverse=True)
            for i in range(len(index) - 1):
                if index[i] == index[i + 1]:
                    base_dir = os.path.join(config["path"]["output"], base_name)
                    base_name = f"{base_name}_{index[i]}"
                    return base_name, base_dir
    return None, None