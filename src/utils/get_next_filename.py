import os
import re

def get_next_filename(path, prefix, ext): # e.g. prefix="img", ext=".gif"

    pattern = re.compile(rf"{prefix}(\d+){ext}$")
    max_num = 0

    for filename in os.listdir(path): 
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num

    return f"{prefix}{max_num + 1}{ext}", max_num