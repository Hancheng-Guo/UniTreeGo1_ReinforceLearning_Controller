import numpy as np
from collections import deque

from src.reward.common.get_foot_state import get_foot_state


idle_loop = [{"state": 0b1111, "step": 0},]
walk_loop = [{"state": 0b1110, "step": 1},
             {"state": 0b1010, "step": 1},
             {"state": 0b1011, "step": 2},
             {"state": 0b1001, "step": 1},
             {"state": 0b1101, "step": 1},
             {"state": 0b0101, "step": 1},
             {"state": 0b0111, "step": 2},
             {"state": 0b0110, "step": 1},]
pace_loop = [{"state": 0b1010, "step": 4},
             {"state": 0b1110, "step": 3},
             {"state": 0b1111, "step": 2},
             {"state": 0b0111, "step": 1},
             {"state": 0b0101, "step": 4},
             {"state": 0b1101, "step": 3},
             {"state": 0b1111, "step": 2},
             {"state": 0b1011, "step": 1},]
trot_loop = [{"state": 0b1111, "step": 2},
             {"state": 0b1011, "step": 1},
             {"state": 0b1001, "step": 4},
             {"state": 0b1101, "step": 3},
             {"state": 0b1111, "step": 2},
             {"state": 0b0111, "step": 1},
             {"state": 0b0110, "step": 4},
             {"state": 0b1110, "step": 3},]
canter_loop_A = [{"state": 0b1110, "step": 2},
                 {"state": 0b1100, "step": 1},
                 {"state": 0b1000, "step": 1},
                 {"state": 0b0000, "step": 1},
                 {"state": 0b0001, "step": 2},
                 {"state": 0b0011, "step": 1},
                 {"state": 0b0111, "step": 1},
                 {"state": 0b0110, "step": 1},]
canter_loop_B = [{"state": 0b1101, "step": 2},
                 {"state": 0b1100, "step": 1},
                 {"state": 0b0100, "step": 1},
                 {"state": 0b0000, "step": 1},
                 {"state": 0b0010, "step": 2},
                 {"state": 0b0011, "step": 1},
                 {"state": 0b1011, "step": 1},
                 {"state": 0b1001, "step": 1},]
gallop_loop_A = [{"state": 0b1000, "step": 2},
                 {"state": 0b1100, "step": 1},
                 {"state": 0b0100, "step": 1},
                 {"state": 0b0000, "step": 1},
                 {"state": 0b0010, "step": 2},
                 {"state": 0b0011, "step": 1},
                 {"state": 0b0001, "step": 1},
                 {"state": 0b0000, "step": 1},]
gallop_loop_B = [{"state": 0b0100, "step": 2},
                 {"state": 0b1100, "step": 1},
                 {"state": 0b1000, "step": 1},
                 {"state": 0b0000, "step": 1},
                 {"state": 0b0001, "step": 2},
                 {"state": 0b0011, "step": 1},
                 {"state": 0b0010, "step": 1},
                 {"state": 0b0000, "step": 1},]

gait_loop_dict = {
    "idle":   {"speed": [-1e-7, 1e-7], "loop": [idle_loop]},
    "walk":   {"speed": None,          "loop": [walk_loop]},
    "pace":   {"speed": None,          "loop": [pace_loop]},
    "trot":   {"speed": [1e-7, 4.0],   "loop": [trot_loop]},
    "canter": {"speed": [4.0, 8.0],    "loop": [canter_loop_A, canter_loop_B]},
    "gallop": {"speed": [8.0, np.inf], "loop": [gallop_loop_A, gallop_loop_B]},
}


def speed_to_gait_name(speed: float):
    for gait_name, gait_info in gait_loop_dict.items():
        gait_speed = gait_info["speed"]
        if gait_speed is not None and speed >= gait_speed[0] and speed < gait_speed[1]:
            return gait_name
    return "idle"


def speed_to_gait_index(speed: float):
    for i, (_, gait_info) in enumerate(gait_loop_dict.items()):
        gait_speed = gait_info["speed"]
        if gait_speed is not None and speed >= gait_speed[0] and speed < gait_speed[1]:
            return np.array(i)
    return np.array(0)


def gait_loop_duration_tanh(rwd):
    info = {}

    # get legal gait type
    velocity_control = np.linalg.norm(rwd.env.control_vector[0:2])
    gait_target = speed_to_gait_name(velocity_control)
    info["gait_target"] = gait_target

    # get current feet_state 
    feet_state = get_foot_state(rwd.env)
    info["feet_state"] = bin(feet_state)

    # get/delete gait_loop_options
    if gait_target == rwd.gait_type and len(rwd.gait_loop_options) > 0: # loop continue and has legal loop
        for i in range(len(rwd.gait_loop_options) - 1, -1, -1):
            gait_loop_option = rwd.gait_loop_options[i]
            gait_allowed_steps = gait_loop_option[0]["step"]
            while gait_allowed_steps >= 0:
                if gait_loop_option[0]["state"] == feet_state: # feet_state matched
                    break
                gait_loop_option.append(gait_loop_option.popleft())
                gait_allowed_steps -= 1
            else:
                rwd.gait_loop_options.pop(i) # delete illegal loop 
    else: # loop change or loop continue but hasn't legal loop
        rwd.gait_type = gait_target
        # get new gait_loop_options
        rwd.gait_loop_options = []
        gait_info = gait_loop_dict[gait_target]
        for gait_loop in gait_info["loop"]: # filt legal loop and add to gait_loop_options
            for i, gait_loop_item in enumerate(gait_loop):
                if feet_state == gait_loop_item["state"]:
                    rwd.gait_loop_options.append(deque(gait_loop[i:] + gait_loop[:i],
                                                       maxlen=len(gait_loop)))

    # get next gait_loop_option and update gait_loop_duration
    if len(rwd.gait_loop_options) > 0: # have legal loop
        next_gait_option = [gait_loop_option[i]["state"]
                            for gait_loop_option in rwd.gait_loop_options
                            for i in range(gait_loop_option[0]["step"] + 1)]
        rwd.gait_loop_duration += 1
        info["in_gait_loop"] = True
    else: # it isn't in a legal loop
        next_gait_option = []
        rwd.gait_loop_duration = 0
        info["in_gait_loop"] = False
    info["next_gait_option"] = next_gait_option
    info["gait_loop_duration"] = rwd.gait_loop_duration

    # calculate reward
    gait_loop_duration_tanh = np.tanh(rwd.gait_loop_k * rwd.gait_loop_duration)
    info["gait_loop_duration_tanh"] = gait_loop_duration_tanh

    return gait_loop_duration_tanh, info


def gait_loop_duration_tanh_mode_weighted(rwd):
    from src.env.base import mode as mode_list
    info = {}

    # get observation and weight of mode
    mode_obs = rwd.env._decontruct_obs(rwd.env.obs)["mode_vector"]
    mode_weight = {mode_list[i]: weight for i, weight in enumerate(mode_obs)}
    assert len(mode_weight) == len(mode_list)

    # get current feet_state 
    feet_state = get_foot_state(rwd.env)
    info["feet_state"] = bin(feet_state)

    for mode in mode_list:
        rwd.gait_loop_options[mode] = rwd.gait_loop_options.get(mode, [])

        # get/delete gait_loop_options
        if len(rwd.gait_loop_options[mode]) > 0: # loop continue and has legal loop
            for i in range(len(rwd.gait_loop_options[mode]) - 1, -1, -1):
                gait_loop_option = rwd.gait_loop_options[mode][i]
                gait_allowed_steps = gait_loop_option[0]["step"]
                while gait_allowed_steps >= 0:
                    if gait_loop_option[0]["state"] == feet_state: # feet_state matched
                        break
                    gait_loop_option.append(gait_loop_option.popleft())
                    gait_allowed_steps -= 1
                else:
                    rwd.gait_loop_options[mode].pop(i) # delete illegal loop 
        else: # loop change or loop continue but hasn't legal loop
            for gait_loop in gait_loop_dict[mode]["loop"]: # filt legal loop and add to gait_loop_options
                for i, gait_loop_item in enumerate(gait_loop):
                    if feet_state == gait_loop_item["state"]:
                        rwd.gait_loop_options[mode].append(deque(gait_loop[i:] + gait_loop[:i],
                                                                 maxlen=len(gait_loop)))
        
        # refresh gait_loop_duration
        if len(rwd.gait_loop_options[mode]) > 0: # have legal loop
            rwd.mode_duration[mode] = rwd.mode_duration.get(mode, 0) + 1
        else: # it isn't in a legal loop
            rwd.mode_duration[mode] = 0

    # calculate and store information about the most probable gait mode
    propable_mode = mode_list[np.argmax(mode_obs)]
    if rwd.gait_type == propable_mode:
        rwd.gait_loop_duration += 1
    else:
        rwd.gait_type = propable_mode
        rwd.gait_loop_duration = 0
    info["propable_mode"] = propable_mode
    info["in_gait_loop"] = len(rwd.gait_loop_options[propable_mode]) > 0
    info["gait_loop_duration"] = rwd.gait_loop_duration
    info["next_gait_option"] = [gait_loop_option[i]["state"]
                                         for gait_loop_option in rwd.gait_loop_options[propable_mode]
                                         for i in range(gait_loop_option[0]["step"] + 1)]

    # calculate reward
    if propable_mode == "idle":
        gait_loop_duration_tanh_mode_weigted = mode_weight["idle"] * np.tanh(rwd.gait_loop_k * rwd.mode_duration["idle"])
    else:
        gait_loop_duration_tanh_mode_weigted = 0.
        for mode in mode_list:
            if mode == "idle":
                continue
            gait_loop_duration_tanh_mode_weigted += mode_weight[mode] * np.tanh(rwd.gait_loop_k * rwd.mode_duration[mode])
    info["gait_loop_duration_tanh_mode_weigted"] = gait_loop_duration_tanh_mode_weigted

    return gait_loop_duration_tanh_mode_weigted, info