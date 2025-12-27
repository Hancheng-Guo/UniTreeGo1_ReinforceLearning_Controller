import numpy as np
from collections import deque

from src.reward.common.get_foot_state import get_foot_state


# x_velocity_control = 0
idle_loop = [0b1111]
# # x_velocity_control ∈ (0, 2]
# walk_loop = [0b1110, 0b1010, 0b1011, 0b1101, 0b0101, 0b0111]
# x_velocity_control ∈ (0, 6]
trot_loop = [0b1001, 0b0110]
# x_velocity_control ∈ (6, 8]
canter_loop_A = [0b1110, 0b1000, 0b0000, 0b0001, 0b0111, 0b0110]
canter_loop_B = [0b1101, 0b0100, 0b0000, 0b0010, 0b1011, 0b1001]
# x_velocity_control ∈ (8, inf)
gallop_loop_A = [0b1000, 0b1100, 0b0100, 0b0000, 0b0010, 0b0011, 0b0001, 0b0000]
gallop_loop_B = [0b0100, 0b1100, 0b1000, 0b0000, 0b0001, 0b0011, 0b0010, 0b0000]


gait_loop_dict = {
    "idle": [idle_loop],
    "trot": [trot_loop],
    "canter": [canter_loop_A, canter_loop_B],
    "gallop": [gallop_loop_A, gallop_loop_B],
}

in_gait_loop = None


def gait_loop_duration_tanh(rwd):
    info = {}

    # get legal gait type
    velocity_control = np.linalg.norm(rwd.env.control_vector[0:2])
    if velocity_control > 8:
        gait_target = "gallop"
    elif velocity_control > 6:
        gait_target = "canter"
    elif velocity_control > 0:
        gait_target = "trot"
    else:
        gait_target = "idle"
    info["gait_target"] = gait_target

    # get current feet_state 
    feet_state = get_foot_state(rwd.env)
    info["feet_state"] = bin(feet_state)

    # get/delete gait_loop_options
    if gait_target == rwd.gait_type and len(rwd.gait_loop_options) > 0: # loop continue and has legal loop
        for i in range(len(rwd.gait_loop_options) - 1, -1, -1):
            gait_loop_option = rwd.gait_loop_options[i]
            if gait_loop_option[0] == feet_state: # feet_state continue
                continue
            elif gait_loop_option[1 % len(gait_loop_option)] == feet_state: # feet_state go on
                gait_loop_option.append(gait_loop_option.popleft())
            else:
                rwd.gait_loop_options.pop(i) # delete illegal loop 
    else: # loop change or loop continue but hasn't legal loop
        rwd.gait_type = gait_target
        # get new gait_loop_options
        rwd.gait_loop_options = []
        for gait_loop in gait_loop_dict[gait_target]: # filt legal loop and add to gait_loop_options
            for i, gait_loop_state in enumerate(gait_loop):
                if feet_state == gait_loop_state:
                    rwd.gait_loop_options.append(deque(gait_loop[i:] + gait_loop[:i],
                                                       maxlen=len(gait_loop)))

    # get next gait_loop_option and update gait_loop_duration
    if len(rwd.gait_loop_options) > 0: # have legal loop
        next_gait_option = [gait_loop_option[1 % len(gait_loop_option)]
                            for gait_loop_option in rwd.gait_loop_options]
        rwd.gait_loop_duration += 1
        info["in_gait_loop"] = True
        in_gait_loop = True
    else: # it isn't in a legal loop
        next_gait_option = []
        rwd.gait_loop_duration = 0
        info["in_gait_loop"] = False
        in_gait_loop = False
    info["next_gait_option"] = next_gait_option
    info["gait_loop_duration"] = rwd.gait_loop_duration

    # calculate reward
    gait_loop_duration_tanh = np.tanh(rwd.gait_loop_k * rwd.gait_loop_duration)
    info["gait_loop_duration_tanh"] = gait_loop_duration_tanh

    return gait_loop_duration_tanh, info