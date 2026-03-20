import numpy as np
from collections import deque


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
    "idle":   {"speed": [0.0, 1e-7],   "loop": [idle_loop]},
    "walk":   {"speed": None,          "loop": [walk_loop]},
    "pace":   {"speed": None,          "loop": [pace_loop]},
    "trot":   {"speed": [1e-7, 5.0],   "loop": [trot_loop]},
    "canter": {"speed": [5.0, 8.0],    "loop": [canter_loop_A, canter_loop_B]},
    "gallop": {"speed": [8.0, np.inf], "loop": [gallop_loop_A, gallop_loop_B]},
}


def command_to_gait_name(command: np.array):
    if command.size != 0:
        command_l2 = np.linalg.norm(command)
        for gait_name, gait_info in gait_loop_dict.items():
            gait_speed = gait_info["speed"]
            if gait_speed is None:
                continue
            if command_l2 >= gait_speed[0] and command_l2 < gait_speed[1]:
                return gait_name
    return "idle"


# def speed_to_gait_index(speed: float):
#     for i, (_, gait_info) in enumerate(gait_loop_dict.items()):
#         gait_speed = gait_info["speed"]
#         if gait_speed is not None and speed >= gait_speed[0] and speed < gait_speed[1]:
#             return np.array(i)
#     return np.array(0)


def gait_loop_duration_tanh(rwd, gait_target:str=None):
    info = {}

    # get legal gait type
    if gait_target is None:
        gait_target = command_to_gait_name(rwd.env.envdata.previer_cmd_vec)
    info["gait_target"] = gait_target

    # get current foot_state 
    foot_state = rwd.env.envdata.foot_state
    info["foot_state"] = bin(foot_state)

    # get/delete gait_loop_options
    if gait_target == rwd.gait_type and len(rwd.gait_loop_options) > 0: # loop continue and has legal loop
        for i in range(len(rwd.gait_loop_options) - 1, -1, -1):
            gait_loop_option = rwd.gait_loop_options[i]
            gait_allowed_steps = gait_loop_option[0]["step"]
            while gait_allowed_steps >= 0:
                if gait_loop_option[0]["state"] == foot_state: # foot_state matched
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
                if foot_state == gait_loop_item["state"]:
                    rwd.gait_loop_options.append(deque(gait_loop[i:] + gait_loop[:i],
                                                       maxlen=len(gait_loop)))

    # get next gait_loop_option and update gait_loop_duration
    if len(rwd.gait_loop_options) > 0: # have legal loop
        next_gait_option = [gait_loop_option[i]["state"]
                            for gait_loop_option in rwd.gait_loop_options
                            for i in range(gait_loop_option[0]["step"] + 1)]
        rwd.gait_loop_duration += rwd.env.dt
        info["in_gait_loop"] = True
    else: # it isn't in a legal loop
        next_gait_option = []
        rwd.gait_loop_duration = 0.
        info["in_gait_loop"] = False
    info["next_gait_option"] = next_gait_option
    info["gait_loop_duration"] = rwd.gait_loop_duration

    # calculate reward
    gait_loop_duration_tanh = np.tanh(rwd.gait_loop_k * rwd.gait_loop_duration)
    info["gait_loop_duration_tanh"] = gait_loop_duration_tanh

    return gait_loop_duration_tanh, info


def trot_loop_duration_tanh(rwd):
    gait_target = command_to_gait_name(rwd.env.envdata.previer_cmd_vec)
    gait_target = "idle" if gait_target == "idle" else "trot"
    return gait_loop_duration_tanh(rwd, gait_target)


def gait_loop_duration_tanh_mode_weighted(rwd):
    from src.env.base import mode as mode_list
    info = {}

    # get observation and weight of mode
    mode_obs = rwd.env._decontruct_obs(rwd.env.obs)["mode_vector"]
    mode_weight = {mode_list[i]: weight for i, weight in enumerate(mode_obs)}
    assert len(mode_weight) == len(mode_list)

    # get current foot_state 
    foot_state = rwd.env.envdata.foot_state
    info["foot_state"] = bin(foot_state)

    for mode in mode_list:
        rwd.gait_loop_options[mode] = rwd.gait_loop_options.get(mode, [])

        # get/delete gait_loop_options
        if len(rwd.gait_loop_options[mode]) > 0: # loop continue and has legal loop
            for i in range(len(rwd.gait_loop_options[mode]) - 1, -1, -1):
                gait_loop_option = rwd.gait_loop_options[mode][i]
                gait_allowed_steps = gait_loop_option[0]["step"]
                while gait_allowed_steps >= 0:
                    if gait_loop_option[0]["state"] == foot_state: # foot_state matched
                        break
                    gait_loop_option.append(gait_loop_option.popleft())
                    gait_allowed_steps -= 1
                else:
                    rwd.gait_loop_options[mode].pop(i) # delete illegal loop 
        else: # loop change or loop continue but hasn't legal loop
            for gait_loop in gait_loop_dict[mode]["loop"]: # filt legal loop and add to gait_loop_options
                for i, gait_loop_item in enumerate(gait_loop):
                    if foot_state == gait_loop_item["state"]:
                        rwd.gait_loop_options[mode].append(deque(gait_loop[i:] + gait_loop[:i],
                                                                 maxlen=len(gait_loop)))
        
        # refresh gait_loop_duration
        if len(rwd.gait_loop_options[mode]) > 0: # have legal loop
            rwd.mode_duration[mode] = rwd.mode_duration.get(mode, 0) + rwd.env.dt
        else: # it isn't in a legal loop
            rwd.mode_duration[mode] = 0.

    # calculate and store information about the most probable gait mode
    target_mode = mode_list[np.argmax(mode_obs)]
    if len(rwd.gait_loop_options[target_mode]) > 0:
        rwd.gait_loop_duration += rwd.env.dt
    else:
        rwd.gait_type = target_mode
        rwd.gait_loop_duration = 0.
    info["target_mode"] = target_mode
    info["in_gait_loop"] = len(rwd.gait_loop_options[target_mode]) > 0
    info["gait_loop_duration"] = rwd.gait_loop_duration
    info["next_gait_option"] = [gait_loop_option[i]["state"]
                                         for gait_loop_option in rwd.gait_loop_options[target_mode]
                                         for i in range(gait_loop_option[0]["step"] + 1)]

    # calculate reward
    if target_mode == "idle":
        gait_loop_duration_tanh_mode_weigted = mode_weight["idle"] * np.tanh(rwd.gait_loop_k * rwd.mode_duration["idle"])
    else:
        gait_loop_duration_tanh_mode_weigted = 0.
        for mode in mode_list:
            if mode == "idle":
                continue
            gait_loop_duration_tanh_mode_weigted += mode_weight[mode] * np.tanh(rwd.gait_loop_k * rwd.mode_duration[mode])
    info["gait_loop_duration_tanh_mode_weigted"] = gait_loop_duration_tanh_mode_weigted

    return gait_loop_duration_tanh_mode_weigted, info


def gait_transfer(rwd, gait_target:str=None):
    # get legal gait type
    if gait_target is None:
        gait_target = command_to_gait_name(rwd.env.envdata.previer_cmd_vec)

    # get current foot_state 
    foot_state = rwd.env.envdata.foot_state

    # get/delete gait_loop_options
    if gait_target == rwd.gait_type and len(rwd.gait_loop_options) > 0: # loop continue and has legal loop
        for i in range(len(rwd.gait_loop_options) - 1, -1, -1):
            gait_loop_option = rwd.gait_loop_options[i]
            gait_allowed_steps = gait_loop_option[0]["step"]
            while gait_allowed_steps >= 0:
                if gait_loop_option[0]["state"] == foot_state: # foot_state matched
                    break
                gait_allowed_steps -= 1
            else:
                rwd.gait_loop_options.pop(i) # delete illegal loop
    else: # gait loop change or loop continue but hasn't legal loop
        # get new gait_loop_options
        rwd.gait_loop_options = []
        rwd.gait_loop_duration = 0.
        gait_info = gait_loop_dict[gait_target]
        for gait_loop in gait_info["loop"]: # filt legal loop and add to gait_loop_options
            for i, gait_loop_item in enumerate(gait_loop):
                if foot_state == gait_loop_item["state"]:
                    rwd.gait_loop_options.append(deque(gait_loop[i:] + gait_loop[:i],
                                                       maxlen=len(gait_loop)))
                    
    # update rwd
    if len(rwd.gait_loop_options) > 0:
        rwd.gait_type = gait_target
        rwd.gait_loop_duration += rwd.env.dt
    else:
        rwd.gait_type = None
        rwd.gait_loop_duration = 0.
        
    # get next gait_loop_option
    foot_state_target_list = []
    if len(rwd.gait_loop_options) > 0: # have legal loop
        for gait_loop_option in rwd.gait_loop_options:
            foot_state_target_list.append(gait_loop_option[gait_loop_option[0]["step"]]["state"])
    else: # it isn't in a legal loop
        foot_state_target_same_len = -np.inf
        for gait_loop_option in gait_loop_dict[gait_target]["loop"]:
            for gait_loop_item in gait_loop_option:
                same_len = np.sum([(gait_loop_item["state"] & foot_state >> i) & 1
                                   for i in range(len(rwd.env.envdata.foot_landed_time)-1, -1, -1)])
                if same_len > foot_state_target_same_len:
                    foot_state_target_list = []
                    foot_state_target_list.append(gait_loop_item["state"])
                    foot_state_target_same_len = same_len
                elif same_len == foot_state_target_same_len:
                    foot_state_target_list.append(gait_loop_item["state"])
    

    are_foot_landed = (rwd.env.envdata.foot_landed_time > 0)
    are_foot_lifted = (rwd.env.envdata.foot_lifted_time > 0)
    foot_landed_time = rwd.env.envdata.foot_landed_time
    foot_lifted_time = rwd.env.envdata.foot_lifted_time
    foot_contract_fz = np.clip(rwd.env.envdata.foot_fz, 0, np.inf)
    foot_lift_height = np.clip(rwd.env.data.geom_xpos[rwd.env._foot_ids][:,2], 0, np.inf)
    info_list = []
    gait_transfer_indicator_list = []

    for foot_state_target_tmp in foot_state_target_list:
        are_foot_going_to_land = np.array([(foot_state_target_tmp >> i) & 1
                                        for i in range(len(rwd.env.envdata.foot_landed_time)-1, -1, -1)]) > 0
        are_foot_going_to_lift = np.logical_not(are_foot_going_to_land)
    
        keep_lifted_indicator = (are_foot_going_to_lift * are_foot_lifted *
                                 np.tanh(rwd.gait_loop_k * foot_lifted_time))
        keep_landed_indicator = (are_foot_going_to_land * are_foot_landed *
                                 np.tanh(rwd.gait_loop_k * foot_landed_time))
        encourage_lifting_indicator = (are_foot_going_to_lift * are_foot_landed *
                                       np.exp(-foot_contract_fz / rwd.foot_contract_fz_sigma))
        encourage_landing_indicator = (are_foot_going_to_land * are_foot_lifted *
                                       np.exp(-foot_lift_height / rwd.foot_lift_height_sigma))
        gait_transfer_indicator = np.sum(keep_lifted_indicator + keep_landed_indicator +
                                   encourage_lifting_indicator + encourage_landing_indicator)
        gait_transfer_indicator_list.append(gait_transfer_indicator)
        info_list.append({
            "keep_lifted_indicator": keep_lifted_indicator,
            "keep_landed_indicator": keep_landed_indicator,
            "encourage_lifting_indicator": encourage_lifting_indicator,
            "encourage_landing_indicator": encourage_landing_indicator,
            "gait_transfer_indicator": gait_transfer_indicator
        })

    best_foot_state_target_index = gait_transfer_indicator_list.index(max(gait_transfer_indicator_list))
    foot_state_target = foot_state_target_list[best_foot_state_target_index]
    info = {
        "gait_target": gait_target,
        "gait_loop_duration": rwd.gait_loop_duration,
        "foot_state": f"{foot_state:04b}",
        "foot_state_target": f"{foot_state_target:04b}",
        **info_list[best_foot_state_target_index],
    }

    return gait_transfer_indicator_list[best_foot_state_target_index], info

def sync_reward(rwd, idx_1, idx_2):
    se_land = np.clip((rwd.env.envdata.foot_landed_time[idx_1] - rwd.env.envdata.foot_landed_time[idx_2])**2, 0, 4**2)
    se_lift = np.clip((rwd.env.envdata.foot_lifted_time[idx_1] - rwd.env.envdata.foot_lifted_time[idx_2])**2, 0, 4**2)
    return np.exp(-(se_land + se_lift) / 4**2)

def async_reward(rwd, idx_1, idx_2):
    se_act1 = np.clip((rwd.env.envdata.foot_landed_time[idx_1] - rwd.env.envdata.foot_lifted_time[idx_2])**2, 0, 4**2)
    se_act2 = np.clip((rwd.env.envdata.foot_lifted_time[idx_1] - rwd.env.envdata.foot_landed_time[idx_2])**2, 0, 4**2)
    return np.exp(-(se_act1 + se_act2) / 4**2)
def trot_sync(rwd):
    gait_target = command_to_gait_name(rwd.env.envdata.previer_cmd_vec)

    if gait_target == "idle":
        sync_indicator = sync_reward(rwd,0,3)*sync_reward(rwd,1,2)*sync_reward(rwd,0,2)*sync_reward(rwd,1,3)*sync_reward(rwd,0,1)*sync_reward(rwd,2,3)
    else:
        sync_indicator = sync_reward(rwd,0,3)*sync_reward(rwd,1,2)*async_reward(rwd,0,2)*async_reward(rwd,1,3)*async_reward(rwd,0,1)*async_reward(rwd,2,3)

    return sync_indicator, {}


