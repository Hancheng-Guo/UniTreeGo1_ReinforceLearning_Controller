import numpy as np


def action_change_l2(rwd):
    action_change = (rwd.env.envdata.previous_action_deque[-1]
                     - rwd.env.envdata.previous_action_deque[-2])
    action_change_l2 = np.mean(np.square(action_change))

    info = {
        "action_change_l2": action_change_l2
    }
    return action_change_l2, info