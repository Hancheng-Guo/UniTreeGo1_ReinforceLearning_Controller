import numpy as np


def action_change_l2(rwd):
    if rwd.env.action_old is None:
        return 0., {"action_change_l2": 0.}
    
    action_change = rwd.env.action - rwd.env.action_old
    action_change_l2 = np.mean(np.square(action_change))

    info = {
        "action_change_l2": action_change_l2
    }
    return action_change_l2, info