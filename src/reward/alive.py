import numpy as np


fatal_contact = ["trunk", "FR_hip", "FL_hip", "RR_hip", "RL_hip"]


def is_alive(rwd):
    if not np.isfinite(rwd.env.state_vector()).all():
        return False, {"is_alive": False}
    
    for c in rwd.env.data.contact:
        body1 = rwd.env.model.body(rwd.env.model.geom_bodyid[c.geom1]).name
        body2 = rwd.env.model.body(rwd.env.model.geom_bodyid[c.geom2]).name
        fatal_touching = ((body1 == "world" and body2 in fatal_contact) or
                          (body1 in fatal_contact and body2 == "world"))
        if fatal_touching:
            return False, {"is_alive": False}
    
    return True, {"is_alive": True}