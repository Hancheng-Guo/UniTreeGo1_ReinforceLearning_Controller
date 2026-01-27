import numpy as np


def hinge_angular_velocity_l2(rwd):
    hinge_angular_velocity = rwd.env.state_vector()[25:37]
    hinge_angular_velocity_l2 = np.mean(np.square(hinge_angular_velocity))

    info = {
        "hinge_angular_velocity_l2": hinge_angular_velocity_l2
    }
    return hinge_angular_velocity_l2, info


def hinge_position_l2(rwd):
    hinge_position = rwd.env.state_vector()[7:19]
    hinge_position_l2 = np.mean(np.square(
        (hinge_position - rwd.hinge_position0) / rwd.hinge_position_std))

    info = {
        "hinge_position_l2": hinge_position_l2
    }
    return hinge_position_l2, info


def hinge_exceed_limit_l1(rwd):
    hinge_position = rwd.env.state_vector()[7:19]
    hinge_exceed_upper_limit = np.clip(hinge_position - rwd.hinge_upper_limit, 0., None) 
    hinge_exceed_lower_limit = np.clip(rwd.hinge_lower_limit - hinge_position, 0., None)
    hinge_exceed_limit_l1 = np.sum([hinge_exceed_upper_limit, hinge_exceed_lower_limit])

    info = {
        "hinge_exceed_upper_limit": hinge_exceed_upper_limit,
        "hinge_exceed_lower_limit": hinge_exceed_lower_limit,
        "hinge_exceed_limit_l1": hinge_exceed_limit_l1
    }
    return hinge_exceed_limit_l1, info


def hinge_energy_l1(rwd):
    hinge_force = rwd.env.data.qfrc_actuator[6:19]
    hinge_angular_velocity = rwd.env.state_vector()[25:37]
    hinge_energy_l1 = np.sum(np.abs(hinge_force * hinge_angular_velocity))
    
    info = {
        "hinge_energy_l1": hinge_energy_l1
    }
    return hinge_energy_l1, info