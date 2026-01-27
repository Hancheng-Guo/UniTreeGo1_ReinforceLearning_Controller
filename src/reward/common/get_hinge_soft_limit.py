import numpy as np


def get_hinge_soft_upper_limit(rwd, limit_ratio):
    hinge_upper_bounder = np.array([jnt_range[1] for jnt_range in rwd.env.model.jnt_range[1:]])
    hinge_position0=rwd.env.model.key_qpos[0][7:19]
    hinge_upper_limit = (hinge_upper_bounder - hinge_position0) * limit_ratio + hinge_position0
    return hinge_upper_limit


def get_hinge_soft_lower_limit(rwd, limit_ratio):
    hinge_lower_bounder = np.array([jnt_range[0] for jnt_range in rwd.env.model.jnt_range[1:]])
    hinge_position0=rwd.env.model.key_qpos[0][7:19]
    hinge_lower_limit = hinge_position0 - (hinge_position0 - hinge_lower_bounder) * limit_ratio
    return hinge_lower_limit