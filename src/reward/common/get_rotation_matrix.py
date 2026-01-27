import mujoco
import numpy as np


def get_rotation_matrix(quaternion):
    mat = np.zeros((9, 1)) # R00 R01 R02 R10 R11 R12 R20 R21 R22
    mujoco.mju_quat2Mat(mat, quaternion) # Convert quaternion to 3D rotation matrix
    return mat.reshape(3, 3)