import numpy as np
import mujoco

def are_foot_touching_ground(env):
    are_touching = []
    for foot_id in env._foot_ids:
        is_touching = False
        for i in range(env.data.ncon):
            c = env.data.contact[i]
            is_match = ((c.geom1 == foot_id and c.geom2 == env._floor_id) or
                        (c.geom1 == env._floor_id and c.geom2 == foot_id))
            if is_match:
                # out = np.zeros(6, dtype=np.float64)
                # mujoco.mj_contactForce(env.model, env.data, i, out)
                # foot_fz = out[2]
                # if foot_fz > 5.0:
                #     is_touching = True
                # break

                is_touching = True
                break
        
        are_touching.append(is_touching)
    return are_touching


def get_foot_state(env):
    _are_foot_touching_ground = are_foot_touching_ground(env)
    n = len(_are_foot_touching_ground)
    return sum(int(b) << (n - 1 - i) for i, b in enumerate(_are_foot_touching_ground))