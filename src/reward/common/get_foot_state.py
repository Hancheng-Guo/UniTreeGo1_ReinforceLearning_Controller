def are_foot_touching_ground(env):
    are_touching = []
    for foot_id in env._foot_ids:
        is_touching = False
        for c in env.data.contact:
            is_touching = ((c.geom1 == foot_id and c.geom2 == env._floor_id) or
                            (c.geom1 == env._floor_id and c.geom2 == foot_id))
            if is_touching:
                break
        are_touching.append(is_touching)
    return are_touching


def get_foot_state(env):
    _are_foot_touching_ground = are_foot_touching_ground(env)
    n = len(_are_foot_touching_ground)
    return sum(int(b) << (n - 1 - i) for i, b in enumerate(_are_foot_touching_ground))