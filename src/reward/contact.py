def illegal_contact_l1(rwd):
    illegal_contact_l1 = 0.
    for c in rwd.env.data.contact:
        illegal_touching = (
            (c.geom1 not in rwd.env._foot_ids and c.geom2 == rwd.env._floor_id) or
            (c.geom1 == rwd.env._floor_id and c.geom2 not in rwd.env._foot_ids))
        illegal_contact_l1 += 1 if illegal_touching else 0
    
    info = {
        "illegal_contact_l1": illegal_contact_l1
    }
    return illegal_contact_l1, info