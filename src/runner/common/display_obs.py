import mujoco

def get_qpos(model: mujoco._structs.MjModel, n_fill: int):
    axis_str = ["w","x","y","z"]
    qpos_str = []
    for ijnt in range(model.njnt):
        jname = model.joint(ijnt).name.rjust(n_fill)
        jtype = model.jnt_type[ijnt]
        if jtype == 0:
            qpos_str += ([f"{jname}:  free joint > {axis_str[(j+1) % 4]}-axis coordinate" for j in range(3)] + 
                         [f"{jname}:  free joint > {axis_str[j % 4]}-axis direction" for j in range(4)])
        elif jtype == 1:
            qpos_str += [f"{jname}:  ball joint > {axis_str[j % 4]}-axis rotation angle component" for j in range(4)]
        elif jtype == 2:
            qpos_str += [f"{jname}: slide joint > coordinate along specified axis"]
        elif jtype == 3:
            qpos_str += [f"{jname}: hinge joint > angle of specified direction"]
    return qpos_str


def get_qvel(model: mujoco._structs.MjModel, n_fill: int):
    axis_str = ["w","x","y","z"]
    qvel_str = []
    for ijnt in range(model.njnt):
        jname = model.joint(ijnt).name.rjust(n_fill)
        jtype = model.jnt_type[ijnt]
        if jtype == 0:
            qvel_str += ([f"{jname}:  free joint > {axis_str[(j+1) % 4]}-axis linear velocity" for j in range(3)] +
                         [f"{jname}:  free joint > {axis_str[(j+1) % 4]}-axis angular velocity" for j in range(3)])
        elif jtype == 1:
            qvel_str += [f"{jname}:  ball joint > {axis_str[(j+1) % 4]}-axis angular velocity" for j in range(3)]
        elif jtype == 2:
            qvel_str += [f"{jname}: slide joint > linear velocity along specified axis"]
        elif jtype == 3:
            qvel_str += [f"{jname}: hinge joint > angular velocity of specified direction"]
    return qvel_str
