from tools.extract_gym_env import extract_gym_env

def print_env_list(model, skipped_qpos):
    state_count = 0
    coordinate_str = ["w","x","y","z"]
    n_fill = 6
    for i in range(model.njnt):
        jname = model.joint(i).name
        n_fill = max(len(jname), n_fill)

    print("\n")
    print("=== QPOS ===")
    for i in range(model.njnt):
        jname = model.joint(i).name
        adr = model.jnt_qposadr[i]
        jtype = model.jnt_type[i]
        if jtype == 0:
            for j in range(3):
                if j < skipped_qpos: continue
                print("[%3d] %s:  free joint > %s-axis coordinate" % (state_count, jname.rjust(n_fill), coordinate_str[(j+1) % 4]))
                state_count += 1
            for j in range(4):
                print("[%3d] %s:  free joint > %s-axis direction" % (state_count, jname.rjust(n_fill), coordinate_str[j % 4]))
                state_count += 1
        elif jtype == 1:
            for j in range(3):
                print("[%3d] %s:  ball joint > %s-axis rotation angle component" % (state_count, jname.rjust(n_fill), coordinate_str[j % 4]))
                state_count += 1
        elif jtype == 2:
            print("[%3d] %s: slide joint > coordinate along specified axis" % (state_count, jname.rjust(n_fill)))
            state_count += 1
        elif jtype == 3:
            print("[%3d] %s: hinge joint > angle of specified direction" % (state_count, jname.rjust(n_fill)))
            state_count += 1
        else:
            print("display qpos rror!")

    print("=== QVEL ===")
    for i in range(model.njnt):
        jname = model.joint(i).name
        adr = model.jnt_dofadr[i]
        jtype = model.jnt_type[i]
        if jtype == 0:
            for j in range(3):
                print("[%3d] %s:  free joint > %s-axis linear velocity" % (state_count, jname.rjust(n_fill), coordinate_str[(j+1) % 4]))
                state_count += 1
            for j in range(3):
                print("[%3d] %s:  free joint > %s-axis angular velocity" % (state_count, jname.rjust(n_fill), coordinate_str[(j+1) % 4]))
                state_count += 1
        elif jtype == 1:
            for j in range(3):
                print("[%3d] %s:  ball joint > %s-axis angular velocity" % (state_count, jname.rjust(n_fill), coordinate_str[(j+1) % 4]))
                state_count += 1
        elif jtype == 2:
            print("[%3d] %s: slide joint > linear velocity along specified axis" % (state_count, jname.rjust(n_fill)))
            state_count += 1
        elif jtype == 3:
            print("[%3d] %s: hinge joint > angular velocity of specified direction" % (state_count, jname.rjust(n_fill)))
            state_count += 1
        else:
            print("display qvel error!")


def display_state(env):

    env = extract_gym_env(env)
    mujoco_model = env.unwrapped.model
    skipped_qpos = env.unwrapped.observation_structure['skipped_qpos']
    print_env_list(mujoco_model, skipped_qpos)

        

        

