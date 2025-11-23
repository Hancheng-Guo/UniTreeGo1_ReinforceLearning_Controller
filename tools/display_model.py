import stable_baselines3
from config import CONFIG


def extract_gym_env(env):
    if isinstance(env, stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv):
        return env.envs[env.num_envs - 1].env
    else: # gymnasium.wrappers.common.TimeLimit
        return env
    

def print_state_space(model, skipped_qpos):
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
            for j in range(4):
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
    mujoco_model = env.unwrapped.model
    skipped_qpos = env.unwrapped.observation_structure['skipped_qpos']
    print_state_space(mujoco_model, skipped_qpos)


def display_body(env):
    model = env.unwrapped.model
    n_fill = 0
    for i in range(model.nbody):
        bname = model.body(i).name
        n_fill = max(len(bname), n_fill)
    print("\n")
    print("=== BODY ===")
    for i in range(model.nbody):
        print("[%3d] %s > pos: %s" % (i, model.body(i).name.rjust(n_fill), ["%+.4f" % x for x in model.body(i).pos]))


def display_model(env):
    env = extract_gym_env(env)
    if CONFIG["is"]["model_state_def_visiable"]:
        display_state(env)
    if CONFIG["is"]["model_body_part_visiable"]:
        display_body(env)
        

        

        

