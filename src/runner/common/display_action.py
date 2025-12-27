import mujoco
from src.runner.common.extract_gym_env import extract_gym_env


def display_action(env):
    env = extract_gym_env(env)
    model = env.unwrapped.model
    print("\n")
    print("=== ACTION ===")
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)

        joint_id = model.actuator_trnid[i][0]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)

        print(f"action[{i}] -> actuator: {act_name}, joint: {joint_name}")