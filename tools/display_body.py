from tools.extract_gym_env import extract_gym_env


def display_body(env):

    env = extract_gym_env(env)
    model = env.unwrapped.model

    n_fill = 0
    for i in range(model.nbody):
        bname = model.body(i).name
        n_fill = max(len(bname), n_fill)

    print("\n")
    print("=== BODY ===")
    for i in range(model.nbody):
        print("[%3d] %s > pos: %s" % (i, model.body(i).name.rjust(n_fill), ["%+.4f" % x for x in model.body(i).pos]))
