import subprocess
import threading
from config import CONFIG

def run_tensorboard():
    result = subprocess.run("tensorboard --logdir " + CONFIG["path"]["tensorboard"], shell=True)
    print(result.stdout)

def init_tensorboard():
    t = threading.Thread(target=run_tensorboard)
    t.start()
