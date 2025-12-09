import subprocess
import threading
from dataclasses import dataclass
from typing import Optional, List
from src.config.config import CONFIG


class ThreadTensorBoard(): 
    def __init__(self):
        self.thread = None
        self.process = None

    def _reader(self, pipe):
        with pipe:
            for line in iter(pipe.readline, b""):
                print("[TensorBoard]", line.decode().rstrip())

    def _run(self):
        process = subprocess.Popen(
            "tensorboard --logdir " + CONFIG["path"]["output"] + " --bind_all",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            )
        threading.Thread(target=self._reader, args=(process.stdout,), daemon=True).start()
        threading.Thread(target=self._reader, args=(process.stderr,), daemon=True).start()
        self.process = process

    def run(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
        self.thread = None
        self.process = None
        print("TensorBoard stopped")
