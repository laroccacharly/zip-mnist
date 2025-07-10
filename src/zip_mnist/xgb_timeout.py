import xgboost as xgb
import time
from .config import get_config

class TimeoutCallback(xgb.callback.TrainingCallback):
    def __init__(self):
        self.start_time = time.time()
        self.time_limit_seconds = get_config("time_limit_seconds")

    def after_iteration(self, model, epoch, evals_log):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.time_limit_seconds:
            print(f"Time limit of {self.time_limit_seconds} seconds reached. Stopping training.")
            return True
        return False