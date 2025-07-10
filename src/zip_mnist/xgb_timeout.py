import xgboost as xgb

import time

class TimeoutCallback(xgb.callback.TrainingCallback):
    def __init__(self, time_limit_seconds: float = 10):
        self.start_time = time.time()
        self.time_limit_seconds = time_limit_seconds

    def after_iteration(self, model, epoch, evals_log):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.time_limit_seconds:
            print(f"Time limit of {self.time_limit_seconds} seconds reached. Stopping training.")
            return True
        return False