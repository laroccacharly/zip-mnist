import xgboost as xgb
from typing import Dict, Any, Tuple, List
import time
from sklearn.metrics import accuracy_score
from .data import load_mnist, transform_features
from .xgb_timeout import TimeoutCallback
from pydantic import BaseModel
from .schema import Job, insert_job

class XGBoostConfig(BaseModel):
    max_num_features: int = 100
    learning_rate: float = 0.3
    num_rounds: int = 100
    time_limit_seconds: float = 10

def get_xgb_callbacks(): 
    callbacks = []
    callbacks.append(TimeoutCallback())
    return callbacks

def run_xgb(): 
    X_train, y_train, X_test, y_test = load_mnist()
    X_train = transform_features(X_train)
    X_test = transform_features(X_test)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
            'objective': 'multi:softmax',
            'num_class': 10,
            'eval_metric': 'merror',
            # 'n_jobs': -1,
            # 'eta': 0.3,
    }

    total_time = time.time()
    model = xgb.train(
        params,
        dtrain, 
        # 100,
        callbacks=get_xgb_callbacks()
    )

    total_training_time = time.time() - total_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    y_pred = model.predict(dtest)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    job = Job(
        model_name="xgb",
        test_accuracy=accuracy,
        total_training_time=total_training_time
    )
    insert_job(job)

if __name__ == "__main__":
    run_xgb()
