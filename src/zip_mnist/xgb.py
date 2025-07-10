import xgboost as xgb
import numpy as np
import time
from sklearn.metrics import accuracy_score
from .data import load_mnist, transform_features
from .xgb_timeout import TimeoutCallback
from .schema import Job, insert_job
from sklearn.decomposition import PCA
from .config import get_config

def get_xgb_callbacks(): 
    callbacks = []
    callbacks.append(TimeoutCallback())
    return callbacks

def fit_pca(X: np.ndarray) -> PCA:
    pca = PCA(n_components=get_config("pca_n_components"))
    pca.fit(X)
    return pca 

def run_xgb(): 
    X_train, y_train, X_test, y_test = load_mnist()
    X_train = transform_features(X_train)
    X_test = transform_features(X_test)

    if get_config("enable_pca"):
        pca = fit_pca(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

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
        callbacks=get_xgb_callbacks()
    )

    total_training_time = time.time() - total_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Train accuracy
    y_pred = model.predict(dtrain)
    train_accuracy = accuracy_score(y_train, y_pred)
    print(f'Train Accuracy: {train_accuracy * 100:.2f}%')

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
