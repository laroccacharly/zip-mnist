import xgboost as xgb
import numpy as np
import time
from sklearn.metrics import accuracy_score
from .data import load_mnist, transform_features
from .xgb_timeout import TimeoutCallback
from .schema import Job, insert_job
from sklearn.decomposition import PCA
from .autoencoder import fit_autoencoder, encode
from .config import get_config, set_config, Config 

def get_xgb_callbacks(): 
    callbacks = []
    callbacks.append(TimeoutCallback())
    return callbacks

def fit_pca(X: np.ndarray) -> PCA:
    pca = PCA(n_components=get_config("pca_n_components"))
    pca.fit(X)
    return pca 

def run_xgb(config: Config = Config()): 
    set_config(config)
    time_start = time.time()

    X_train, y_train, X_test, y_test = load_mnist()
    X_train = transform_features(X_train)
    X_test = transform_features(X_test)

    if get_config("enable_pca"):
        pca = fit_pca(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

    if get_config("enable_autoencoder"):
        encoder = fit_autoencoder(X_train)
        X_train = encode(encoder, X_train)
        X_test = encode(encoder, X_test)

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

    time_before_training = time.time()
    model = xgb.train(
        params,
        dtrain, 
        callbacks=get_xgb_callbacks()
    )

    time_end = time.time()
    time_training = time_end - time_before_training
    time_total = time_end - time_start
    print(f"Total training time: {time_training:.2f} seconds")

    # Train accuracy
    y_pred = model.predict(dtrain)
    train_accuracy = accuracy_score(y_train, y_pred)
    print(f'Train Accuracy: {train_accuracy * 100:.2f}%')

    y_pred = model.predict(dtest)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    job = Job(
        name=get_config("name"),
        test_accuracy=accuracy,
        train_accuracy=train_accuracy,
        total_training_time=time_training,
        total_time=time_total
    )
    print(job.model_dump_json())
    if get_config("save_to_db"):
        insert_job(job)

if __name__ == "__main__":
    run_xgb()
