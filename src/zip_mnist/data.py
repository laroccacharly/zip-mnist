from keras.datasets import mnist
import numpy as np
from typing import Tuple

from .zip_np import compress_numpy_array
from .config import get_config

def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("Loaded MNIST data")
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    return x_train, y_train, x_test, y_test

def reshape(X: np.ndarray) -> np.ndarray:
    return X.reshape(-1, 28 * 28)

def normalize(X: np.ndarray) -> np.ndarray:
    max_val = X.max()
    min_val = X.min()
    return (X - min_val) / (max_val - min_val)

def limit_features(X: np.ndarray) -> np.ndarray:
    total_features = X.shape[1]
    max_num_features =  int(total_features * get_config("features_ratio"))
    indices = np.linspace(0, total_features - 1, num=max_num_features, dtype=int)
    return X[:, indices]

def transform_features(X: np.ndarray) -> np.ndarray:
    X = reshape(X)
    if get_config("enable_compression"):
        X = compress_numpy_array(X)
    if get_config("enable_normalization"):
        X = normalize(X)
    X = limit_features(X)
    return X