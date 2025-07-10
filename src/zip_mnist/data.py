from keras.datasets import mnist
import numpy as np
from typing import Tuple

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
    return X / 255.0

def limit_features(X: np.ndarray, features_ratio: float = 0.2) -> np.ndarray:
    total_features = X.shape[1]
    max_num_features =  int(total_features * features_ratio)
    indices = np.linspace(0, total_features - 1, num=max_num_features, dtype=int)
    return X[:, indices]

def transform_features(X: np.ndarray) -> np.ndarray:
    X = reshape(X)
    X = normalize(X)
    X = limit_features(X)
    return X