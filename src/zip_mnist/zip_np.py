import zlib
import numpy as np
from .config import get_config

def truncate_or_pad_features(X: np.ndarray, target_num_features: int = get_config("num_features_after_compression")) -> np.ndarray:
    current_len = len(X)
    if current_len > target_num_features:
        return X[:target_num_features]
    elif current_len < target_num_features:
        pad_width = target_num_features - current_len
        return np.pad(X, (0, pad_width), mode='constant', constant_values=0)
    else:
        return X

def compress_numpy_array(X: np.ndarray) -> np.ndarray:
    # input shape is (num_samples, 784) for mnist where flattened 28x28 image is 784 features
    output = np.zeros((X.shape[0], get_config("num_features_after_compression")), dtype=np.uint8)
    compressed_bytes_counts = []
    for i in range(X.shape[0]):
        row = X[i]
        compressed_bytes = zlib.compress(row.tobytes())
        compressed_bytes_counts.append(len(compressed_bytes))
        compressed_np = np.frombuffer(compressed_bytes, dtype=np.uint8)
        adjusted_np = truncate_or_pad_features(compressed_np)
        output[i] = adjusted_np
    print(f"Average compressed bytes count: {sum(compressed_bytes_counts) / len(compressed_bytes_counts)}")
    print(f"STD of compressed bytes count: {np.std(compressed_bytes_counts)}")
    print(f"Output shape after compression: {output.shape}")
    return output





