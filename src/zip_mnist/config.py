from pydantic import BaseModel

class Config(BaseModel):
    time_limit_seconds: float = 10
    features_ratio: float = 1
    enable_compression: bool = False
    num_features_after_compression: int = 500
    enable_normalization: bool = True
    enable_pca: bool = True
    pca_n_components: int = 400

_config = Config()
print(f"Config initialized: {_config.model_dump()}")

def get_config(key: str = None):
    if key:
        return getattr(_config, key)
    return _config