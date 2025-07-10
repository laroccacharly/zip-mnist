from pydantic import BaseModel

class Config(BaseModel):
    name: str = "baseline"
    time_limit_seconds: float = 10
    features_ratio: float = 1
    enable_compression: bool = False
    num_features_after_compression: int = 400
    enable_normalization: bool = True
    enable_pca: bool = False
    pca_n_components: int = 400
    enable_autoencoder: bool = False
    autoencoder_encoding_dim: int = 400
    save_to_db: bool = True

_config = Config()

def get_config(key: str = None):
    global _config
    if key:
        return getattr(_config, key)
    return _config

def set_config(config: Config):
    global _config
    _config = config
    print(f"Config: {_config.model_dump()}")
