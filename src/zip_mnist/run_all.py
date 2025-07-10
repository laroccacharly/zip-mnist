from .config import Config
from .xgb import run_xgb

def run_all():
    baseline = Config() 
    linear_sparsity_config = Config(name="linear_sparsity", features_ratio=0.5)
    zip_config = Config(name="zip", enable_compression=True)
    pca_config = Config(name="pca", enable_pca=True)
    autoencoder_config = Config(name="autoencoder", enable_autoencoder=True)

    configs = [
        baseline,
        linear_sparsity_config,
        zip_config,
        pca_config,
        autoencoder_config,
    ]

    for config in configs:
        run_xgb(config)

if __name__ == "__main__":
    run_all()