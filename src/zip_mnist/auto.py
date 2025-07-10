from .config import Config
from .xgb import run_xgb

if __name__ == "__main__":
    config = Config(name="auto", enable_autoencoder=True, save_to_db=False)
    run_xgb(config)