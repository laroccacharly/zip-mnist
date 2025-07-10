from keras import models, layers, optimizers, callbacks
import numpy as np
from .config import get_config

def build_autoencoder(input_dim: int) -> tuple[models.Model, models.Model]:
    encoding_dim = get_config("autoencoder_encoding_dim")

    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim * 2, activation='relu')(input_layer)  # Intermediate layer
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)  # Bottleneck

    # Decoder
    decoded = layers.Dense(encoding_dim * 2, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)  # Output matches input (0-1 normalized)

    # Full autoencoder
    autoencoder = models.Model(input_layer, decoded)

    # Separate encoder model
    encoder = models.Model(input_layer, encoded)

    autoencoder.compile(
        loss="mean_squared_error",  # For reconstruction of continuous values
        optimizer=optimizers.Adam(learning_rate=0.001),
        metrics=["mse"]
    )

    return autoencoder, encoder

def fit_autoencoder(X: np.ndarray) -> models.Model:
    print(f"Fitting autoencoder with {X.shape[1]} features")
    autoencoder, encoder = build_autoencoder(X.shape[1])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    _history = autoencoder.fit(
        X, X, 
        epochs=10,  
        batch_size=128,  
        shuffle=True,
        validation_split=0.2,  
        callbacks=[early_stopping],
        verbose=1  
    )

    return encoder

def encode(encoder: models.Model, X: np.ndarray) -> np.ndarray:
    print(f"Encoding {X.shape[1]} features")
    return encoder.predict(X) 

