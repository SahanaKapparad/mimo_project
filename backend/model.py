# ============================================================
# model.py — CNN model definition + training function
# ============================================================

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, Dense, Flatten,
                                     BatchNormalization, Dropout, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ReduceLROnPlateau, EarlyStopping,
                                        LambdaCallback)

from simulation import (generate_mimo_channel, add_noise, ls_interpolate,
                        channel_to_features, pilot_idx)


def build_cnn(n_fft, n_rx, n_tx, epochs=40, batch_size=128):
    feat_dim = 2 * n_rx * n_tx
    out_dim  = n_fft * feat_dim

    inp = Input(shape=(n_fft, feat_dim, 1), name='ls_input')
    x   = Conv2D(64,  (3, 1), padding='same', activation='relu')(inp)
    x   = BatchNormalization()(x)
    x   = Conv2D(128, (3, 1), padding='same', activation='relu')(x)
    x   = BatchNormalization()(x)
    x   = Conv2D(64,  (3, 1), padding='same', activation='relu')(x)
    x   = BatchNormalization()(x)
    x   = Flatten()(x)
    x   = Dense(512, activation='relu')(x)
    x   = Dropout(0.2)(x)
    x   = Dense(256, activation='relu')(x)
    out = Dense(out_dim, name='channel_output')(x)
    return Model(inputs=inp, outputs=out)


def train_model(num_samples, n_fft, n_tx, n_rx, l_paths,
                snr_range, epochs, progress_callback=None):
    """
    Full training pipeline.
    progress_callback(percent: int, message: str) — optional live updates.
    Returns (trained_model, history).
    """

    def log(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)
        print(msg)

    log(5, f"Generating {num_samples} channel samples…")
    H_actual = generate_mimo_channel(num_samples, n_fft, n_tx, n_rx, l_paths)
    rand_snr = np.random.choice(snr_range, size=num_samples)

    H_ls_full = np.zeros_like(H_actual)
    for snr in snr_range:
        idx = np.where(rand_snr == snr)[0]
        if len(idx) == 0:
            continue
        H_p  = H_actual[idx][:, [i for i in range(0, n_fft, n_fft // 16)], :, :]
        H_pn = add_noise(H_p, snr)
        H_ls_full[idx] = ls_interpolate(H_pn, n_fft, pilot_idx, n_tx, n_rx)

    log(20, "Preparing features…")
    X_raw = channel_to_features(H_ls_full)
    Y_raw = channel_to_features(H_actual).reshape(num_samples, -1)

    Xm    = X_raw.mean(axis=(1,2,3), keepdims=True)
    Xs    = X_raw.std(axis=(1,2,3),  keepdims=True) + 1e-8
    X_norm = (X_raw - Xm) / Xs

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_norm, Y_raw, test_size=0.2, random_state=42
    )

    log(30, "Building CNN…")
    model = build_cnn(n_fft, n_rx, n_tx)
    model.compile(optimizer=Adam(1e-3), loss='mean_squared_error', metrics=['mean_absolute_error'])

    epoch_counter = [0]
    def on_epoch_end(epoch, logs):
        epoch_counter[0] += 1
        pct = 30 + int(60 * epoch_counter[0] / epochs)
        log(pct, f"Epoch {epoch+1}/{epochs} — loss: {logs['loss']:.5f} "
                 f"val_loss: {logs.get('val_loss', 0):.5f}")

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-6, verbose=0),
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True, verbose=0),
        LambdaCallback(on_epoch_end=on_epoch_end),
    ]

    log(30, "Training started…")
    history = model.fit(
        X_train, Y_train,
        epochs=epochs, batch_size=128,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=0,
    )

    loss, mae = model.evaluate(X_test, Y_test, verbose=0)
    log(100, f"Training done — Test MSE: {loss:.6f}  MAE: {mae:.6f}")
    return model, history