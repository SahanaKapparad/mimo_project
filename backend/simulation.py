# ============================================================
# simulation.py — Channel generation, LS estimation, BER
# Used by main.py (FastAPI backend)
# ============================================================

import numpy as np

# ── Constants ───────────────────────────────────────────────
N_FFT    = 64
N_PILOTS = 16
pilot_spacing = N_FFT // N_PILOTS
pilot_idx     = np.arange(0, N_FFT, pilot_spacing)

QPSK_MAP = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)


# ─────────────────────────────────────────────────────────────
# CHANNEL GENERATION
# ─────────────────────────────────────────────────────────────

def generate_mimo_channel(n_samples, n_fft, n_tx, n_rx, n_paths, decay=1.5):
    """
    Generates MIMO multipath channel with exponential power-delay profile.
    Returns H: (n_samples, n_fft, n_rx, n_tx) complex
    """
    pdp = np.exp(-np.arange(n_paths) / decay)
    pdp /= pdp.sum()

    tap_real = np.random.randn(n_samples, n_paths, n_rx, n_tx)
    tap_imag = np.random.randn(n_samples, n_paths, n_rx, n_tx)
    h_time   = (tap_real + 1j * tap_imag) * np.sqrt(pdp / 2)[None, :, None, None]

    h_time_pad = np.zeros((n_samples, n_fft, n_rx, n_tx), dtype=complex)
    h_time_pad[:, :n_paths, :, :] = h_time
    H_freq = np.fft.fft(h_time_pad, n=n_fft, axis=1)
    return H_freq


# ─────────────────────────────────────────────────────────────
# NOISE + LS ESTIMATION
# ─────────────────────────────────────────────────────────────

def add_noise(H, snr_db):
    """Add complex AWGN scaled to given SNR."""
    snr_lin   = 10 ** (snr_db / 10)
    sig_power = np.mean(np.abs(H) ** 2)
    noise_std = np.sqrt(sig_power / (2 * snr_lin))
    noise     = noise_std * (
        np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape)
    )
    return H + noise


def ls_interpolate(H_pilot_noisy, n_fft, pilot_idx, n_tx, n_rx):
    """
    Linear interpolation from pilot positions to all subcarriers.
    H_pilot_noisy: (S, Np, Nr, Nt) → returns (S, K, Nr, Nt)
    """
    S = H_pilot_noisy.shape[0]
    H_ls = np.zeros((S, n_fft, n_rx, n_tx), dtype=complex)
    for s in range(S):
        for r in range(n_rx):
            for t in range(n_tx):
                vals = H_pilot_noisy[s, :, r, t]
                H_ls[s, :, r, t] = (
                    np.interp(np.arange(n_fft), pilot_idx, np.real(vals))
                    + 1j * np.interp(np.arange(n_fft), pilot_idx, np.imag(vals))
                )
    return H_ls


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def channel_to_features(H):
    """
    H: (S, K, Nr, Nt) complex → (S, K, 2*Nr*Nt, 1) float32
    """
    S, K, Nr, Nt = H.shape
    real = np.real(H).reshape(S, K, Nr * Nt)
    imag = np.imag(H).reshape(S, K, Nr * Nt)
    feat = np.concatenate([real, imag], axis=-1)[:, :, :, np.newaxis]
    return feat.astype(np.float32)


def pred_to_channel(pred_flat, x_mean, x_std, n_fft, n_rx, n_tx):
    """
    Reverse feature engineering: flat prediction → (S, K, Nr, Nt) complex
    """
    S    = pred_flat.shape[0]
    feat = 2 * n_rx * n_tx
    H_f  = pred_flat.reshape(S, n_fft, feat, 1)
    H_f  = H_f * x_std + x_mean
    H_f  = H_f[:, :, :, 0]
    real = H_f[:, :, :n_rx * n_tx].reshape(S, n_fft, n_rx, n_tx)
    imag = H_f[:, :, n_rx * n_tx:].reshape(S, n_fft, n_rx, n_tx)
    return real + 1j * imag


# ─────────────────────────────────────────────────────────────
# BER CALCULATION
# ─────────────────────────────────────────────────────────────

def bits_to_qpsk(bits):
    bits = bits.reshape(-1, 2)
    idx  = bits[:, 0] * 2 + bits[:, 1]
    return QPSK_MAP[idx]


def qpsk_to_bits(syms):
    dists = np.abs(syms[:, None] - QPSK_MAP[None, :]) ** 2
    idx   = np.argmin(dists, axis=1)
    bits  = np.zeros((len(idx), 2), dtype=int)
    bits[:, 0] = idx // 2
    bits[:, 1] = idx % 2
    return bits.flatten()


def zf_equalise(Y_rx, H_est):
    """Zero-forcing equaliser per subcarrier."""
    K, Nr, Nt = H_est.shape
    X_hat = np.zeros((K, Nt), dtype=complex)
    for k in range(K):
        H_k = H_est[k]
        X_hat[k] = (np.linalg.pinv(H_k) if Nr >= Nt
                    else H_k.conj().T) @ Y_rx[k]
    return X_hat


def compute_ber(H_true, H_est, snr_db):
    """
    Full MIMO-OFDM link BER simulation for one channel realisation.
    H_true / H_est: (K, Nr, Nt)
    """
    K, Nr, Nt = H_true.shape
    snr_lin   = 10 ** (snr_db / 10)
    n_sym     = K * Nt
    bits_tx   = np.random.randint(0, 2, n_sym * 2)
    X_freq    = bits_to_qpsk(bits_tx).reshape(K, Nt)

    Y_rx = np.zeros((K, Nr), dtype=complex)
    for k in range(K):
        Y_rx[k] = H_true[k] @ X_freq[k] + (
            np.random.randn(Nr) + 1j * np.random.randn(Nr)
        ) / np.sqrt(2 * snr_lin)

    X_hat    = zf_equalise(Y_rx, H_est)
    bits_rx  = qpsk_to_bits(X_hat.flatten())
    return float(np.sum(bits_tx != bits_rx) / len(bits_tx))