# ============================================================
# FastAPI Backend — AI Channel Estimation for MIMO-OFDM
# ============================================================
# Endpoints:
#   POST /estimate      → run CNN channel estimation
#   POST /ber           → compute BER at a given SNR
#   POST /ber-snr-curve → full BER vs SNR sweep
#   POST /train         → trigger model training job
#   GET  /status        → server + model status
# ============================================================

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import os
import time
import threading

# ── Import your simulation engine ──────────────────────────
from simulation import (
    generate_mimo_channel,
    add_noise,
    ls_interpolate,
    channel_to_features,
    pred_to_channel,
    compute_ber,
    pilot_idx,
)
from model import build_cnn, train_model

# ── App setup ───────────────────────────────────────────────
app = FastAPI(
    title="MIMO-OFDM Channel Estimator API",
    description="AI-based channel estimation for 5G/6G MIMO-OFDM systems",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ─────────────────────────────────────────────
MODEL_PATH   = "cnn_mimo_channel_estimator.keras"
model        = None
training_job = {"running": False, "progress": 0, "log": []}
N_FFT        = 64
N_TX         = 2
N_RX         = 2
L_PATHS      = 6
N_PILOTS     = 16

def load_model_if_exists():
    global model
    if os.path.exists(MODEL_PATH):
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
        print(f"[startup] Loaded model from {MODEL_PATH}")

load_model_if_exists()

# ─────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────

class EstimateRequest(BaseModel):
    snr_db: float = Field(10.0, description="SNR in dB for the noisy observation")
    n_samples: int = Field(10, ge=1, le=200, description="Number of channel samples")
    n_tx: int = Field(2, ge=1, le=4)
    n_rx: int = Field(2, ge=1, le=4)

class EstimateResponse(BaseModel):
    nmse_ls_db:  float
    nmse_cnn_db: float
    sample_actual_real:  List[float]   # first sample, Tx0->Rx0, real part
    sample_ls_real:      List[float]
    sample_cnn_real:     List[float]

class BerRequest(BaseModel):
    snr_db: float = Field(10.0)
    n_samples: int = Field(50, ge=1, le=200)

class BerResponse(BaseModel):
    ber_ls:  float
    ber_cnn: float
    snr_db:  float

class BerSnrRequest(BaseModel):
    snr_range: List[float] = Field(
        default=[0, 5, 10, 15, 20],
        description="List of SNR values in dB"
    )
    n_samples: int = Field(30, ge=5, le=100)

class BerSnrResponse(BaseModel):
    snr_values:    List[float]
    ber_ls_values: List[float]
    ber_cnn_values: List[float]

class TrainRequest(BaseModel):
    num_samples: int  = Field(5000, ge=500,  le=20000)
    epochs:      int  = Field(30,   ge=5,    le=100)
    snr_min:     float = Field(0.0)
    snr_max:     float = Field(20.0)

class TrainResponse(BaseModel):
    message: str
    job_id:  str

class StatusResponse(BaseModel):
    model_loaded:  bool
    model_path:    str
    training:      bool
    training_progress: int
    training_log:  List[str]
    server_time:   float

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def nmse_db(H_est, H_true):
    err = np.abs(H_est - H_true) ** 2
    ref = np.abs(H_true) ** 2
    return float(10 * np.log10(err.sum() / ref.sum()))

def run_estimation(snr_db, n_samples, n_tx, n_rx):
    """Generate channel, LS estimate, CNN estimate. Returns tuple."""
    H_true = generate_mimo_channel(n_samples, N_FFT, n_tx, n_rx, L_PATHS)
    H_p    = H_true[:, pilot_idx, :, :]
    H_pn   = add_noise(H_p, snr_db)
    H_ls   = ls_interpolate(H_pn, N_FFT, pilot_idx, n_tx, n_rx)

    # CNN inference
    X_raw  = channel_to_features(H_ls)
    Xm     = X_raw.mean(axis=(1,2,3), keepdims=True)
    Xs     = X_raw.std(axis=(1,2,3),  keepdims=True) + 1e-8
    X_norm = (X_raw - Xm) / Xs
    pred   = model.predict(X_norm, verbose=0)
    H_cnn  = pred_to_channel(pred, Xm, Xs, N_FFT, n_rx, n_tx)
    return H_true, H_ls, H_cnn

# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    return {"message": "MIMO-OFDM Channel Estimator API is running"}


@app.get("/status", response_model=StatusResponse, tags=["health"])
def status():
    return StatusResponse(
        model_loaded       = model is not None,
        model_path         = MODEL_PATH,
        training           = training_job["running"],
        training_progress  = training_job["progress"],
        training_log       = training_job["log"][-20:],
        server_time        = time.time(),
    )


@app.post("/estimate", response_model=EstimateResponse, tags=["inference"])
def estimate(req: EstimateRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded. Train first via POST /train")

    H_true, H_ls, H_cnn = run_estimation(
        req.snr_db, req.n_samples, req.n_tx, req.n_rx
    )

    return EstimateResponse(
        nmse_ls_db  = nmse_db(H_ls,  H_true),
        nmse_cnn_db = nmse_db(H_cnn, H_true),
        sample_actual_real = np.real(H_true[0, :, 0, 0]).tolist(),
        sample_ls_real     = np.real(H_ls[0,   :, 0, 0]).tolist(),
        sample_cnn_real    = np.real(H_cnn[0,  :, 0, 0]).tolist(),
    )


@app.post("/ber", response_model=BerResponse, tags=["inference"])
def ber_single(req: BerRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded.")

    H_true, H_ls, H_cnn = run_estimation(req.snr_db, req.n_samples, N_TX, N_RX)

    bers_ls  = [compute_ber(H_true[i], H_ls[i],  req.snr_db) for i in range(req.n_samples)]
    bers_cnn = [compute_ber(H_true[i], H_cnn[i], req.snr_db) for i in range(req.n_samples)]

    return BerResponse(
        ber_ls  = float(np.mean(bers_ls)),
        ber_cnn = float(np.mean(bers_cnn)),
        snr_db  = req.snr_db,
    )


@app.post("/ber-snr-curve", response_model=BerSnrResponse, tags=["inference"])
def ber_snr_curve(req: BerSnrRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded.")

    ls_curve, cnn_curve = [], []
    for snr in req.snr_range:
        H_true, H_ls, H_cnn = run_estimation(snr, req.n_samples, N_TX, N_RX)
        bl  = [compute_ber(H_true[i], H_ls[i],  snr) for i in range(req.n_samples)]
        bc  = [compute_ber(H_true[i], H_cnn[i], snr) for i in range(req.n_samples)]
        ls_curve.append(float(np.mean(bl)))
        cnn_curve.append(float(np.mean(bc)))

    return BerSnrResponse(
        snr_values     = list(req.snr_range),
        ber_ls_values  = ls_curve,
        ber_cnn_values = cnn_curve,
    )


@app.post("/train", response_model=TrainResponse, tags=["training"])
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    if training_job["running"]:
        raise HTTPException(409, "A training job is already running.")

    job_id = f"job_{int(time.time())}"
    background_tasks.add_task(
        _background_train, req.num_samples, req.epochs,
        req.snr_min, req.snr_max, job_id
    )
    return TrainResponse(message="Training started", job_id=job_id)


def _background_train(num_samples, epochs, snr_min, snr_max, job_id):
    global model, training_job
    training_job = {"running": True, "progress": 0, "log": [f"[{job_id}] Starting…"]}

    try:
        snr_range = list(np.arange(snr_min, snr_max + 1, 5))
        training_job["log"].append(f"Generating {num_samples} samples, SNRs: {snr_range}")

        new_model, _ = train_model(
            num_samples=num_samples,
            n_fft=N_FFT, n_tx=N_TX, n_rx=N_RX,
            l_paths=L_PATHS, snr_range=snr_range,
            epochs=epochs,
            progress_callback=lambda p, msg: training_job.update(
                {"progress": p, "log": training_job["log"] + [msg]}
            )
        )
        new_model.save(MODEL_PATH)
        model = new_model
        training_job["log"].append("Training complete. Model saved.")
    except Exception as e:
        training_job["log"].append(f"ERROR: {e}")
    finally:
        training_job["running"]  = False
        training_job["progress"] = 100