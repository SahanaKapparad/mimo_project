import { useState, useEffect, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, BarChart, Bar, Cell
} from "recharts";

const API = "https://mimoproject-production.up.railway.app";

const palette = {
  actual: "#22d3ee",
  ls:     "#f97316",
  cnn:    "#a78bfa",
  lsBer:  "#f97316",
  cnnBer: "#34d399",
};

// ── Reusable card ──────────────────────────────────────────
function Card({ title, children, className = "" }) {
  return (
    <div className={`card ${className}`}>
      {title && <div className="card-title">{title}</div>}
      {children}
    </div>
  );
}

// ── Stat badge ─────────────────────────────────────────────
function Stat({ label, value, unit = "", color = "#a78bfa" }) {
  return (
    <div className="stat">
      <div className="stat-value" style={{ color }}>{value}</div>
      <div className="stat-label">{label}{unit && <span className="stat-unit"> {unit}</span>}</div>
    </div>
  );
}

// ── Slider input ───────────────────────────────────────────
function SliderInput({ label, min, max, step = 1, value, onChange }) {
  return (
    <div className="slider-wrap">
      <div className="slider-header">
        <span className="slider-label">{label}</span>
        <span className="slider-val">{value}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="slider"
      />
    </div>
  );
}

// ── Loading spinner ────────────────────────────────────────
function Spinner() {
  return <div className="spinner" />;
}

// ── Main App ───────────────────────────────────────────────
export default function App() {
  const [status, setStatus]         = useState(null);
  const [activeTab, setActiveTab]   = useState("estimate");

  // Estimate state
  const [snr, setSnr]               = useState(10);
  const [nSamples, setNSamples]     = useState(20);
  const [nTx, setNTx]               = useState(2);
  const [nRx, setNRx]               = useState(2);
  const [estResult, setEstResult]   = useState(null);
  const [estLoading, setEstLoading] = useState(false);
  const [estError, setEstError]     = useState("");

  // BER-SNR state
  const [berLoading, setBerLoading] = useState(false);
  const [berResult, setBerResult]   = useState(null);
  const [berError, setBerError]     = useState("");
  const [berSamples, setBerSamples] = useState(30);

  // Train state
  const [trainParams, setTrainParams] = useState({
    num_samples: 5000, epochs: 30, snr_min: 0, snr_max: 20,
  });
  const [trainLoading, setTrainLoading] = useState(false);
  const [trainMsg, setTrainMsg]         = useState("");

  // Poll status
  useEffect(() => {
    const poll = async () => {
      try {
        const r = await fetch(`${API}/status`);
        if (r.ok) setStatus(await r.json());
      } catch {}
    };
    poll();
    const id = setInterval(poll, 3000);
    return () => clearInterval(id);
  }, []);

  // ── Estimate handler ──────────────────────────────────────
  const runEstimate = useCallback(async () => {
    setEstLoading(true); setEstError("");
    try {
      const r = await fetch(`${API}/estimate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ snr_db: snr, n_samples: nSamples, n_tx: nTx, n_rx: nRx }),
      });
      if (!r.ok) throw new Error((await r.json()).detail);
      setEstResult(await r.json());
    } catch (e) { setEstError(e.message); }
    setEstLoading(false);
  }, [snr, nSamples, nTx, nRx]);

  // ── BER-SNR handler ───────────────────────────────────────
  const runBerSnr = useCallback(async () => {
    setBerLoading(true); setBerError("");
    try {
      const r = await fetch(`${API}/ber-snr-curve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          snr_range: [0, 5, 10, 15, 20],
          n_samples: berSamples,
        }),
      });
      if (!r.ok) throw new Error((await r.json()).detail);
      setBerResult(await r.json());
    } catch (e) { setBerError(e.message); }
    setBerLoading(false);
  }, [berSamples]);

  // ── Train handler ─────────────────────────────────────────
  const runTrain = useCallback(async () => {
    setTrainLoading(true); setTrainMsg("");
    try {
      const r = await fetch(`${API}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(trainParams),
      });
      if (!r.ok) throw new Error((await r.json()).detail);
      const d = await r.json();
      setTrainMsg(`${d.message} — ${d.job_id}`);
    } catch (e) { setTrainMsg(`Error: ${e.message}`); }
    setTrainLoading(false);
  }, [trainParams]);

  // ── Channel comparison chart data ─────────────────────────
  const channelData = estResult
    ? estResult.sample_actual_real.map((v, i) => ({
        i,
        actual: +v.toFixed(4),
        ls:     +estResult.sample_ls_real[i].toFixed(4),
        cnn:    +estResult.sample_cnn_real[i].toFixed(4),
      }))
    : [];

  // ── BER chart data ────────────────────────────────────────
  const berData = berResult
    ? berResult.snr_values.map((snr, i) => ({
        snr,
        ls:  +berResult.ber_ls_values[i].toFixed(5),
        cnn: +berResult.ber_cnn_values[i].toFixed(5),
      }))
    : [];

  // ── NMSE comparison bar data ──────────────────────────────
  const nmseData = estResult
    ? [
        { name: "LS",  nmse: +estResult.nmse_ls_db.toFixed(2)  },
        { name: "CNN", nmse: +estResult.nmse_cnn_db.toFixed(2) },
      ]
    : [];

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-left">
          <div className="logo">MIMO<span>EST</span></div>
          <div>
            <div className="header-title">5G/6G Channel Estimator</div>
            <div className="header-sub">AI-Powered MIMO-OFDM — CNN vs LS</div>
          </div>
        </div>
        <div className="status-pill" data-ok={status?.model_loaded}>
          <div className="status-dot" />
          {status ? (status.model_loaded ? "Model ready" : "No model") : "Connecting…"}
        </div>
      </header>

      {/* ── Tabs ── */}
      <nav className="tabs">
        {["estimate", "ber-snr", "train", "log"].map(t => (
          <button
            key={t} className="tab"
            data-active={activeTab === t}
            onClick={() => setActiveTab(t)}
          >
            { t === "estimate" ? "Channel Estimate"
            : t === "ber-snr" ? "BER vs SNR"
            : t === "train"   ? "Train Model"
            : "Training Log" }
          </button>
        ))}
      </nav>

      <main className="main">

        {/* ════════════════ ESTIMATE TAB ════════════════ */}
        {activeTab === "estimate" && (
          <div className="tab-content">
            <div className="two-col">
              <Card title="Parameters">
                <SliderInput label="SNR (dB)" min={0} max={30} value={snr} onChange={setSnr} />
                <SliderInput label="Samples"  min={5} max={200} step={5} value={nSamples} onChange={setNSamples} />
                <SliderInput label="Tx Antennas" min={1} max={4} value={nTx} onChange={setNTx} />
                <SliderInput label="Rx Antennas" min={1} max={4} value={nRx} onChange={setNRx} />
                <button className="btn" onClick={runEstimate} disabled={estLoading || !status?.model_loaded}>
                  {estLoading ? <><Spinner /> Running…</> : "Run Estimation"}
                </button>
                {estError && <div className="error">{estError}</div>}
                {!status?.model_loaded && (
                  <div className="hint">Train a model first via the Train tab.</div>
                )}
              </Card>

              {estResult && (
                <Card title="NMSE Comparison">
                  <ResponsiveContainer width="100%" height={180}>
                    <BarChart data={nmseData} barSize={48}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis dataKey="name" tick={{ fill: "#9ca3af", fontSize: 12 }} />
                      <YAxis tick={{ fill: "#9ca3af", fontSize: 12 }} unit=" dB" />
                      <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #333", borderRadius: 8 }} />
                      <Bar dataKey="nmse" radius={[4,4,0,0]}>
                        {nmseData.map((e, i) => (
                          <Cell key={i} fill={i === 0 ? palette.ls : palette.cnn} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <div className="stats-row">
                    <Stat label="NMSE LS"  value={estResult.nmse_ls_db.toFixed(2)}  unit="dB" color={palette.ls} />
                    <Stat label="NMSE CNN" value={estResult.nmse_cnn_db.toFixed(2)} unit="dB" color={palette.cnn} />
                  </div>
                </Card>
              )}
            </div>

            {estResult && (
              <Card title="Channel Estimate — Real Part (Subcarrier 0 → 63)" className="full-card">
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={channelData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
                    <XAxis dataKey="i" label={{ value: "Subcarrier", position: "insideBottom", offset: -4, fill: "#9ca3af", fontSize: 11 }} tick={{ fill: "#9ca3af", fontSize: 10 }} />
                    <YAxis tick={{ fill: "#9ca3af", fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #333", borderRadius: 8, fontSize: 12 }} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Line type="monotone" dataKey="actual" stroke={palette.actual} dot={false} strokeWidth={2} name="True Channel" />
                    <Line type="monotone" dataKey="ls"     stroke={palette.ls}     dot={false} strokeWidth={1.5} strokeDasharray="4 3" name="LS Estimate" />
                    <Line type="monotone" dataKey="cnn"    stroke={palette.cnn}    dot={false} strokeWidth={2} name="CNN Estimate" />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            )}
          </div>
        )}

        {/* ════════════════ BER-SNR TAB ════════════════ */}
        {activeTab === "ber-snr" && (
          <div className="tab-content">
            <Card title="BER vs SNR Settings">
              <SliderInput label="Samples per SNR point" min={5} max={100} step={5} value={berSamples} onChange={setBerSamples} />
              <button className="btn" onClick={runBerSnr} disabled={berLoading || !status?.model_loaded}>
                {berLoading ? <><Spinner /> Computing…</> : "Compute BER Curve"}
              </button>
              {berError && <div className="error">{berError}</div>}
            </Card>

            {berResult && (
              <Card title="BER vs SNR — QPSK / Zero-Forcing Equaliser" className="full-card">
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={berData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
                    <XAxis dataKey="snr" label={{ value: "SNR (dB)", position: "insideBottom", offset: -4, fill: "#9ca3af", fontSize: 11 }} tick={{ fill: "#9ca3af", fontSize: 10 }} />
                    <YAxis scale="log" domain={["auto","auto"]} tick={{ fill: "#9ca3af", fontSize: 10 }} label={{ value: "BER (log)", angle: -90, position: "insideLeft", fill: "#9ca3af", fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #333", borderRadius: 8, fontSize: 12 }} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Line type="monotone" dataKey="ls"  stroke={palette.lsBer}  dot={{ r: 4 }} strokeWidth={2} strokeDasharray="5 3" name="LS Estimation" />
                    <Line type="monotone" dataKey="cnn" stroke={palette.cnnBer} dot={{ r: 4 }} strokeWidth={2} name="CNN Estimation" />
                  </LineChart>
                </ResponsiveContainer>

                <div className="ber-table">
                  <div className="ber-header">
                    <span>SNR</span><span>LS BER</span><span>CNN BER</span><span>Improvement</span>
                  </div>
                  {berData.map(d => (
                    <div className="ber-row" key={d.snr}>
                      <span>{d.snr} dB</span>
                      <span style={{ color: palette.lsBer }}>{d.ls.toFixed(5)}</span>
                      <span style={{ color: palette.cnnBer }}>{d.cnn.toFixed(5)}</span>
                      <span style={{ color: "#a78bfa" }}>
                        {d.ls > 0 ? `${((1 - d.cnn / d.ls) * 100).toFixed(1)}%` : "—"}
                      </span>
                    </div>
                  ))}
                </div>
              </Card>
            )}
          </div>
        )}

        {/* ════════════════ TRAIN TAB ════════════════ */}
        {activeTab === "train" && (
          <div className="tab-content">
            <Card title="Training Parameters">
              <SliderInput label="Training samples" min={500} max={20000} step={500} value={trainParams.num_samples} onChange={v => setTrainParams(p => ({ ...p, num_samples: v }))} />
              <SliderInput label="Epochs" min={5} max={100} value={trainParams.epochs} onChange={v => setTrainParams(p => ({ ...p, epochs: v }))} />
              <SliderInput label="SNR min (dB)" min={-5} max={10} value={trainParams.snr_min} onChange={v => setTrainParams(p => ({ ...p, snr_min: v }))} />
              <SliderInput label="SNR max (dB)" min={10} max={30} value={trainParams.snr_max} onChange={v => setTrainParams(p => ({ ...p, snr_max: v }))} />
              <button className="btn btn-train" onClick={runTrain} disabled={trainLoading || status?.training}>
                {trainLoading || status?.training ? <><Spinner /> Training in progress…</> : "Start Training"}
              </button>
              {trainMsg && <div className="hint">{trainMsg}</div>}
            </Card>

            {status?.training && (
              <Card title="Training Progress">
                <div className="progress-bar-wrap">
                  <div className="progress-bar" style={{ width: `${status.training_progress}%` }} />
                </div>
                <div className="progress-pct">{status.training_progress}%</div>
              </Card>
            )}
          </div>
        )}

        {/* ════════════════ LOG TAB ════════════════ */}
        {activeTab === "log" && (
          <div className="tab-content">
            <Card title="Training Log">
              <div className="log-box">
                {status?.training_log?.length
                  ? [...status.training_log].reverse().map((l, i) => (
                      <div key={i} className="log-line">{l}</div>
                    ))
                  : <div className="hint">No log entries yet.</div>
                }
              </div>
            </Card>
          </div>
        )}
      </main>

      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0d0d1a; }
        .app {
          min-height: 100vh;
          background: #0d0d1a;
          color: #e2e2ef;
          font-family: 'JetBrains Mono', 'Fira Code', monospace;
        }
        .header {
          display: flex; align-items: center; justify-content: space-between;
          padding: 18px 28px; border-bottom: 1px solid #1e1e3a;
          background: #0a0a18;
        }
        .header-left { display: flex; align-items: center; gap: 16px; }
        .logo {
          font-size: 22px; font-weight: 700; letter-spacing: -1px;
          color: #fff;
        }
        .logo span { color: #a78bfa; }
        .header-title { font-size: 15px; font-weight: 600; color: #e2e2ef; }
        .header-sub { font-size: 11px; color: #6b6b8a; margin-top: 2px; }
        .status-pill {
          display: flex; align-items: center; gap: 7px;
          padding: 6px 14px; border-radius: 20px;
          background: #1a1a2e; border: 1px solid #2a2a4a;
          font-size: 12px; color: #9ca3af;
        }
        .status-dot {
          width: 8px; height: 8px; border-radius: 50%;
          background: #4b5563;
        }
        .status-pill[data-ok="true"] .status-dot { background: #34d399; }
        .status-pill[data-ok="true"] { color: #34d399; border-color: #065f46; }
        .tabs {
          display: flex; gap: 0; padding: 0 28px;
          border-bottom: 1px solid #1e1e3a; background: #0a0a18;
        }
        .tab {
          padding: 12px 20px; background: none; border: none;
          color: #6b6b8a; font-size: 12px; font-family: inherit;
          cursor: pointer; border-bottom: 2px solid transparent;
          transition: color .2s, border-color .2s;
          letter-spacing: 0.5px; text-transform: uppercase;
        }
        .tab[data-active="true"] { color: #a78bfa; border-color: #a78bfa; }
        .tab:hover { color: #c4b5fd; }
        .main { padding: 24px 28px; }
        .tab-content { display: flex; flex-direction: column; gap: 20px; }
        .two-col { display: grid; grid-template-columns: 340px 1fr; gap: 20px; }
        .card {
          background: #12122a; border: 1px solid #1e1e3a;
          border-radius: 12px; padding: 20px;
        }
        .full-card { width: 100%; }
        .card-title {
          font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
          color: #6b6b8a; margin-bottom: 16px; font-weight: 600;
        }
        .slider-wrap { margin-bottom: 16px; }
        .slider-header { display: flex; justify-content: space-between; margin-bottom: 6px; }
        .slider-label { font-size: 12px; color: #9ca3af; }
        .slider-val {
          font-size: 12px; color: #a78bfa; font-weight: 600;
          background: #1e1e3a; padding: 1px 8px; border-radius: 4px;
        }
        .slider {
          width: 100%; accent-color: #a78bfa;
          height: 4px; cursor: pointer;
        }
        .btn {
          width: 100%; padding: 11px; margin-top: 6px;
          background: #4c1d95; border: 1px solid #7c3aed;
          color: #e9d5ff; border-radius: 8px; font-family: inherit;
          font-size: 13px; cursor: pointer; font-weight: 600;
          display: flex; align-items: center; justify-content: center; gap: 8px;
          transition: background .2s;
        }
        .btn:hover:not(:disabled) { background: #5b21b6; }
        .btn:disabled { opacity: 0.45; cursor: not-allowed; }
        .btn-train { background: #065f46; border-color: #059669; color: #d1fae5; }
        .btn-train:hover:not(:disabled) { background: #047857; }
        .error { margin-top: 10px; color: #f87171; font-size: 12px; }
        .hint  { margin-top: 10px; color: #6b6b8a; font-size: 12px; }
        .stats-row { display: flex; gap: 16px; margin-top: 16px; }
        .stat { flex: 1; background: #0d0d1a; border-radius: 8px; padding: 12px; text-align: center; }
        .stat-value { font-size: 22px; font-weight: 700; }
        .stat-label { font-size: 11px; color: #6b6b8a; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
        .stat-unit { color: #4b5563; }
        .ber-table { margin-top: 20px; font-size: 12px; }
        .ber-header {
          display: grid; grid-template-columns: repeat(4,1fr);
          padding: 8px 12px; background: #0d0d1a; border-radius: 6px;
          color: #6b6b8a; font-weight: 600; text-transform: uppercase;
          letter-spacing: 0.5px; margin-bottom: 4px;
        }
        .ber-row {
          display: grid; grid-template-columns: repeat(4,1fr);
          padding: 8px 12px; border-bottom: 1px solid #1e1e3a;
        }
        .progress-bar-wrap {
          background: #1e1e3a; border-radius: 4px; height: 8px; overflow: hidden;
        }
        .progress-bar {
          height: 100%; background: linear-gradient(90deg, #7c3aed, #34d399);
          border-radius: 4px; transition: width .5s ease;
        }
        .progress-pct { font-size: 13px; color: #a78bfa; margin-top: 8px; text-align: right; }
        .log-box {
          background: #080810; border-radius: 8px; padding: 16px;
          max-height: 400px; overflow-y: auto; font-size: 12px;
          line-height: 1.8; border: 1px solid #1e1e3a;
        }
        .log-line { color: #6ee7b7; padding: 1px 0; }
        .spinner {
          width: 14px; height: 14px; border: 2px solid transparent;
          border-top-color: currentColor; border-radius: 50%;
          animation: spin .7s linear infinite; display: inline-block;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        @media (max-width: 700px) {
          .two-col { grid-template-columns: 1fr; }
          .main { padding: 16px; }
        }
      `}</style>
    </div>
  );
}