import { ChangeEvent, DragEvent, useEffect, useMemo, useState } from "react";

type Prediction = {
  anomaly_score: number;
  threshold: number;
  is_anomaly: boolean;
  model_version: string;
  heatmap_data_url: string;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

function formatScore(value: number) {
  return value.toFixed(3);
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function selectFile(next: File | undefined) {
    if (!next) return;
    if (!next.type.match(/^image\/(png|jpeg)$/)) {
      setError("Choose a PNG or JPEG image.");
      return;
    }
    setFile(next);
    setPrediction(null);
    setError("");
  }

  async function inspect() {
    if (!file) return;
    setBusy(true);
    setError("");
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch(`${API_BASE}/v1/predict`, { method: "POST", body: form });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "Inspection failed.");
      setPrediction(body as Prediction);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Inspection failed.");
    } finally {
      setBusy(false);
    }
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragging(false);
    selectFile(event.dataTransfer.files[0]);
  }

  return (
    <main>
      <nav>
        <a className="brand" href="#top" aria-label="VisionGuard home">
          <span className="brand-mark">VG</span>
          <span>VisionGuard <b>AI</b></span>
        </a>
        <span className="system-pill"><i /> Normal-only detector</span>
      </nav>

      <section className="hero" id="top">
        <div>
          <p className="eyebrow">Visual quality inspection / v0.1</p>
          <h1>Surface defects,<br /><em>made visible.</em></h1>
          <p className="lede">
            Upload an inspection image. VisionGuard compares local surface patches with a
            memory of normal samples and returns both a decision and a localized heatmap.
          </p>
        </div>
        <div className="method-card">
          <span>Inspection logic</span>
          <strong>Normal memory</strong>
          <div className="method-line"><i /><i /><i /><i /><i /></div>
          <strong>k-NN patch distance</strong>
          <small>No defective samples enter training.</small>
        </div>
      </section>

      <section className="workspace">
        <div className="panel input-panel">
          <header>
            <span>01</span>
            <div><h2>Inspection input</h2><p>PNG or JPEG · maximum 5 MB</p></div>
          </header>
          <label
            className={`dropzone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            {previewUrl ? (
              <img src={previewUrl} alt="Selected inspection input" />
            ) : (
              <div className="drop-copy">
                <span className="scan-icon">＋</span>
                <strong>Drop inspection image</strong>
                <small>or click to browse</small>
              </div>
            )}
            <input
              type="file"
              accept="image/png,image/jpeg"
              onChange={(event: ChangeEvent<HTMLInputElement>) => selectFile(event.target.files?.[0])}
            />
          </label>
          <div className="file-row">
            <span>{file?.name ?? "No image selected"}</span>
            <button type="button" disabled={!file || busy} onClick={inspect}>
              {busy ? "Inspecting…" : "Run inspection"}<b>↗</b>
            </button>
          </div>
          {error && <p className="error" role="alert">{error}</p>}
        </div>

        <div className="panel result-panel">
          <header>
            <span>02</span>
            <div><h2>Model output</h2><p>Decision + anomaly localization</p></div>
          </header>
          {prediction ? (
            <div className="result-content">
              <div className="heatmap-wrap">
                <img src={prediction.heatmap_data_url} alt="Anomaly heatmap overlay" />
                <span className="heatmap-label">HEATMAP OVERLAY</span>
              </div>
              <div className="decision-row">
                <div className={`decision ${prediction.is_anomaly ? "reject" : "pass"}`}>
                  <small>Inspection decision</small>
                  <strong>{prediction.is_anomaly ? "ANOMALY" : "NORMAL"}</strong>
                </div>
                <div className="score">
                  <small>Anomaly score</small>
                  <strong>{formatScore(prediction.anomaly_score)}</strong>
                  <span>threshold {formatScore(prediction.threshold)}</span>
                </div>
              </div>
              <div className="meter">
                <i style={{ width: `${Math.min(100, prediction.anomaly_score / Math.max(prediction.threshold * 1.8, 0.01) * 100)}%` }} />
                <b style={{ left: `${Math.min(98, 100 / 1.8)}%` }} />
              </div>
              <p className="model-version">Model {prediction.model_version}</p>
            </div>
          ) : (
            <div className="empty-result">
              <div className="radar"><i /><i /><i /></div>
              <strong>Awaiting inspection</strong>
              <p>The localized heatmap and anomaly score will appear here.</p>
            </div>
          )}
        </div>
      </section>

      <footer>
        <span>Portfolio reference implementation</span>
        <span>Synthetic data · not production-calibrated</span>
      </footer>
    </main>
  );
}
