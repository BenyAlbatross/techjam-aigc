"use client";

import Image from "next/image";
import {
  ArrowRight,
  ChevronDown,
  CircleDot,
  Database,
  FileSearch,
  Filter,
  FlaskConical,
  Grid3X3,
  ImagePlus,
  Layers3,
  Search,
  ShieldCheck,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import { useDeferredValue, useMemo, useState } from "react";
import type { GalleryImage, GalleryPayload, Prediction, Truth } from "@/lib/types";

const truthLabels: Record<Truth, string> = { real: "Authentic", ai: "AI-generated", unknown: "Unknown" };

function shortId(value: string) {
  return `${value.slice(0, 7)}…${value.slice(-5)}`;
}

function percentage(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function conditionLabel(condition: string) {
  const labels: Record<string, string> = { clean: "Clean", color_jitter_20: "Color jitter · 20%", center_crop_80: "Center crop · 80%" };
  if (labels[condition]) return labels[condition];
  if (condition.startsWith("jpeg_q")) return `JPEG · q${condition.slice(6)}`;
  if (condition.startsWith("blur_sigma")) return `Blur · σ${condition.slice(10)}`;
  if (condition.startsWith("resize_")) return `Resize · ${condition.slice(7)}×`;
  if (condition.startsWith("noise_sigma")) return `Noise · σ${condition.slice(11)}`;
  return condition.replaceAll("_", " ");
}

function conditionedUrl(image: GalleryImage, condition: string) {
  return condition === "clean" ? image.imageUrl : `${image.imageUrl}?condition=${encodeURIComponent(condition)}`;
}

function ImageCard({ image, active, model, condition, onSelect }: {
  image: GalleryImage;
  active: boolean;
  model: string;
  condition: string;
  onSelect: () => void;
}) {
  const prediction = image.predictions.find((item) => item.model === model && item.condition === condition);
  const incorrect = prediction ? prediction.decision !== image.truth : false;
  return (
    <button className={`image-card ${active ? "active" : ""}`} onClick={onSelect} aria-pressed={active}>
      <span className="image-wrap">
        <Image src={conditionedUrl(image, condition)} alt={`${conditionLabel(condition)} ${truthLabels[image.truth]} sample ${shortId(image.id)}`} fill sizes="(max-width: 760px) 46vw, 220px" />
        <span className={`truth-badge ${image.truth}`}>{truthLabels[image.truth]}</span>
        {incorrect ? <span className="error-badge">Mismatch</span> : null}
      </span>
      <span className="card-copy">
        <span className="card-title"><span>{image.sourceFamily}</span><span className="mono">{prediction ? percentage(prediction.probabilityAi) : "—"}</span></span>
        <span className="card-meta"><span>{conditionLabel(condition)}</span><span>{image.format ?? "Image"}</span></span>
      </span>
    </button>
  );
}

function Lineage({ image, condition }: { image: GalleryImage; condition: string }) {
  return (
    <section className="lineage-section" aria-labelledby="lineage-title">
      <div className="section-heading">
        <div><span className="eyebrow">Provenance</span><h2 id="lineage-title">Modification chain</h2></div>
        <span className="quiet-badge">{condition === "clean" ? "1 known step" : "2 known steps"}</span>
      </div>
      <div className="lineage-track">
        <div className="lineage-node muted-node">
          <span className="node-icon"><Database size={17} /></span>
          <span><strong>Dataset source</strong><small className="mono">{shortId(image.baseId)}</small></span>
        </div>
        <ArrowRight className="lineage-arrow" size={22} />
        <div className="operation-node"><CircleDot size={14} /><span>{condition === "clean" ? "Canonicalize" : conditionLabel(condition)}</span><small>{condition === "clean" ? "RGB · 512² · Lanczos" : "Deterministic benchmark transform"}</small></div>
        <ArrowRight className="lineage-arrow" size={22} />
        <div className="lineage-node current-node">
          <Image src={conditionedUrl(image, condition)} alt="Selected benchmark derivative" width={54} height={54} />
          <span><strong>Selected result</strong><small>{conditionLabel(condition)}</small></span>
        </div>
      </div>
      {image.truth === "ai" ? (
        <p className="lineage-note"><Sparkles size={14} /> Generator reference is undisclosed in SID Set. No source edge invented.</p>
      ) : null}
    </section>
  );
}

function Metadata({ image, condition }: { image: GalleryImage; condition: string }) {
  const rows = [
    ["Truth", truthLabels[image.truth]],
    ["Transform", conditionLabel(condition)],
    ["Source family", image.sourceFamily],
    ["Generator", image.generatorFamily],
    ["Geometry", `${image.width ?? "?"} × ${image.height ?? "?"}`],
    ["Format", image.format ?? "Unknown"],
    ["Dataset", image.dataset],
    ["Rights", image.license],
  ];
  return (
    <section className="metadata-section">
      <div className="section-heading"><div><span className="eyebrow">Record</span><h2>Image metadata</h2></div><ShieldCheck size={19} /></div>
      <dl className="metadata-grid">
        {rows.map(([term, detail]) => <div key={term}><dt>{term}</dt><dd>{detail}</dd></div>)}
      </dl>
      <details className="audit-details">
        <summary>Audit identifiers <ChevronDown size={15} /></summary>
        <div><span>Sample ID</span><code>{image.id}</code></div>
        <div><span>Base ID</span><code>{image.baseId}</code></div>
        <div><span>SHA-256</span><code>{image.sha256}</code></div>
      </details>
    </section>
  );
}

function ScoreRow({ prediction, featured }: { prediction: Prediction; featured: boolean }) {
  return (
    <div className={`score-row ${featured ? "featured" : ""}`}>
      <div className="score-row-top"><span>{prediction.model.replaceAll("_", " ")}</span><strong>{percentage(prediction.probabilityAi)}</strong></div>
      <div className="score-track"><span style={{ width: `${prediction.probabilityAi * 100}%` }} /></div>
      <div className="score-caption"><span>{prediction.decision === "ai" ? "AI-generated" : "Authentic"}</span><span>threshold {prediction.threshold.toFixed(2)}</span></div>
    </div>
  );
}

function TestPanel({ models }: { models: string[] }) {
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState("ateeqq_siglip");
  const [state, setState] = useState<"idle" | "running" | "done" | "error">("idle");
  const [message, setMessage] = useState("");
  const [session, setSession] = useState<string | null>(null);

  async function runTest() {
    if (!file) return;
    setState("running");
    const body = new FormData();
    body.set("image", file);
    body.set("model", model);
    try {
      const response = await fetch("/api/inference", { method: "POST", body });
      const result = await response.json() as { error?: string; probabilityAi?: number; session?: string };
      if (!response.ok || result.probabilityAi === undefined) throw new Error(result.error ?? "Inference failed");
      setSession(result.session ?? null);
      setMessage(`${percentage(result.probabilityAi)} AI probability`);
      setState("done");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Inference failed");
      setState("error");
    }
  }

  async function clear() {
    if (session) await fetch(`/api/inference?session=${encodeURIComponent(session)}`, { method: "DELETE" });
    setFile(null); setState("idle"); setMessage(""); setSession(null);
  }

  return (
    <section className="test-panel">
      <div className="section-heading compact"><div><span className="eyebrow">Sandbox</span><h2>Test a new image</h2></div><FlaskConical size={18} /></div>
      <label className="drop-zone">
        <input type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
        <Upload size={21} />
        <strong>{file ? file.name : "Drop or choose image"}</strong>
        <small>PNG, JPEG or WebP · max 10 MB</small>
      </label>
      <label className="select-label">Model<select value={model} onChange={(event) => setModel(event.target.value)}>
        {(models.length ? models : ["ateeqq_siglip"]).filter((item) => ["ateeqq_siglip", "community_forensics", "univfd"].includes(item)).map((item) => <option key={item}>{item}</option>)}
      </select></label>
      <button className="button primary full" disabled={!file || state === "running"} onClick={runTest}>
        {state === "running" ? "Running locally…" : "Run detector"}
      </button>
      {message ? <div className={`test-result ${state}`}><span>{message}</span><button onClick={clear} aria-label="Clear test session"><X size={15} /></button></div> : null}
      <p className="microcopy">Unknown truth · excluded from benchmark, training, and calibration.</p>
    </section>
  );
}

export function GalleryWorkbench({ initialData }: { initialData: GalleryPayload }) {
  const [selectedId, setSelectedId] = useState(initialData.images[0]?.id ?? "");
  const [query, setQuery] = useState("");
  const [truth, setTruth] = useState<Truth | "all">("all");
  const [model, setModel] = useState(initialData.models.includes("ateeqq_siglip") ? "ateeqq_siglip" : initialData.models[0] ?? "");
  const [condition, setCondition] = useState("clean");
  const deferredQuery = useDeferredValue(query.toLowerCase());
  const filtered = useMemo(() => initialData.images.filter((image) => {
    const matchesTruth = truth === "all" || image.truth === truth;
    const haystack = `${image.id} ${image.sourceFamily} ${image.generatorFamily}`.toLowerCase();
    return matchesTruth && haystack.includes(deferredQuery);
  }), [deferredQuery, initialData.images, truth]);
  const selected = initialData.images.find((image) => image.id === selectedId) ?? filtered[0] ?? initialData.images[0];

  if (!selected) {
    return <main className="loading-shell"><ImagePlus size={28} /><p>No local gallery manifest found.</p><code>work/manifests/sid_set_1000x2_canonical.json</code></main>;
  }

  const selectedPrediction = selected.predictions.find((item) => item.model === model && item.condition === condition);
  const cleanPrediction = selected.predictions.find((item) => item.model === model && item.condition === "clean");
  const scoreDelta = selectedPrediction && cleanPrediction ? selectedPrediction.probabilityAi - cleanPrediction.probabilityAi : null;
  const conditionScores = selected.predictions.filter((item) => item.condition === condition);
  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark"><FileSearch size={20} /></span><span><strong>TRACE LENS</strong><small>AIGC evidence browser</small></span></div>
        <div className="dataset-state"><span className="pulse" />SID Set · controlled gate <span>{initialData.totalImages.toLocaleString()} images</span></div>
        <button className="icon-button" aria-label="Open filters"><Filter size={18} /></button>
      </header>

      <aside className="filter-rail">
        <div className="rail-icon active"><Grid3X3 size={18} /><span>Gallery</span></div>
        <div className="rail-icon"><Layers3 size={18} /><span>Datasets</span></div>
        <div className="rail-icon"><FlaskConical size={18} /><span>Models</span></div>
      </aside>

      <section className="workspace">
        <div className="toolbar">
          <label className="search-box"><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search source or sample ID" /></label>
          <div className="segmented" aria-label="Filter by truth">
            {(["all", "real", "ai"] as const).map((value) => <button key={value} className={truth === value ? "active" : ""} onClick={() => setTruth(value)}>{value === "all" ? "All" : truthLabels[value]}</button>)}
          </div>
          <label className="model-select">Condition<select value={condition} onChange={(event) => setCondition(event.target.value)}>{initialData.conditions.map((item) => <option key={item} value={item}>{conditionLabel(item)}</option>)}</select></label><label className="model-select">Model<select value={model} onChange={(event) => setModel(event.target.value)}>{initialData.models.map((item) => <option key={item}>{item}</option>)}</select></label>
        </div>

        <div className="content-grid">
          <section className="gallery-pane" aria-label="Image gallery">
            <div className="pane-heading"><span><strong>{filtered.length}</strong> loaded samples</span><span>Showing {conditionLabel(condition)} panel</span></div>
            <div className="gallery-grid">
              {filtered.map((image) => <ImageCard key={image.id} image={image} active={image.id === selected.id} model={model} condition={condition} onSelect={() => setSelectedId(image.id)} />)}
            </div>
          </section>

          <section className="detail-pane">
            <div className="hero-preview">
              <Image src={conditionedUrl(selected, condition)} alt={`Selected ${conditionLabel(condition)} ${truthLabels[selected.truth]} image`} fill sizes="(max-width: 1100px) 90vw, 680px" priority />
              <div className="hero-caption"><span className={`truth-badge ${selected.truth}`}>{truthLabels[selected.truth]}</span><span className="mono">{shortId(selected.id)}</span></div>
            </div>
            <Lineage image={selected} condition={condition} />
            <Metadata image={selected} condition={condition} />
          </section>
        </div>
      </section>

      <aside className="inspector">
        <div className="inspector-heading"><div><span className="eyebrow">Model evidence</span><h1>Detector output</h1></div><span className="live-dot">Fixed thresholds</span></div>
        {selectedPrediction ? (
          <div className="primary-score">
            <div className="score-orbit" style={{ "--score": `${selectedPrediction.probabilityAi * 360}deg` } as React.CSSProperties}>
              <div><strong>{percentage(selectedPrediction.probabilityAi)}</strong><span>AI probability</span></div>
            </div>
            <div className="verdict"><span>Verdict</span><strong>{selectedPrediction.decision === "ai" ? "AI-generated" : "Authentic"}</strong><small>{selectedPrediction.model.replaceAll("_", " ")}</small></div>
          </div>
        ) : <div className="empty-score">No prediction shard for this image.</div>}
        {scoreDelta !== null && condition !== "clean" ? <div className="delta-callout"><span>Clean to transformed</span><strong>{scoreDelta >= 0 ? "+" : ""}{percentage(scoreDelta)}</strong><small>{selectedPrediction?.decision !== cleanPrediction?.decision ? "Decision flipped" : "Decision stable"}</small></div> : null}\n        <div className="all-scores"><div className="subheading"><span>All model outputs</span><small>{conditionScores.length} available</small></div>{conditionScores.map((prediction) => <ScoreRow key={prediction.model} prediction={prediction} featured={prediction.model === model} />)}</div>
        <TestPanel models={initialData.models} />
      </aside>
    </main>
  );
}
