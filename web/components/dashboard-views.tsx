"use client";

import type { ConditionMetric, DatasetRecord } from "@/lib/types";

const pct = (value: number | null) => value === null ? "—" : `${(value * 100).toFixed(1)}%`;
const signed = (value: number | null) => value === null ? "—" : `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)} pp`;
const label = (value: string) => value.replaceAll("_", " ");

export function AnalyticsDashboard({ metrics, models, conditions, model, condition, onModel, onCondition, onDrilldown, evaluatedRows, updatedAt }: {
  metrics: ConditionMetric[]; models: string[]; conditions: string[]; model: string; condition: string;
  onModel: (value: string) => void; onCondition: (value: string) => void; onDrilldown: (outcome: "mismatch" | "fp" | "fn") => void; evaluatedRows: number; updatedAt: string | null;
}) {
  const selected = metrics.find((item) => item.model === model && item.condition === condition);
  const modelMetrics = metrics.filter((item) => item.model === model).sort((a, b) => b.mismatchRate - a.mismatchRate);
  if (!selected) return <div className="empty-view">No benchmark analytics available.</div>;
  const c = selected.confusion;
  return <div className="dashboard-view">
    <div className="view-header">
      <div><span className="eyebrow">Full benchmark · fixed thresholds</span><h1>Robustness analytics</h1><p>Performance, error composition, and condition-linked mismatch behavior.</p></div>
      <div className="view-controls"><label>Model<select value={model} onChange={(event) => onModel(event.target.value)}>{models.map((item) => <option key={item}>{item}</option>)}</select></label><label>Condition<select value={condition} onChange={(event) => onCondition(event.target.value)}>{conditions.map((item) => <option key={item}>{label(item)}</option>)}</select></label></div>
    </div>
    <div className="source-strip"><strong>Source</strong><span>local SID Set prediction shards</span><strong>Coverage</strong><span>{evaluatedRows.toLocaleString()} scored rows · {metrics.length} model-condition slices</span><strong>Updated</strong><span>{updatedAt ? new Date(updatedAt).toLocaleString() : "Unknown"}</span></div><div className="kpi-grid">
      <article><span>Balanced accuracy</span><strong>{pct(selected.balancedAccuracy)}</strong><small>average sensitivity + specificity</small></article>
      <article><span>Mismatches</span><strong>{(c.fp + c.fn).toLocaleString()}</strong><small>{pct(selected.mismatchRate)} of {c.total.toLocaleString()}</small></article>
      <article><span>False-positive rate</span><strong>{pct(selected.falsePositiveRate)}</strong><small>authentic predicted AI</small></article>
      <article><span>False-negative rate</span><strong>{pct(selected.falseNegativeRate)}</strong><small>AI predicted authentic</small></article>
    </div>
    <div className="analytics-grid">
      <section className="analytics-card confusion-card"><div className="card-head"><div><span className="eyebrow">Classification outcomes</span><h2>Truth × prediction matrix</h2></div><button onClick={() => onDrilldown("mismatch")}>View mismatches</button></div>
        <div className="matrix-axis">Predicted</div><div className="confusion-matrix">
          <span className="axis-label"></span><strong>Authentic</strong><strong>AI-generated</strong>
          <strong>Authentic truth</strong><button className="correct" title="True negative">{c.tn.toLocaleString()}<small>TN</small></button><button className="incorrect" onClick={() => onDrilldown("fp")}>{c.fp.toLocaleString()}<small>FP</small></button>
          <strong>AI truth</strong><button className="incorrect" onClick={() => onDrilldown("fn")}>{c.fn.toLocaleString()}<small>FN</small></button><button className="correct" title="True positive">{c.tp.toLocaleString()}<small>TP</small></button>
        </div>
      </section>
      <section className="analytics-card"><div className="card-head"><div><span className="eyebrow">Condition impact</span><h2>Mismatch rate</h2></div><small>selected model</small></div>
        <div className="bar-list">{modelMetrics.map((item) => <button key={item.condition} className={item.condition === condition ? "active" : ""} onClick={() => onCondition(item.condition)}><span>{label(item.condition)}</span><i><b style={{ width: `${Math.max(2, item.mismatchRate * 100)}%` }} /></i><strong>{pct(item.mismatchRate)}</strong></button>)}</div>
      </section>
    </div>
    <section className="analytics-card condition-table-card"><div className="card-head"><div><span className="eyebrow">Paired with clean predictions</span><h2>Conditions correlated with mismatches</h2></div><small>association, not causation</small></div>
      <div className="metric-table"><div className="metric-row heading"><span>Condition</span><span>Error</span><span>Lift vs clean</span><span>Clean-error φ</span><span>Newly wrong</span><span>Recovered</span></div>
      {modelMetrics.map((item) => <button className="metric-row" key={item.condition} onClick={() => onCondition(item.condition)}><strong>{label(item.condition)}</strong><span>{pct(item.mismatchRate)}</span><span>{item.condition === "clean" ? "baseline" : signed(item.mismatchLift)}</span><span>{item.cleanMismatchCorrelation?.toFixed(2) ?? "—"}</span><span>{pct(item.newlyWrongRate)}</span><span>{pct(item.recoveredRate)}</span></button>)}</div>
      <p className="method-note">φ measures whether the same samples fail under clean and transformed conditions. Newly wrong and recovered rates use paired sample IDs.</p>
    </section>
    <section className="analytics-card heatmap-card"><div className="card-head"><div><span className="eyebrow">All evaluated slices</span><h2>Model × condition error heatmap</h2></div></div>
      <div className="heatmap" style={{ gridTemplateColumns: `150px repeat(${conditions.length}, minmax(45px, 1fr))` }}><span></span>{conditions.map((item) => <strong key={item} title={label(item)}>{item === "clean" ? "Clean" : label(item).split(" ")[0]}</strong>)}{models.flatMap((m) => [<b key={m}>{label(m)}</b>, ...conditions.map((cond) => { const value = metrics.find((x) => x.model === m && x.condition === cond)?.mismatchRate ?? 0; return <button key={`${m}-${cond}`} style={{ "--heat": value } as React.CSSProperties} onClick={() => { onModel(m); onCondition(cond); }} title={`${label(m)} · ${label(cond)}: ${pct(value)}`}>{pct(value)}</button>; })])}</div>
    </section>
  </div>;
}

export function DatasetCatalog({ datasets }: { datasets: DatasetRecord[] }) {
  return <div className="dashboard-view"><div className="view-header"><div><span className="eyebrow">Virtual registry · rights-aware</span><h1>Datasets</h1><p>Grouped by approval state. Grouping never grants permission.</p></div></div>
    <div className="status-summary">{(["approved", "review", "blocked"] as const).map((status) => <article key={status}><strong>{datasets.filter((d) => d.status === status).length}</strong><span>{status}</span></article>)}</div>
    <div className="catalog-grid">{datasets.map((dataset) => <article className="catalog-card" key={dataset.id}><div><span className={`status-pill ${dataset.status}`}>{dataset.status}</span>{dataset.selected ? <span className="selected-pill">selected</span> : null}</div><h2>{dataset.name}</h2><code>{dataset.repository}</code><dl><div><dt>License</dt><dd>{dataset.license}</dd></div><div><dt>Roles</dt><dd>{dataset.roles.map(label).join(", ")}</dd></div></dl>{dataset.reason ? <p>{dataset.reason}</p> : null}</article>)}</div>
  </div>;
}

export function ModelCatalog({ metrics, models, onOpen }: { metrics: ConditionMetric[]; models: string[]; onOpen: (model: string) => void }) {
  return <div className="dashboard-view"><div className="view-header"><div><span className="eyebrow">Benchmark inventory</span><h1>Models</h1><p>Comparable fixed-threshold outputs across the same 15-condition SID Set gate.</p></div></div>
    <div className="model-grid">{models.map((model) => { const slices = metrics.filter((m) => m.model === model); const clean = slices.find((m) => m.condition === "clean"); const worst = [...slices].sort((a,b) => b.mismatchRate - a.mismatchRate)[0]; return <button className="model-card" key={model} onClick={() => onOpen(model)}><span className="model-index">{String(models.indexOf(model) + 1).padStart(2, "0")}</span><h2>{label(model)}</h2><div><span>Clean balanced accuracy<strong>{pct(clean?.balancedAccuracy ?? null)}</strong></span><span>Worst condition<strong>{worst ? label(worst.condition) : "—"}</strong></span><span>Worst mismatch rate<strong>{pct(worst?.mismatchRate ?? null)}</strong></span></div><small>{slices.length} evaluated conditions · open analytics →</small></button>; })}</div>
  </div>;
}
