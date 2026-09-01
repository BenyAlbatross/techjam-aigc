"use client";

import { Check, ChevronDown, RotateCcw } from "lucide-react";
import { useMemo, useState, type CSSProperties } from "react";
import type { ConditionMetric, DatasetRecord } from "@/lib/types";

const pct = (value: number | null) => value === null ? "—" : `${(value * 100).toFixed(1)}%`;
const signed = (value: number | null) => value === null ? "—" : `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)} pp`;
const label = (value: string) => value.replaceAll("_", " ");

export function MultiSelect({ title, values, selected, onChange }: { title: string; values: string[]; selected: string[]; onChange: (values: string[]) => void }) {
  function toggle(value: string) {
    const next = selected.includes(value) ? selected.filter((item) => item !== value) : [...selected, value];
    if (next.length) onChange(next);
  }
  const summary = selected.length === values.length ? "All" : selected.length === 1 ? label(selected[0]) : `${selected.length} selected`;
  return <details className="multi-select">
    <summary><span><small>{title}</small><strong>{summary}</strong></span><ChevronDown size={14} /></summary>
    <div className="multi-menu">
      <div className="multi-actions"><button onClick={() => onChange(values)}>Select all</button><button onClick={() => onChange([values[0]])}>Only first</button></div>
      {values.map((value) => <label key={value}><input type="checkbox" checked={selected.includes(value)} onChange={() => toggle(value)} /><span className="check-box">{selected.includes(value) ? <Check size={12} /> : null}</span><span>{label(value)}</span></label>)}
    </div>
  </details>;
}

export function AnalyticsDashboard({ metrics, models, conditions, model, condition, onModel, onCondition, onDrilldown, evaluatedRows, updatedAt }: {
  metrics: ConditionMetric[]; models: string[]; conditions: string[]; model: string; condition: string;
  onModel: (value: string) => void; onCondition: (value: string) => void; onDrilldown: (outcome: "mismatch" | "fp" | "fn") => void; evaluatedRows: number; updatedAt: string | null;
}) {
  const [selectedModels, setSelectedModels] = useState(() => models);
  const [selectedConditions, setSelectedConditions] = useState(() => conditions);
  const selectedMetrics = useMemo(() => metrics.filter((item) => selectedModels.includes(item.model) && selectedConditions.includes(item.condition)), [metrics, selectedConditions, selectedModels]);
  const c = useMemo(() => selectedMetrics.reduce((sum, item) => ({ tp: sum.tp + item.confusion.tp, tn: sum.tn + item.confusion.tn, fp: sum.fp + item.confusion.fp, fn: sum.fn + item.confusion.fn, total: sum.total + item.confusion.total }), { tp: 0, tn: 0, fp: 0, fn: 0, total: 0 }), [selectedMetrics]);
  const positives = c.tp + c.fn, negatives = c.tn + c.fp;
  const balancedAccuracy = positives && negatives ? (c.tp / positives + c.tn / negatives) / 2 : null;
  const visibleMetrics = [...selectedMetrics].sort((a, b) => b.mismatchRate - a.mismatchRate);
  const reset = () => { setSelectedModels(models); setSelectedConditions(conditions); };

  if (!selectedMetrics.length) return <div className="empty-view">No benchmark analytics available.</div>;
  return <div className="dashboard-view">
    <h1 className="visually-hidden">Robustness analytics</h1>
    <div className="analytics-filterbar">
      <MultiSelect title="Models" values={models} selected={selectedModels} onChange={(next) => { setSelectedModels(next); onModel(next[0]); }} />
      <MultiSelect title="Conditions" values={conditions} selected={selectedConditions} onChange={(next) => { setSelectedConditions(next); onCondition(next[0]); }} />
      <button className="reset-filters" onClick={reset}><RotateCcw size={14} /> Reset</button>
      <span className="filter-scope">{selectedMetrics.length} slices · {c.total.toLocaleString()} scored rows</span>
    </div>
    <div className="source-strip"><strong>Source</strong><span>local SID Set shards + imported TRACE-RX evaluation</span><strong>Available coverage</strong><span>{evaluatedRows.toLocaleString()} rows</span><strong>Updated</strong><span>{updatedAt ? new Date(updatedAt).toLocaleString() : "Unknown"}</span></div>
    <div className="kpi-grid">
      <article><span>Balanced accuracy</span><strong>{pct(balancedAccuracy)}</strong><small>selected slices</small></article>
      <article><span>Mismatches</span><strong>{(c.fp + c.fn).toLocaleString()}</strong><small>{pct(c.total ? (c.fp + c.fn) / c.total : 0)} of selected rows</small></article>
      <article><span>False-positive rate</span><strong>{pct(negatives ? c.fp / negatives : 0)}</strong><small>authentic predicted AI</small></article>
      <article><span>False-negative rate</span><strong>{pct(positives ? c.fn / positives : 0)}</strong><small>AI predicted authentic</small></article>
    </div>
    <div className="analytics-grid">
      <section className="analytics-card confusion-card"><div className="card-head"><div><h2>Truth × prediction matrix</h2><small>Aggregated across selected slices</small></div><button onClick={() => onDrilldown("mismatch")}>View gallery</button></div>
        <div className="matrix-axis">Predicted</div><div className="confusion-matrix">
          <span></span><strong>Authentic</strong><strong>AI-generated</strong>
          <strong>Authentic truth</strong><button className="correct" title="True negative">{c.tn.toLocaleString()}<small>TN</small></button><button className="incorrect" onClick={() => onDrilldown("fp")}>{c.fp.toLocaleString()}<small>FP</small></button>
          <strong>AI truth</strong><button className="incorrect" onClick={() => onDrilldown("fn")}>{c.fn.toLocaleString()}<small>FN</small></button><button className="correct" title="True positive">{c.tp.toLocaleString()}<small>TP</small></button>
        </div>
      </section>
      <section className="analytics-card"><div className="card-head"><div><h2>Mismatch rate by slice</h2><small>Model and condition combinations</small></div></div>
        <div className="bar-list">{visibleMetrics.map((item) => <button key={`${item.model}-${item.condition}`} onClick={() => { onModel(item.model); onCondition(item.condition); setSelectedModels([item.model]); setSelectedConditions([item.condition]); }}><span>{label(item.model)} · {label(item.condition)}</span><i><b style={{ width: `${Math.max(2, item.mismatchRate * 100)}%` }} /></i><strong>{pct(item.mismatchRate)}</strong></button>)}</div>
      </section>
    </div>
    <section className="analytics-card condition-table-card"><div className="card-head"><div><h2>Condition associations with mismatches</h2><small>Paired with each model’s clean predictions</small></div><small>association, not causation</small></div>
      <div className="metric-table"><div className="metric-row heading"><span>Model / condition</span><span>Error</span><span>Lift vs clean</span><span>Clean-error φ</span><span>Newly wrong</span><span>Recovered</span></div>
      {visibleMetrics.map((item) => <button className="metric-row" key={`${item.model}-${item.condition}`} onClick={() => { setSelectedModels([item.model]); setSelectedConditions([item.condition]); onModel(item.model); onCondition(item.condition); }}><strong>{label(item.model)} · {label(item.condition)}</strong><span>{pct(item.mismatchRate)}</span><span>{item.condition === "clean" ? "baseline" : signed(item.mismatchLift)}</span><span>{item.cleanMismatchCorrelation?.toFixed(2) ?? "—"}</span><span>{pct(item.newlyWrongRate)}</span><span>{pct(item.recoveredRate)}</span></button>)}</div>
    </section>
    <section className="analytics-card heatmap-card"><div className="card-head"><div><h2>Model × condition error heatmap</h2><small>Only selected models and conditions</small></div></div>
      <div className="heatmap" style={{ gridTemplateColumns: `150px repeat(${selectedConditions.length}, minmax(54px, 1fr))` }}><span></span>{selectedConditions.map((item) => <strong key={item} title={label(item)}>{label(item)}</strong>)}{selectedModels.flatMap((m) => [<b key={m}>{label(m)}</b>, ...selectedConditions.map((cond) => { const metric = metrics.find((x) => x.model === m && x.condition === cond); const value = metric?.mismatchRate ?? null; return <button key={m + "-" + cond} disabled={!metric} style={{ "--heat": value ?? 0 } as CSSProperties} onClick={() => { setSelectedModels([m]); setSelectedConditions([cond]); onModel(m); onCondition(cond); }} title={`${label(m)} · ${label(cond)}: ${pct(value)}`}>{pct(value)}</button>; })])}</div>
    </section>
  </div>;
}

export function DatasetCatalog({ datasets }: { datasets: DatasetRecord[] }) {
  return <div className="dashboard-view"><div className="view-header"><div><h1>Datasets</h1><p>Grouped by approval state. Grouping never grants permission.</p></div></div>
    <div className="status-summary">{(["approved", "review", "blocked"] as const).map((status) => <article key={status}><strong>{datasets.filter((d) => d.status === status).length}</strong><span>{status}</span></article>)}</div>
    <div className="catalog-grid">{datasets.map((dataset) => <article className="catalog-card" key={dataset.id}><div><span className={`status-pill ${dataset.status}`}>{dataset.status}</span>{dataset.selected ? <span className="selected-pill">selected</span> : null}</div><h2>{dataset.name}</h2><code>{dataset.repository}</code><dl><div><dt>License</dt><dd>{dataset.license}</dd></div><div><dt>Roles</dt><dd>{dataset.roles.map(label).join(", ")}</dd></div></dl>{dataset.reason ? <p>{dataset.reason}</p> : null}</article>)}</div>
  </div>;
}

export function ModelCatalog({ metrics, models, onOpen }: { metrics: ConditionMetric[]; models: string[]; onOpen: (model: string) => void }) {
  return <div className="dashboard-view"><div className="view-header"><div><h1>Models</h1><p>Available fixed-threshold evaluation slices.</p></div></div>
    <div className="model-grid">{models.map((model) => { const slices = metrics.filter((m) => m.model === model); const clean = slices.find((m) => m.condition === "clean"); const worst = [...slices].sort((a,b) => b.mismatchRate - a.mismatchRate)[0]; return <button className="model-card" key={model} onClick={() => onOpen(model)}><span className="model-index">{String(models.indexOf(model) + 1).padStart(2, "0")}</span><h2>{label(model)}</h2><div><span>Clean balanced accuracy<strong>{pct(clean?.balancedAccuracy ?? null)}</strong></span><span>Worst condition<strong>{worst ? label(worst.condition) : "—"}</strong></span><span>Worst mismatch rate<strong>{pct(worst?.mismatchRate ?? null)}</strong></span></div><small>{slices.length} evaluated conditions · open analytics →</small></button>; })}</div>
  </div>;
}
