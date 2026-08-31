#!/usr/bin/env python3
"""Generate a self-contained visual error-analysis report from detector scores."""

from __future__ import annotations

import argparse
import base64
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


OUTCOME_LABELS = {
    "false_negative": "False negative",
    "false_positive": "False positive",
    "true_positive": "True positive",
    "true_negative": "True negative",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True, help="CSV with target and score columns")
    parser.add_argument("--output", type=Path, required=True, help="Destination HTML file")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--score-column",
        help="Defaults to fused_logit, pred, probability, or logit (in that order)",
    )
    parser.add_argument(
        "--score-type",
        choices=("auto", "logit", "probability"),
        default="auto",
        help="How to interpret --score-column; auto treats columns ending in 'logit' as logits",
    )
    parser.add_argument("--image-column", default="local_path")
    parser.add_argument("--title", default="AIGC detector visual error analysis")
    parser.add_argument("--thumbnail-size", type=int, default=240)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument(
        "--max-correct",
        type=int,
        default=0,
        help="Maximum correct examples to embed; 0 includes all",
    )
    parser.add_argument(
        "--max-wrong",
        type=int,
        default=0,
        help="Maximum incorrect examples to embed; 0 includes all",
    )
    return parser.parse_args()


def _score_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in frame:
            raise ValueError(f"Score column {requested!r} is absent from the CSV.")
        return requested
    for candidate in ("fused_logit", "pred", "probability", "logit"):
        if candidate in frame:
            return candidate
    raise ValueError("Scores need one of: fused_logit, pred, probability, or logit.")


def prepare_predictions(
    frame: pd.DataFrame,
    *,
    score_column: str | None = None,
    score_type: str = "auto",
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, str, str]:
    """Validate score rows and add probabilities, decisions, and outcome labels."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1].")
    if "target" not in frame:
        raise ValueError("Scores CSV must contain binary ground truth in 'target'.")
    column = _score_column(frame, score_column)
    kind = score_type
    if kind == "auto":
        kind = "logit" if column.casefold().endswith("logit") else "probability"
    if kind not in {"logit", "probability"}:
        raise ValueError("score_type must be auto, logit, or probability.")

    result = frame.copy()
    target = pd.to_numeric(result["target"], errors="raise").to_numpy()
    if not np.isin(target, (0, 1)).all():
        raise ValueError("'target' must contain only 0 (authentic) and 1 (AIGC).")
    values = pd.to_numeric(result[column], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{column!r} contains non-finite values.")
    if kind == "logit":
        probability = 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))
    else:
        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"Probability column {column!r} must lie in [0, 1].")
        probability = values

    predicted = (probability >= threshold).astype(np.int8)
    outcome = np.select(
        (
            (target == 1) & (predicted == 0),
            (target == 0) & (predicted == 1),
            (target == 1) & (predicted == 1),
        ),
        ("false_negative", "false_positive", "true_positive"),
        default="true_negative",
    )
    result["aigc_probability"] = probability
    result["predicted_target"] = predicted
    result["correct"] = predicted == target
    result["outcome"] = outcome
    result["decision_margin"] = np.abs(probability - threshold)
    return result, column, kind


def _image_data_uri(path: Path, *, size: int, quality: int) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Report image does not exist: {path}")
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _resolved_path(value: object, repo_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _display(value: object, fallback: str = "—") -> str:
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    return text or fallback


def _summary_card(label: str, value: object, tone: str = "") -> str:
    return (
        f'<div class="metric {escape(tone)}"><span>{escape(label)}</span>'
        f'<strong>{escape(str(value))}</strong></div>'
    )


def _group_table(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        return ""
    rows = []
    grouped = frame.groupby(column, dropna=False, sort=False)
    for value, group in grouped:
        counts = group["outcome"].value_counts()
        rows.append({
            "group": _display(value),
            "n": len(group),
            "accuracy": float(group["correct"].mean()),
            "false_positive": int(counts.get("false_positive", 0)),
            "false_negative": int(counts.get("false_negative", 0)),
            "mean_score": float(group["aigc_probability"].mean()),
        })
    rows.sort(key=lambda item: (-item["n"], item["group"]))
    body = "".join(
        "<tr>"
        f'<td>{escape(item["group"])}</td>'
        f'<td>{item["n"]:,}</td>'
        f'<td>{item["accuracy"]:.1%}</td>'
        f'<td>{item["false_positive"]:,}</td>'
        f'<td>{item["false_negative"]:,}</td>'
        f'<td>{item["mean_score"]:.3f}</td>'
        "</tr>"
        for item in rows
    )
    return f"""
      <section class="panel">
        <h2>Breakdown by {escape(column.replace('_', ' '))}</h2>
        <div class="table-wrap"><table>
          <thead><tr><th>Group</th><th>Images</th><th>Accuracy</th><th>FP</th><th>FN</th><th>Mean AIGC score</th></tr></thead>
          <tbody>{body}</tbody>
        </table></div>
      </section>
    """


def _histogram(frame: pd.DataFrame) -> str:
    boundaries = np.linspace(0.0, 1.0, 11)
    correct = frame.loc[frame["correct"], "aigc_probability"].to_numpy()
    wrong = frame.loc[~frame["correct"], "aigc_probability"].to_numpy()
    correct_counts, _ = np.histogram(correct, boundaries)
    wrong_counts, _ = np.histogram(wrong, boundaries)
    maximum = max(int(correct_counts.max(initial=0)), int(wrong_counts.max(initial=0)), 1)
    rows = []
    for index, (good, bad) in enumerate(zip(correct_counts, wrong_counts, strict=True)):
        left, right = boundaries[index:index + 2]
        rows.append(f"""
          <div class="hist-row">
            <span>{left:.1f}–{right:.1f}</span>
            <div class="bars">
              <div class="bar good" style="width:{100 * int(good) / maximum:.2f}%">{int(good)}</div>
              <div class="bar bad" style="width:{100 * int(bad) / maximum:.2f}%">{int(bad)}</div>
            </div>
          </div>
        """)
    return "".join(rows)


def _select_options(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        return ""
    values = sorted({_display(value) for value in frame[column]})
    return "".join(
        f'<option value="{escape(value.casefold(), quote=True)}">{escape(value)}</option>'
        for value in values
    )


def _metadata_rows(row: pd.Series) -> str:
    fields = (
        ("Asset", "parent_id"),
        ("Source", "source_dataset"),
        ("Family", "generator_family"),
        ("Model", "generation_model"),
        ("Noise", "noise_sigma"),
        ("Blockiness", "blockiness"),
        ("HF energy", "structural_hf_energy"),
    )
    values = []
    for label, column in fields:
        if column not in row or pd.isna(row[column]) or str(row[column]).strip() == "":
            continue
        value = row[column]
        rendered = f"{float(value):.4g}" if isinstance(value, (float, np.floating)) else str(value)
        values.append(f"<dt>{escape(label)}</dt><dd>{escape(rendered)}</dd>")
    return "".join(values)


def _image_card(
    row: pd.Series,
    *,
    repo_root: Path,
    image_column: str,
    thumbnail_size: int,
    jpeg_quality: int,
) -> str:
    path_value = row[image_column]
    path = _resolved_path(path_value, repo_root)
    image_uri = _image_data_uri(path, size=thumbnail_size, quality=jpeg_quality)
    truth = "AIGC" if int(row["target"]) else "Authentic"
    prediction = "AIGC" if int(row["predicted_target"]) else "Authentic"
    outcome = str(row["outcome"])
    correctness = "wrong" if not bool(row["correct"]) else "correct"
    source = _display(row.get("source_dataset", ""), "unknown")
    family = _display(row.get("generator_family", ""), "unknown")
    searchable = " ".join(
        _display(row.get(column, ""), "")
        for column in ("parent_id", "source_dataset", "generator_family", "generation_model", image_column)
    ).casefold()
    return f"""
      <article class="image-card {correctness}"
        data-correctness="{correctness}"
        data-outcome="{escape(outcome, quote=True)}"
        data-truth="{int(row['target'])}"
        data-family="{escape(family.casefold(), quote=True)}"
        data-source="{escape(source.casefold(), quote=True)}"
        data-search="{escape(searchable, quote=True)}"
        data-score="{float(row['aigc_probability']):.12f}"
        data-margin="{float(row['decision_margin']):.12f}">
        <div class="image-shell">
          <img loading="lazy" src="{image_uri}" alt="{escape(_display(row.get('parent_id', 'image')), quote=True)}">
          <span class="outcome-badge">{escape(OUTCOME_LABELS[outcome])}</span>
        </div>
        <div class="card-body">
          <div class="score-line"><strong>{float(row['aigc_probability']):.3f}</strong><span>AIGC score</span></div>
          <div class="truth-line"><span>Truth: <b>{truth}</b></span><span>Predicted: <b>{prediction}</b></span></div>
          <dl>{_metadata_rows(row)}</dl>
          <p class="path" title="{escape(str(path_value), quote=True)}">{escape(str(path_value))}</p>
        </div>
      </article>
    """


def _limited_rows(frame: pd.DataFrame, *, correct: bool, maximum: int) -> pd.DataFrame:
    selected = frame[frame["correct"].eq(correct)].sort_values(
        ["decision_margin", "aigc_probability"], ascending=[False, False]
    )
    return selected if maximum <= 0 else selected.head(maximum)


def _gallery(
    rows: Iterable[tuple[object, pd.Series]],
    *,
    repo_root: Path,
    image_column: str,
    thumbnail_size: int,
    jpeg_quality: int,
) -> str:
    return "".join(
        _image_card(
            row,
            repo_root=repo_root,
            image_column=image_column,
            thumbnail_size=thumbnail_size,
            jpeg_quality=jpeg_quality,
        )
        for _, row in rows
    )


def generate_report(
    scores_path: Path,
    output_path: Path,
    *,
    repo_root: Path,
    threshold: float = 0.5,
    score_column: str | None = None,
    score_type: str = "auto",
    image_column: str = "local_path",
    title: str = "AIGC detector visual error analysis",
    thumbnail_size: int = 240,
    jpeg_quality: int = 82,
    max_correct: int = 0,
    max_wrong: int = 0,
) -> dict[str, object]:
    """Create the report and return its main metrics for tests and automation."""

    if thumbnail_size < 32:
        raise ValueError("thumbnail_size must be at least 32 pixels.")
    if not 1 <= jpeg_quality <= 95:
        raise ValueError("jpeg_quality must lie in [1, 95].")
    if max_correct < 0 or max_wrong < 0:
        raise ValueError("Image limits cannot be negative.")
    frame = pd.read_csv(scores_path)
    if image_column not in frame:
        raise ValueError(f"Image column {image_column!r} is absent from the CSV.")
    prepared, resolved_score_column, resolved_score_type = prepare_predictions(
        frame,
        score_column=score_column,
        score_type=score_type,
        threshold=threshold,
    )
    counts = prepared["outcome"].value_counts()
    false_negatives = int(counts.get("false_negative", 0))
    false_positives = int(counts.get("false_positive", 0))
    true_positives = int(counts.get("true_positive", 0))
    true_negatives = int(counts.get("true_negative", 0))
    wrong = false_negatives + false_positives
    correct = true_positives + true_negatives
    accuracy = correct / len(prepared) if len(prepared) else float("nan")
    wrong_rows = _limited_rows(prepared, correct=False, maximum=max_wrong)
    correct_rows = _limited_rows(prepared, correct=True, maximum=max_correct)
    detector_hash = (
        _display(prepared["detector_sha256"].iloc[0])
        if "detector_sha256" in prepared and prepared["detector_sha256"].nunique() == 1
        else "not recorded"
    )

    metrics = "".join((
        _summary_card("Images", f"{len(prepared):,}"),
        _summary_card("Accuracy", f"{accuracy:.1%}"),
        _summary_card("Correct", f"{correct:,}", "positive"),
        _summary_card("Incorrect", f"{wrong:,}", "negative"),
        _summary_card("False negatives", f"{false_negatives:,}", "negative"),
        _summary_card("False positives", f"{false_positives:,}", "warning"),
    ))
    group_tables = _group_table(prepared, "generator_family") + _group_table(
        prepared, "source_dataset"
    )
    wrong_gallery = _gallery(
        wrong_rows.iterrows(),
        repo_root=repo_root,
        image_column=image_column,
        thumbnail_size=thumbnail_size,
        jpeg_quality=jpeg_quality,
    )
    correct_gallery = _gallery(
        correct_rows.iterrows(),
        repo_root=repo_root,
        image_column=image_column,
        thumbnail_size=thumbnail_size,
        jpeg_quality=jpeg_quality,
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0b1020; --panel:#121a2f; --panel2:#18223d; --text:#f3f6ff; --muted:#a9b3cf; --line:#2b3759; --good:#45d19a; --bad:#ff6b7c; --warn:#f6c760; --accent:#8da2ff; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at 20% -10%,#263866 0,transparent 36rem),var(--bg); color:var(--text); font:15px/1.45 Inter,ui-sans-serif,system-ui,sans-serif; }}
    main {{ width:min(1500px,calc(100% - 32px)); margin:auto; padding:42px 0 72px; }}
    h1 {{ margin:.15rem 0 .5rem; font-size:clamp(2rem,4vw,3.7rem); line-height:1.04; letter-spacing:-.045em; }}
    h2 {{ margin:0 0 18px; font-size:1.35rem; }}
    p {{ color:var(--muted); }}
    code {{ overflow-wrap:anywhere; color:#dce3ff; }}
    .eyebrow {{ color:var(--accent); text-transform:uppercase; letter-spacing:.14em; font-weight:800; font-size:.76rem; }}
    .lede {{ max-width:920px; font-size:1.04rem; }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:26px 0; }}
    .metric,.panel,.controls {{ background:linear-gradient(145deg,rgba(24,34,61,.96),rgba(16,24,45,.96)); border:1px solid var(--line); border-radius:16px; box-shadow:0 14px 35px rgba(0,0,0,.16); }}
    .metric {{ padding:16px; }} .metric span {{ display:block; color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }} .metric strong {{ display:block; margin-top:4px; font-size:1.7rem; }}
    .metric.positive strong {{ color:var(--good); }} .metric.negative strong {{ color:var(--bad); }} .metric.warning strong {{ color:var(--warn); }}
    .panel {{ padding:22px; margin:18px 0; }}
    .audit-note {{ border-left:4px solid var(--warn); }}
    .two-column {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:18px; }}
    .hist-row {{ display:grid; grid-template-columns:70px 1fr; gap:10px; align-items:center; margin:7px 0; color:var(--muted); font-variant-numeric:tabular-nums; }}
    .bars {{ display:grid; gap:3px; }} .bar {{ min-width:2px; border-radius:4px; padding:1px 5px; color:#08101d; font-size:.72rem; font-weight:800; }} .bar.good {{ background:var(--good); }} .bar.bad {{ background:var(--bad); }}
    .legend {{ display:flex; gap:18px; color:var(--muted); font-size:.84rem; }} .legend i {{ display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:5px; }}
    .table-wrap {{ overflow:auto; }} table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }} th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }} th:first-child,td:first-child {{ text-align:left; }} th {{ color:var(--muted); font-size:.77rem; text-transform:uppercase; letter-spacing:.05em; }}
    .controls {{ position:sticky; top:8px; z-index:10; display:grid; grid-template-columns:2fr repeat(4,minmax(120px,1fr)); gap:10px; padding:14px; margin:28px 0 20px; backdrop-filter:blur(12px); }}
    input,select {{ width:100%; padding:10px 12px; background:#0d1428; color:var(--text); border:1px solid var(--line); border-radius:9px; }}
    .section-title {{ display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin:34px 0 14px; }} .section-title h2 {{ margin:0; }} .section-title span {{ color:var(--muted); }}
    .gallery {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(235px,1fr)); gap:14px; }}
    .image-card {{ min-width:0; overflow:hidden; background:var(--panel); border:1px solid var(--line); border-radius:14px; }} .image-card.wrong {{ border-color:rgba(255,107,124,.62); }} .image-card[hidden] {{ display:none; }}
    .image-shell {{ position:relative; display:grid; place-items:center; height:240px; background:#070b15; }} .image-shell img {{ width:100%; height:100%; object-fit:contain; }}
    .outcome-badge {{ position:absolute; top:9px; left:9px; padding:5px 8px; border-radius:7px; background:rgba(5,10,22,.88); font-size:.74rem; font-weight:850; text-transform:uppercase; letter-spacing:.05em; }} .wrong .outcome-badge {{ color:var(--bad); }} .correct .outcome-badge {{ color:var(--good); }}
    .card-body {{ padding:13px; }} .score-line {{ display:flex; align-items:baseline; gap:8px; }} .score-line strong {{ font-size:1.55rem; }} .score-line span,.truth-line,.path {{ color:var(--muted); font-size:.8rem; }} .truth-line {{ display:flex; justify-content:space-between; gap:8px; margin:4px 0 11px; }}
    dl {{ display:grid; grid-template-columns:auto 1fr; gap:3px 8px; margin:0; font-size:.78rem; }} dt {{ color:var(--muted); }} dd {{ margin:0; overflow-wrap:anywhere; }} .path {{ overflow:hidden; white-space:nowrap; text-overflow:ellipsis; margin:10px 0 0; }}
    .empty {{ display:none; padding:34px; text-align:center; color:var(--muted); }}
    @media(max-width:850px) {{ .two-column {{ grid-template-columns:1fr; }} .controls {{ position:static; grid-template-columns:1fr 1fr; }} .controls input {{ grid-column:1/-1; }} }}
  </style>
</head>
<body><main>
  <div class="eyebrow">Model-facing visual EDA</div>
  <h1>{escape(title)}</h1>
  <p class="lede">The report embeds the exact images referenced by <code>{escape(str(scores_path))}</code>. Wrong predictions are ordered by confidence first so the strongest model failures are easiest to inspect.</p>
  <div class="metrics">{metrics}</div>
  <section class="panel audit-note">
    <h2>Interpretation guardrails</h2>
    <p>The AIGC score is derived from <code>{escape(resolved_score_column)}</code> as a {escape(resolved_score_type)} and classified at {threshold:.3f}. This run is not calibrated, so 0.5 accuracy is an operating-point diagnostic; ranking metrics and score distributions remain important. Detector SHA-256: <code>{escape(detector_hash)}</code>.</p>
  </section>
  <div class="two-column">
    <section class="panel"><h2>AIGC score distribution</h2><div class="legend"><span><i style="background:var(--good)"></i>Correct</span><span><i style="background:var(--bad)"></i>Wrong</span></div>{_histogram(prepared)}</section>
    <section class="panel"><h2>Confusion matrix</h2><div class="table-wrap"><table><thead><tr><th>Truth</th><th>Predicted authentic</th><th>Predicted AIGC</th></tr></thead><tbody><tr><td>Authentic</td><td>{true_negatives:,} TN</td><td>{false_positives:,} FP</td></tr><tr><td>AIGC</td><td>{false_negatives:,} FN</td><td>{true_positives:,} TP</td></tr></tbody></table></div></section>
  </div>
  {group_tables}
  <div class="controls">
    <input id="search" type="search" placeholder="Search asset, source, family, model, or path">
    <select id="outcome"><option value="all">All outcomes</option><option value="wrong">All wrong</option><option value="correct">All correct</option><option value="false_negative">False negatives</option><option value="false_positive">False positives</option><option value="true_positive">True positives</option><option value="true_negative">True negatives</option></select>
    <select id="truth"><option value="all">All truth labels</option><option value="1">Truth: AIGC</option><option value="0">Truth: authentic</option></select>
    <select id="family"><option value="all">All families</option>{_select_options(prepared, 'generator_family')}</select>
    <select id="source"><option value="all">All sources</option>{_select_options(prepared, 'source_dataset')}</select>
  </div>
  <section id="wrong-section">
    <div class="section-title"><h2>Incorrect predictions</h2><span><b id="wrong-visible">{len(wrong_rows):,}</b> shown · {wrong:,} total</span></div>
    <div class="gallery">{wrong_gallery}</div>
  </section>
  <section id="correct-section">
    <div class="section-title"><h2>Correct predictions</h2><span><b id="correct-visible">{len(correct_rows):,}</b> shown · {correct:,} total</span></div>
    <div class="gallery">{correct_gallery}</div>
  </section>
  <div id="empty" class="empty">No images match the current filters.</div>
</main>
<script>
  const controls = ['search','outcome','truth','family','source'].map(id => document.getElementById(id));
  const cards = [...document.querySelectorAll('.image-card')];
  function applyFilters() {{
    const query = document.getElementById('search').value.trim().toLowerCase();
    const outcome = document.getElementById('outcome').value;
    const truth = document.getElementById('truth').value;
    const family = document.getElementById('family').value;
    const source = document.getElementById('source').value;
    let visible = 0, correct = 0, wrong = 0;
    for (const card of cards) {{
      const outcomeMatch = outcome === 'all' || card.dataset.outcome === outcome || card.dataset.correctness === outcome;
      const show = outcomeMatch && (truth === 'all' || card.dataset.truth === truth) && (family === 'all' || card.dataset.family === family) && (source === 'all' || card.dataset.source === source) && (!query || card.dataset.search.includes(query));
      card.hidden = !show;
      if (show) {{ visible++; card.dataset.correctness === 'correct' ? correct++ : wrong++; }}
    }}
    document.getElementById('correct-visible').textContent = correct.toLocaleString();
    document.getElementById('wrong-visible').textContent = wrong.toLocaleString();
    document.getElementById('correct-section').hidden = correct === 0;
    document.getElementById('wrong-section').hidden = wrong === 0;
    document.getElementById('empty').style.display = visible === 0 ? 'block' : 'none';
  }}
  controls.forEach(control => control.addEventListener(control.id === 'search' ? 'input' : 'change', applyFilters));
</script>
</body></html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return {
        "rows": len(prepared),
        "accuracy": accuracy,
        "correct": correct,
        "wrong": wrong,
        "false_negative": false_negatives,
        "false_positive": false_positives,
        "true_positive": true_positives,
        "true_negative": true_negatives,
        "embedded_correct": len(correct_rows),
        "embedded_wrong": len(wrong_rows),
        "score_column": resolved_score_column,
        "score_type": resolved_score_type,
    }


def main() -> None:
    args = parse_args()
    metrics = generate_report(
        args.scores.resolve(),
        args.output.resolve(),
        repo_root=args.repo_root.resolve(),
        threshold=args.threshold,
        score_column=args.score_column,
        score_type=args.score_type,
        image_column=args.image_column,
        title=args.title,
        thumbnail_size=args.thumbnail_size,
        jpeg_quality=args.jpeg_quality,
        max_correct=args.max_correct,
        max_wrong=args.max_wrong,
    )
    print("Generated", args.output, metrics)


if __name__ == "__main__":
    main()
