import { promises as fs } from "node:fs";
import path from "node:path";
import type { AnalyticsPayload, ConditionMetric, DatasetRecord, Truth } from "@/lib/types";

type Row = { sample_id: string; probability_ai: number; decision: number };
type Slice = { model: string; condition: string; rows: Row[] };

export async function loadAnalytics(
  root: string,
  modelDirs: string[],
  conditions: string[],
  truthBySource: Map<string, Truth>,
): Promise<{ analytics: AnalyticsPayload; datasets: DatasetRecord[] }> {
  const predictionRoot = path.join(root, "work/predictions");
  const slices = (await Promise.all(modelDirs.flatMap((model) => conditions.map(async (condition) => {
    const file = path.join(predictionRoot, model, "sid_set", `${condition}.jsonl`);
    try {
      const raw = await fs.readFile(file, "utf8");
      const rows = raw.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line) as Row);
      const modified = (await fs.stat(file)).mtimeMs;
      return { model, condition, rows, modified };
    } catch { return null; }
  })))).filter((slice): slice is Slice & { modified: number } => Boolean(slice));

  const flagsBySlice = new Map<string, Map<string, boolean>>();
  const metrics: ConditionMetric[] = slices.map((slice) => {
    const confusion = { tp: 0, tn: 0, fp: 0, fn: 0, total: 0 };
    const flags = new Map<string, boolean>();
    let probabilitySum = 0;
    for (const row of slice.rows) {
      const truth = truthBySource.get(row.sample_id);
      if (!truth || truth === "unknown") continue;
      const predictedAi = Boolean(row.decision);
      if (truth === "ai") predictedAi ? confusion.tp++ : confusion.fn++;
      else predictedAi ? confusion.fp++ : confusion.tn++;
      confusion.total++;
      probabilitySum += row.probability_ai;
      flags.set(row.sample_id, predictedAi !== (truth === "ai"));
    }
    flagsBySlice.set(`${slice.model}:${slice.condition}`, flags);
    const positives = confusion.tp + confusion.fn;
    const negatives = confusion.tn + confusion.fp;
    return {
      model: slice.model,
      condition: slice.condition,
      confusion,
      mismatchRate: confusion.total ? (confusion.fp + confusion.fn) / confusion.total : 0,
      falsePositiveRate: negatives ? confusion.fp / negatives : 0,
      falseNegativeRate: positives ? confusion.fn / positives : 0,
      balancedAccuracy: ((positives ? confusion.tp / positives : 0) + (negatives ? confusion.tn / negatives : 0)) / 2,
      meanProbabilityAi: confusion.total ? probabilitySum / confusion.total : 0,
      cleanMismatchCorrelation: null,
      mismatchLift: null,
      newlyWrongRate: null,
      recoveredRate: null,
    };
  });

  for (const metric of metrics) {
    if (metric.condition === "clean") continue;
    const cleanMetric = metrics.find((item) => item.model === metric.model && item.condition === "clean");
    const clean = flagsBySlice.get(`${metric.model}:clean`);
    const transformed = flagsBySlice.get(`${metric.model}:${metric.condition}`);
    if (!cleanMetric || !clean || !transformed) continue;
    let n = 0, sx = 0, sy = 0, sxy = 0, correct = 0, wrong = 0, newlyWrong = 0, recovered = 0;
    for (const [id, cleanMismatch] of clean) {
      const transformedMismatch = transformed.get(id);
      if (transformedMismatch === undefined) continue;
      const x = Number(cleanMismatch), y = Number(transformedMismatch);
      n++; sx += x; sy += y; sxy += x * y;
      if (cleanMismatch) { wrong++; if (!transformedMismatch) recovered++; }
      else { correct++; if (transformedMismatch) newlyWrong++; }
    }
    const denominator = Math.sqrt(sx * (n - sx) * sy * (n - sy));
    metric.cleanMismatchCorrelation = denominator ? (n * sxy - sx * sy) / denominator : null;
    metric.mismatchLift = metric.mismatchRate - cleanMetric.mismatchRate;
    metric.newlyWrongRate = correct ? newlyWrong / correct : null;
    metric.recoveredRate = wrong ? recovered / wrong : null;
  }

  const registry = JSON.parse(await fs.readFile(path.join(root, "configs/datasets.grouped.json"), "utf8")) as { datasets: DatasetRecord[] };
  return {
    analytics: {
      metrics,
      evaluatedRows: metrics.reduce((sum, item) => sum + item.confusion.total, 0),
      samplesPerSlice: Math.max(0, ...metrics.map((item) => item.confusion.total)),
      updatedAt: slices.length ? new Date(Math.max(...slices.map((slice) => slice.modified))).toISOString() : null,
    },
    datasets: registry.datasets,
  };
}
