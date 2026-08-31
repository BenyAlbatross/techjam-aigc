import "server-only";

import { existsSync, promises as fs } from "node:fs";
import path from "node:path";
import type { GalleryImage, GalleryPayload, Prediction, Truth } from "@/lib/types";
import { loadAnalytics } from "@/lib/analytics";

const PROJECT_ROOT = [
  process.env.TRACE_PROJECT_ROOT,
  path.resolve(process.cwd(), ".."),
  path.resolve(process.cwd(), "../../.."),
].filter((candidate): candidate is string => Boolean(candidate))
  .find((candidate) => existsSync(path.join(candidate, "work/manifests/sid_set_1000x2_canonical.json")))
  ?? path.resolve(process.cwd(), "..");
const MANIFEST = path.join(PROJECT_ROOT, "work/manifests/sid_set_1000x2_canonical.json");
const PREDICTIONS = path.join(PROJECT_ROOT, "work/predictions");
const LIMIT = 72;
const CONDITION_ORDER = [
  "clean", "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30",
  "blur_sigma0.5", "blur_sigma1", "blur_sigma2", "resize_0.5", "resize_0.25",
  "noise_sigma0.02", "noise_sigma0.05", "noise_sigma0.10", "color_jitter_20",
  "center_crop_80",
];

type ManifestSample = {
  sample_id: string;
  source_sample_id?: string;
  base_id?: string;
  truth?: "real" | "ai";
  label?: number;
  path: string;
  source_family?: string;
  generator_family?: string;
  license?: string;
  width?: number;
  height?: number;
  file_format?: string;
  sha256: string;
};

type PredictionRow = {
  sample_id: string;
  model: string;
  condition: string;
  probability_ai: number;
  threshold: number;
  decision: number;
  condition_parameters?: Record<string, unknown>;
};

async function readJsonLines(file: string): Promise<PredictionRow[]> {
  const raw = await fs.readFile(file, "utf8");
  return raw.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line) as PredictionRow);
}

export async function loadGallery(): Promise<GalleryPayload> {
  try {
    const manifest = JSON.parse(await fs.readFile(MANIFEST, "utf8")) as {
      dataset?: string;
      license?: string;
      samples: ManifestSample[];
    };
    const selected = manifest.samples.slice(0, LIMIT);
    const truthBySource = new Map<string, Truth>(manifest.samples.map((sample) => [sample.source_sample_id ?? sample.sample_id, sample.truth ?? (sample.label === 1 ? "ai" : "real")]));
    const canonicalBySource = new Map(
      selected.map((sample) => [sample.source_sample_id ?? sample.sample_id, sample.sample_id]),
    );
    const modelDirs = (await fs.readdir(PREDICTIONS, { withFileTypes: true }))
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .sort();
    const byImage = new Map<string, Prediction[]>();
    const conditions = new Set<string>();

    await Promise.all(modelDirs.flatMap((model) => CONDITION_ORDER.map(async (condition) => {
      const shard = path.join(PREDICTIONS, model, "sid_set", `${condition}.jsonl`);
      try {
        for (const row of await readJsonLines(shard)) {
          const canonicalId = canonicalBySource.get(row.sample_id);
          if (!canonicalId) continue;
          conditions.add(row.condition);
          const values = byImage.get(canonicalId) ?? [];
          values.push({
            model: row.model,
            condition: row.condition,
            probabilityAi: row.probability_ai,
            threshold: row.threshold,
            decision: row.decision ? "ai" : "real",
          });
          byImage.set(canonicalId, values);
        }
      } catch {
        // A model can be listed while one condition shard is incomplete.
      }
    })));

    const { analytics, datasets } = await loadAnalytics(PROJECT_ROOT, modelDirs, CONDITION_ORDER, truthBySource);

    const images: GalleryImage[] = selected.map((sample) => ({
      id: sample.sample_id,
      baseId: sample.base_id ?? sample.source_sample_id ?? sample.sample_id,
      parentIds: sample.source_sample_id ? [sample.source_sample_id] : [],
      truth: sample.truth ?? (sample.label === 1 ? "ai" : "real"),
      imageUrl: `/api/images/${encodeURIComponent(sample.sample_id)}`,
      sourceFamily: sample.source_family ?? "Unknown source",
      generatorFamily: sample.generator_family || "Not applicable",
      dataset: manifest.dataset ?? "sid_set",
      license: sample.license ?? manifest.license ?? "Unknown",
      width: sample.width,
      height: sample.height,
      format: sample.file_format,
      sha256: sample.sha256,
      condition: "clean",
      conditionParameters: {},
      predictions: (byImage.get(sample.sample_id) ?? []).sort((a, b) => a.model.localeCompare(b.model)),
    }));

    return {
      images,
      models: modelDirs,
      conditions: CONDITION_ORDER.filter((condition) => conditions.has(condition)),
      totalImages: manifest.samples.length,
      source: "local-benchmark",
      analytics,
      datasets,
    };
  } catch {
    return { images: [], models: [], conditions: [], totalImages: 0, source: "empty", analytics: { metrics: [], evaluatedRows: 0, samplesPerSlice: 0, updatedAt: null }, datasets: [] };
  }
}

export async function resolveImage(id: string, condition = "clean"): Promise<string | null> {
  if (id.length !== 64 || /[^a-f0-9]/.test(id)) return null;
  if (!CONDITION_ORDER.includes(condition)) return null;
  const manifest = JSON.parse(await fs.readFile(MANIFEST, "utf8")) as { samples: ManifestSample[] };
  const sample = manifest.samples.find((item) => item.sample_id === id);
  if (!sample) return null;
  const root = path.resolve(PROJECT_ROOT, "work");
  if (condition !== "clean") {
    const derivative = path.resolve(root, "app-gallery", id, `${condition}.png`);
    return derivative.startsWith(`${root}${path.sep}`) ? derivative : null;
  }
  const candidate = path.resolve(root, sample.path.replace(/^data\//, "data/"));
  return candidate.startsWith(`${root}${path.sep}`) ? candidate : null;
}
