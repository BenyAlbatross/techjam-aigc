import "server-only";

import { promises as fs } from "node:fs";
import path from "node:path";
import type { GalleryImage, GalleryPayload, Prediction } from "@/lib/types";

const PROJECT_ROOT = path.resolve(process.cwd(), "..");
const MANIFEST = path.join(PROJECT_ROOT, "work/manifests/sid_set_1000x2_canonical.json");
const PREDICTIONS = path.join(PROJECT_ROOT, "work/predictions");
const LIMIT = 72;

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
    const canonicalBySource = new Map(
      selected.map((sample) => [sample.source_sample_id ?? sample.sample_id, sample.sample_id]),
    );
    const modelDirs = (await fs.readdir(PREDICTIONS, { withFileTypes: true }))
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .sort();
    const byImage = new Map<string, Prediction[]>();
    const conditions = new Set<string>();

    await Promise.all(modelDirs.map(async (model) => {
      const clean = path.join(PREDICTIONS, model, "sid_set", "clean.jsonl");
      try {
        for (const row of await readJsonLines(clean)) {
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
        // A model can be listed while its clean shard is incomplete.
      }
    }));

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
      conditions: [...conditions],
      totalImages: manifest.samples.length,
      source: "local-benchmark",
    };
  } catch {
    return { images: [], models: [], conditions: [], totalImages: 0, source: "empty" };
  }
}

export async function resolveImage(id: string): Promise<string | null> {
  if (!/^[a-f0-9]{64}$/.test(id)) return null;
  const manifest = JSON.parse(await fs.readFile(MANIFEST, "utf8")) as { samples: ManifestSample[] };
  const sample = manifest.samples.find((item) => item.sample_id === id);
  if (!sample) return null;
  const root = path.resolve(PROJECT_ROOT, "work");
  const candidate = path.resolve(root, sample.path.replace(/^data\//, "data/"));
  return candidate.startsWith(`${root}${path.sep}`) ? candidate : null;
}
