export type Truth = "real" | "ai" | "unknown";

export type Prediction = {
  model: string;
  condition: string;
  probabilityAi: number;
  threshold: number;
  decision: "real" | "ai";
  globalProbability?: number;
  memoryProbability?: number;
};

export type TransformationStep = { id: string; label: string; parameters: Record<string, unknown>; parentId?: string; imageUrl?: string };

export type GalleryImage = {
  id: string;
  baseId: string;
  parentIds: string[];
  truth: Truth;
  imageUrl: string;
  sourceFamily: string;
  generatorFamily: string;
  dataset: string;
  license: string;
  width?: number;
  height?: number;
  format?: string;
  sha256: string;
  condition: string;
  conditionParameters: Record<string, unknown>;
  transformChain: TransformationStep[];
  predictions: Prediction[];
};

export type GalleryPayload = {
  images: GalleryImage[];
  models: string[];
  conditions: string[];
  totalImages: number;
  source: "local-benchmark" | "empty";
  analytics: AnalyticsPayload;
  datasets: DatasetRecord[];
};

export type ConfusionCounts = { tp: number; tn: number; fp: number; fn: number; total: number };
export type ConditionMetric = { model: string; condition: string; confusion: ConfusionCounts; mismatchRate: number; falsePositiveRate: number; falseNegativeRate: number; balancedAccuracy: number; meanProbabilityAi: number | null; cleanMismatchCorrelation: number | null; mismatchLift: number | null; newlyWrongRate: number | null; recoveredRate: number | null };
export type AnalyticsPayload = { metrics: ConditionMetric[]; evaluatedRows: number; samplesPerSlice: number; updatedAt: string | null };
export type DatasetRecord = { id: string; name: string; repository: string; status: "approved" | "review" | "blocked"; roles: string[]; selected: boolean; license: string; reason: string | null };
