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
  predictions: Prediction[];
};

export type GalleryPayload = {
  images: GalleryImage[];
  models: string[];
  conditions: string[];
  totalImages: number;
  source: "local-benchmark" | "empty";
};
