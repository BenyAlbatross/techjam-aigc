# Frontend application plan

Date: 31 August 2026

## Goal

Build a local-first application for inspecting image provenance, transformation
robustness, and detector output. The main surface is a visual gallery. Selecting
an image reveals its full modification chain and metadata. A fixed right panel
shows model scores and lets a user test additional images without adding them to
the controlled benchmark.

This interface is an evidence browser, not an image-authentication product. It
must preserve the project's fixed thresholds, benchmark provenance, and privacy
limits.

## Proposed stack

- Next.js App Router with TypeScript.
- Tailwind CSS and shadcn/ui for accessible interface primitives.
- React Flow only for lineage graphs; use ordinary CSS for the gallery.
- A small Python HTTP service around the existing model adapters for inference.
- Local JSON/JSONL and image files as the first data source. Do not add a database
  until annotations or multi-user persistence require one.

Keep the frontend in `app/` and the Python API in `scripts/app_server.py` (or a
small `server/` package if it grows). Keep large images, uploads, predictions,
and caches under ignored `work/` paths.

## Information architecture

### Main workspace

Use a three-part desktop layout:

1. A compact top bar for dataset, class, source/generator, transformation,
   correctness, and model filters.
2. A wide center canvas containing the photo gallery or selected lineage view.
3. A sticky, resizable right inspector containing model outputs and the test-new-
   image workflow.

On narrow screens, collapse the right inspector into a bottom sheet. Preserve
the image and its label as the strongest visual elements.

### Gallery

Default to a dense, responsive masonry-like grid with stable aspect ratios and
virtualization for large manifests. Every card should show:

- image thumbnail;
- `Real`, `AI-generated`, or `Unknown` truth badge;
- transformation badge such as `Clean`, `JPEG · q30`, or `Resize · 0.25×`;
- model verdict and probability when a model is selected;
- concise source/generator label;
- warning state for false positive, false negative, invalid, or missing data.

Click selects. Shift-click compares. Keyboard arrows move between cards. Filters
and selection should be reflected in the URL so a view can be shared locally.

### Image detail and lineage

The selected-image view starts with a large preview and a horizontal lineage
strip. Simple transformations use a left-to-right chain:

`source image -> operation -> derivative -> operation -> selected image`

Branching or multi-reference generation uses a compact directed graph. Image
nodes contain thumbnails and truth/source badges. Operation nodes contain the
transformation or generation method and parameters. Selecting a node updates the
metadata and model-output inspector without navigating away.

If an AI image has a known reference image, connect it explicitly. If provenance
is unknown, render an `Unknown source` root. Never infer or fabricate a reference
relationship from visual similarity.

### Metadata

Show human-readable fields first, then an expandable audit section.

Primary fields:

- truth/class and detector decision;
- source family and generator family;
- transformation name, ordered step, and parameters;
- dimensions, format, and file size;
- dataset, split, and licence/rights status.

Audit fields:

- sample ID, base ID, parent IDs, and content hash;
- dataset and model revisions;
- threshold, device, config hash, and run time when available;
- local relative path, never an absolute private path.

### Right model panel

For the selected image, show:

- one prominent AI probability meter with the fixed threshold and verdict;
- all available model outputs as sortable rows;
- clean-versus-transformed score delta and decision-flip indicator;
- model revision, status, and limitations;
- missing, stale, invalid, or still-running states.

Model scores should retain full precision in data and use rounded values only for
display. Avoid a red/green-only encoding; pair colour with text and icons.

### Test new images

Use a separate `Ad hoc test` section in the right panel:

1. Drop or choose one or more local images.
2. Validate type, decode, size, and batch limits.
3. Choose an approved cached model; default to `ateeqq_siglip`.
4. Run inference and stream per-image status.
5. Display score, verdict, threshold, latency, and errors.

Ad hoc uploads must be clearly separated from benchmark samples and marked
`Unknown truth / not benchmark evidence`. Store them in an isolated ignored
session directory, strip path disclosure from responses, and provide a visible
session-clear action. Do not silently add them to manifests, reports, training,
or calibration.

## Data contract

Add a UI-facing index generated from the authoritative manifests and prediction
shards. Do not make the browser parse all raw experiment files.

```ts
type ImageRecord = {
  id: string
  baseId: string
  parentIds: string[]
  relation: "source" | "transform" | "reference_generation"
  truth: "real" | "ai" | "unknown"
  imageUrl: string
  sourceFamily?: string
  generatorFamily?: string
  dataset: string
  split?: string
  license: string
  width?: number
  height?: number
  format?: string
  sha256: string
  operation?: { name: string; order: number; parameters: Record<string, unknown> }
}

type ModelOutput = {
  imageId: string
  model: string
  probabilityAi: number
  threshold: number
  decision: "real" | "ai"
  condition: string
  rawScore?: number
  revision?: string
  device?: string
  configHash?: string
}
```

The present `base_id` is enough to group clean and transformed benchmark rows,
but not enough to draw arbitrary chains. Extend the future manifest schema with
`parent_ids`, `relation`, and an ordered `operation` object. Keep `base_id` for
grouped statistical resampling.

## API boundary

Start with four read-oriented endpoints and one bounded inference endpoint:

- `GET /api/images` — paginated/filterable image summaries.
- `GET /api/images/:id` — full metadata and lineage neighborhood.
- `GET /api/images/:id/predictions` — model outputs across conditions.
- `GET /api/models` — approved, cached model availability and fixed thresholds.
- `POST /api/inference` — multipart ad hoc images plus one approved model key.

Serve image bytes through an ID-based endpoint after resolving and validating the
path against an allowlisted data root. Never accept an arbitrary filesystem path
from the browser. Restrict upload count and decoded pixel count, reject malformed
images, generate opaque session IDs, and serialize GPU inference to avoid memory
contention.

## Delivery phases

### Phase 0 — contracts and fixture

- Add the planned manifest schema from `TODO.md`, including explicit lineage.
- Add a deterministic UI-index builder and a small committed fixture containing
  a clean source, several transforms, and one reference-generated branch.
- Validate missing parents, lineage cycles, unsafe paths, conflicting hashes,
  and missing rights.

Exit gate: one command builds and validates a complete UI fixture offline.

### Phase 1 — read-only gallery shell

- Scaffold Next.js, Tailwind, shadcn/ui, linting, and frontend tests.
- Build the responsive gallery, filters, URL state, loading/empty/error states,
  and accessible card navigation.
- Serve fixture images and metadata through the bounded API.

Exit gate: a user can browse, filter, and inspect every fixture image without
running a model.

### Phase 2 — lineage and metadata inspector

- Add the linear chain first, then branching graph support.
- Add node selection, compare mode, metadata grouping, and unknown-provenance
  states.
- Test long chains, missing references, duplicate derivatives, and mobile layout.

Exit gate: source-to-result provenance remains understandable for linear and
branched examples and never invents an edge.

### Phase 3 — model-output panel

- Adapt benchmark JSONL into the UI index without changing thresholds.
- Add score meters, cross-model comparison, transform deltas, flip states, and
  run metadata.
- Preserve full numeric precision and distinguish unavailable from zero.

Exit gate: displayed values match source shards exactly in fixture tests.

### Phase 4 — ad hoc inference

- Wrap `scripts/model_adapters.py` behind the bounded upload endpoint.
- Add queue/progress/error states and session cleanup.
- Run compliance before enabling a model and require a locally cached checkpoint.
- Keep submission-format output separate from richer UI diagnostics.

Exit gate: an uploaded fixture produces the same probability as the existing CLI,
and uploaded data does not enter benchmark artifacts.

### Phase 5 — end-to-end verification and polish

- Add unit tests for adapters and lineage validation.
- Add browser tests for gallery filtering, lineage navigation, score display,
  upload, inference failure, and session clearing.
- Test keyboard use, contrast, reduced motion, large manifests, and GPU queueing.
- Add a short operator guide and architecture diagram.

Exit gate: the full browser -> API -> model -> result path is verified, and the
read-only gallery still works when no GPU or model cache is available.

## First implementation slice

Build a vertical slice before broad styling:

1. one fixture lineage with clean, JPEG, resize, and reference-generated nodes;
2. a 20-image gallery with class/transformation filters;
3. one selected-image detail view;
4. one right-panel model score from a fixture prediction;
5. one mocked ad hoc upload state, with real inference deferred to Phase 4.

This slice validates the core interaction and data contract early. Visual polish,
multi-model comparison, and large-data optimization follow after the lineage and
score semantics are correct.

## Decisions to preserve

- Read-only browsing must work without CUDA.
- Ad hoc tests are not benchmark, training, calibration, or authentication.
- Reference edges require explicit provenance data.
- Published thresholds and model outputs are displayed, never adjusted by the UI.
- Images, uploads, weights, and prediction shards stay out of Git.
- Only relative/opaque identifiers cross the API boundary.
