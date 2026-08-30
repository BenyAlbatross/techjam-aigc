# Public Baseline Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a compliance-gated, Pixi-managed benchmark that measures ten existing AIGC detectors across authentic/generated public images and fifteen redistribution conditions.

**Architecture:** Small standard-library CLIs read immutable TOML registries, build local image manifests, fetch verified model snapshots, run one model at a time into atomic JSONL shards, and produce validated JSON/CSV/Markdown reports. Model inference reuses the existing adapters; transformed images stay in memory; raw data, weights, predictions, and caches remain outside Git.

**Tech Stack:** Python 3.12, Pixi rich platforms, PyTorch 2.13, torchvision 0.28, Transformers 5.16.1, timm 1.0.29, Hugging Face Hub/Datasets 5.0.1, Pillow 12.3, NumPy 2.5.2, pytest, TOML/JSONL.

**Spec:** `docs/superpowers/specs/2026-08-29-public-baseline-robustness-design.md`

## Global Constraints

- Use branch `xuan`; preserve unrelated work.
- Keep Caveman Ultra for agent communication; persisted project prose remains normal English.
- Use the smallest working design; reuse `work/baseline-spike/expanded.py` and `outputs/spark-a916-evaluation/run.py`.
- Do not fine-tune, fit thresholds, calibrate on evaluation data, or ensemble models.
- Count all learned inference-time parameters; reject counts at or above 2,000,000,000.
- Apply all transformations class-symmetrically with seed `20260829`.
- Required controlled grid has fifteen total conditions: clean plus fourteen transformed conditions.
- Required submission rows contain exactly `image_path` and `pred`; `pred` is finite and within `[0, 1]`.
- Never use competition-provided COCO/DALL-E data before complete method freeze.
- Never fetch or benchmark an asset unless its registry status is `approved`.
- Community Forensics Eval is blocked: its current card states `CC BY-NC-SA 4.0` and non-commercial use only.
- NTIRE 2026 train is review-blocked: its current card declares no license.
- SID_Set revision `dc03ead57929879319ce30a82bfcfb8d317b10bd` is the immediate controlled fallback under `CC BY 4.0`; label 2 is excluded.
- No images, weights, credentials, tokens, machine-local paths, caches, or prediction shards enter Git.
- Final submission remains blocked until current Track 5 brief is revalidated, repository is public, third-party inventory is complete, and free judging access is verified.
- Do not attempt identity recognition, face matching, EXIF-based identification, or lookup of depicted people.
- Do not execute real cleanup during development; `cleanup` previews targets unless operator passes `--confirm`.

---

## File Map

- `AGENTS.md`: binding project instructions for future agents.
- `pixi.toml`, `pixi.lock`: CPU/CUDA dependency lock and runnable tasks.
- `models.toml`: ten immutable model definitions and submission eligibility.
- `datasets.toml`: public dataset authorization and contamination records.
- `compliance.toml`: mutable hackathon release checklist.
- `scripts/compliance.py`: registry and release preflight.
- `scripts/data_manifest.py`: SID/NTIRE local manifest construction and validation.
- `scripts/fetch_models.py`: revision-pinned model download and SHA-256 verification.
- `scripts/model_adapters.py`: unified batch scoring for existing loaders.
- `scripts/benchmark.py`: transformations, resumable inference, atomic shards.
- `scripts/report.py`: metrics, grouped bootstrap, coverage, ranking, reports.
- `scripts/infer.py`: directory-to-submission JSON.
- `scripts/cleanup.py`: safe data inventory and confirmed deletion.
- `tests/fixtures/real.ppm`, `tests/fixtures/ai.ppm`: team-created text bitmap fixtures.
- `tests/test_compliance.py`, `tests/test_data_manifest.py`, `tests/test_models.py`, `tests/test_benchmark.py`, `tests/test_report.py`, `tests/test_infer_cleanup.py`: network-free CPU checks.
- `README.md`: Pixi workflow, compliance limits, reproduction commands.
- `outputs/public-baseline-robustness-report.md`: committed summary after the allowed gate.

### Task 1: Lock project instructions and Pixi environments

**Files:**
- Create: `AGENTS.md`
- Create: `pixi.toml`
- Create: `pixi.lock`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: current dependency pins in `requirements-baseline.txt`.
- Produces: Pixi tasks `test`, `compliance`, `fetch`, `benchmark`, `report`, `infer`, `cleanup`, and `cuda-check`.

- [ ] **Step 1: Add project instructions**

`AGENTS.md` must state:

```markdown
# Project instructions

- Communicate with the user in Caveman Ultra unless they disable it.
- Use Superpowers skills before applicable work and Ponytail minimalism for code.
- Work on branch `xuan`; preserve groupmates' and users' changes.
- Current priority: zero-fine-tuning public baselines, transformation robustness, and error rates.
- Do not fine-tune, calibrate, ensemble, or change published thresholds during this phase.
- Enforce fewer than 2 billion learned inference-time parameters.
- Run `pixi run compliance` before downloads or benchmarks. Block unknown, non-commercial, incompatible, or unauthorized assets.
- Never use competition-provided evaluation data for model or threshold selection.
- Keep images, weights, tokens, caches, and prediction shards out of Git.
- Use deterministic, class-symmetric transformations and report FPR, FNR, balanced accuracy, AUROC, confidence intervals, and contamination.
- Required submission output objects contain exactly `image_path` and `pred`.
- Run the smallest relevant network-free test after each code change.
```

- [ ] **Step 2: Add failing environment checks**

Run:

```bash
test -f pixi.toml
test -f AGENTS.md
```

Expected: first command fails because `pixi.toml` does not exist.

- [ ] **Step 3: Add Pixi manifest**

Use current official rich-platform syntax:

```toml
[workspace]
name = "techjam-aigc"
channels = ["https://prefix.dev/conda-forge"]
platforms = [
  { name = "linux-64-cuda", platform = "linux-64", cuda = "13.0" },
  { name = "linux-64-cpu", platform = "linux-64" },
]

[dependencies]
python = "3.12.*"

[pypi-dependencies]
ai-image-detector = { git = "https://github.com/lynote-ai/ai-image-detector.git", rev = "d3f4976d36c59974a25f55a0a7850b9866d3223b" }
datasets = "==5.0.1"
huggingface-hub = "==1.29.0"
numpy = "==2.5.2"
pillow = "==12.3.0"
pytest = ">=8,<10"
safetensors = "==0.8.0"
timm = "==1.0.29"
transformers = "==5.16.1"

[target.linux-64-cuda.pypi-dependencies]
torch = { version = "==2.13.0", index = "https://download.pytorch.org/whl/cu130" }
torchvision = { version = "==0.28.0", index = "https://download.pytorch.org/whl/cu130" }

[target.linux-64-cpu.pypi-dependencies]
torch = { version = "==2.13.0", index = "https://download.pytorch.org/whl/cpu" }
torchvision = { version = "==0.28.0", index = "https://download.pytorch.org/whl/cpu" }

[tasks]
test = "pytest -q"
compliance = "python scripts/compliance.py check --scope benchmark"
fetch = "python scripts/fetch_models.py"
benchmark = "python scripts/benchmark.py"
report = "python scripts/report.py"
infer = "python scripts/infer.py"
cleanup = "python scripts/cleanup.py preview"
cuda-check = "python -c \"import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))\""
```

- [ ] **Step 4: Ignore local environment and benchmark artifacts**

Append:

```gitignore
.pixi/
work/data/
work/manifests/
work/predictions/
work/reports/
work/cleanup-inventory.json
*.zip
*.pth
*.pt
```

- [ ] **Step 5: Install Pixi, lock, and verify both rich platforms**

Run:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
pixi lock
pixi run --platform linux-64-cpu python -c "import torch; assert not torch.cuda.is_available()"
pixi run --platform linux-64-cuda cuda-check
pixi run --platform linux-64-cpu test
```

Expected: lock succeeds; CPU assertion passes; GB10 name prints; pytest reports no tests collected with exit code 5 until Task 2 creates tests.

- [ ] **Step 6: Commit environment foundation**

```bash
git add AGENTS.md pixi.toml pixi.lock .gitignore
git commit -m "build: add Pixi benchmark environments"
```

### Task 2: Enforce model, dataset, and hackathon compliance

**Files:**
- Create: `models.toml`
- Create: `datasets.toml`
- Create: `compliance.toml`
- Create: `scripts/compliance.py`
- Create: `tests/test_compliance.py`

**Interfaces:**
- Produces: `load_registry(path: Path, section: str) -> dict[str, dict]`, `check_models(entries: dict, names: list[str], scope: str) -> list[str]`, `check_datasets(entries: dict, names: list[str]) -> list[str]`, and `check_release(path: Path) -> list[str]`.
- Consumes: no runtime services; uses Python `tomllib`.

- [ ] **Step 1: Write failing registry tests**

```python
from pathlib import Path
import pytest
from scripts.compliance import check_datasets, check_models, load_registry

def test_rejects_blocked_and_oversized_assets(tmp_path: Path):
    path = tmp_path / "models.toml"
    path.write_text(
        '[models.bad]\nstatus="blocked"\nsubmission_status="blocked"\n'
        'revision="abc"\nsha256="deadbeef"\nlicense="unknown"\n'
        'threshold=0.5\nparameters=2000000000\nloader="hf_multiclass"\n'
    )
    entries = load_registry(path, "models")
    errors = check_models(entries, ["bad"], "benchmark")
    assert any("status" in error for error in errors)
    assert any("2,000,000,000" in error for error in errors)

def test_approved_asset_passes():
    entry = {"ok": {
        "status": "approved", "submission_status": "review",
        "repository": "owner/repo", "file": "model.bin",
        "revision": "abc", "sha256": "a" * 64, "license": "MIT",
        "threshold": 0.5, "parameters": 1, "loader": "hf_multiclass",
    }}
    assert check_models(entry, ["ok"], "benchmark") == []

def test_dataset_requires_approval():
    entry = {"ntire": {
        "status": "review", "repository": "owner/data",
        "revision": "abc", "split": "shard_0", "license": "UNDECLARED",
    }}
    assert check_datasets(entry, ["ntire"]) == ["ntire: status is review"]
```

- [ ] **Step 2: Run tests and confirm import failure**

Run: `pixi run --platform linux-64-cpu pytest tests/test_compliance.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.compliance'`.

- [ ] **Step 3: Create exact model registry**

Add these ten records. Every record has `status = "approved"` for technical benchmarking and `submission_status = "review"` until the final candidate's training provenance is cleared. Add `contaminated_datasets = ["community_forensics_eval"]` to both Community Forensics models.

| Name | Repository | Revision | File | SHA-256 | Loader | Threshold | Parameters | License |
|---|---|---|---|---|---|---:|---:|---|
| `ateeqq_siglip` | `Ateeqq/ai-vs-human-image-detector` | `60e82406916921b823616bee33397baab38af3f0` | `model.safetensors` | `ab0b8cad7462a047ff4e2888cb4f11b1abe568d73ce07a2649a3f1541f73675f` | `hf_multiclass` | 0.5 | 92,885,762 | Apache-2.0 |
| `community_forensics` | `buildborderless/CommunityForensics-DeepfakeDet-ViT` | `ac6ee457bea904a373065754107451793b56db00` | `model.safetensors` | `275ba982236ddd6afddf7131f8133e89f537574b964cf8fa5825b4956d741692` | `hf_ai_logit` | 0.5 | 21,811,969 | MIT |
| `frontier_community_forensics` | `Thermostatic/community-forensics-frontier-detector-2026-08` | `16db135220b318d811b207db576d90368980b595` | `model.safetensors` | `2f445aecc310caee9b55d3ae0aeebc404a32d72e5836e30523ce0a9a1720cd47` | `timm_ai_logit` | 0.7956581000631541 | 21,811,969 | MIT |
| `wkaandemir_clip` | `wkaandemir/ai-image-detector` | `fefa013737a0c3477961d36ee8dbbdc751352366` | `model.safetensors` | `41ce93c6c206a4f3929e19cf9b43b663c63a47422ab27a9bbb67757db5f42339` | `timm_real_logit` | 0.08 | 85,846,273 | MIT |
| `divine_resnet50` | `divine2k/ai-image-detectors` | `5dd08026ea41f07ad7c37b79ffaed08282667655` | `resnet50_ai_real_final.pth` | `3f7ac353df62b85f10b75c9a96ebe443f7ad461437fdb1bde9477e842156e6f1` | `torchvision_real_logit` | 0.475 | 23,510,081 | MIT |
| `divine_efficientnet` | `divine2k/ai-image-detectors` | `5dd08026ea41f07ad7c37b79ffaed08282667655` | `efficientNet_BO_Final.pth` | `4f775d3d550550365f4481eb76f794dd850038ea375b64466435d115f37cca6c` | `torchvision_real_logit` | 0.475 | 4,008,829 | MIT |
| `divine_convnext` | `divine2k/ai-image-detectors` | `5dd08026ea41f07ad7c37b79ffaed08282667655` | `convNext_final.pth` | `ec5a7ae3b01eedb5da73ed94a9b6b9110f1c098fb7b50540d99a700efc758d05` | `torchvision_real_logit` | 0.475 | 27,820,897 | MIT |
| `steganograph` | `delpot/steganograph-ia-detector` | `b395557b96de82e5aaf97206054af416d657655a` | `model.safetensors` | `2f27de34eb5eebc250920e4ecbc471a7b8913d07c20b53f5d63fc0e41104a812` | `aidetector_hf` | 0.5 | 85,800,000 | MIT |
| `capcheck` | `capcheck/ai-image-detection` | `a6661e07d38f1a097bba07ca9415538819278f09` | `model.safetensors` | `4f5d0d4e8e6475faccca9385a2869648bb1f18d3f64c0239ad2bca909d3adc50` | `aidetector_hf` | 0.5 | 85,800,000 | Apache-2.0 |
| `univfd` | `siddharthksah/deepsafe-weights` | `71706520a9e12494b3ebc24e6091f5b22b9efcaf` | `universalfakedetect/fc_weights.pth` | `477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847` | `aidetector_univfd` | 0.5 | 428,000,000 | MIT |

For `univfd`, also store auxiliary file `universalfakedetect/ViT-L-14.pt` with SHA-256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`. Store each model's explicit AI/real label direction, architecture where required, temperature `0.595` for wkaandemir, and the Divine midpoint assumption.

- [ ] **Step 4: Create dataset and release registries**

`datasets.toml` contains:

```toml
[datasets.sid_set]
status = "approved"
repository = "saberzl/SID_Set"
revision = "dc03ead57929879319ce30a82bfcfb8d317b10bd"
split = "validation"
license = "CC-BY-4.0"
license_url = "https://huggingface.co/datasets/saberzl/SID_Set/blob/dc03ead57929879319ce30a82bfcfb8d317b10bd/README.md"
use = "controlled_gate"

[datasets.community_forensics_eval]
status = "blocked"
repository = "OwensLab/CommunityForensics-Eval"
revision = "7d4a74a88d2cac93b513c0853bf92c260eaceea0"
split = "CompEval"
license = "CC-BY-NC-SA-4.0"
reason = "Non-commercial restriction is not cleared for a prize competition."

[datasets.ntire_2026_train]
status = "review"
repository = "deepfakesMSU/NTIRE-RobustAIGenDetection-train"
revision = "700b6d08a3268b1e7a191306dec7321dd953b12f"
split = "shard_0"
license = "UNDECLARED"
reason = "Dataset card has no license field or usage grant."

[datasets.social_media_robustness]
status = "blocked"
repository = "danb21/social-media-robustness-sdxl-instantid"
revision = "unavailable"
split = "all"
license = "UNDECLARED"
reason = "User deferred this gated benchmark."
```

`compliance.toml` records official rules URL/check date, problem brief URL, `problem_brief_verified = false`, `repository_public = false`, `third_party_inventory_complete = false`, `english_materials = true`, `free_judging_access_verified = false`, `data_deletion_required = true`, and submission deadline `2026-09-01T12:00:00+08:00`.

- [ ] **Step 5: Implement strict checks**

```python
REQUIRED_MODEL = {
    "status", "submission_status", "repository", "revision", "file",
    "sha256", "license", "threshold", "parameters", "loader",
}

def check_models(entries, names, scope):
    errors = []
    for name in names:
        entry = entries.get(name)
        if entry is None:
            errors.append(f"{name}: absent from registry")
            continue
        missing = REQUIRED_MODEL - entry.keys()
        if missing:
            errors.append(f"{name}: missing {sorted(missing)}")
        if entry.get("status") != "approved":
            errors.append(f"{name}: status is {entry.get('status')}")
        if scope == "submission" and entry.get("submission_status") != "approved":
            errors.append(f"{name}: submission status is {entry.get('submission_status')}")
        if int(entry.get("parameters", 2_000_000_000)) >= 2_000_000_000:
            errors.append(f"{name}: parameter count must be below 2,000,000,000")
        digest = str(entry.get("sha256", ""))
        if len(digest) != 64:
            errors.append(f"{name}: SHA-256 must contain 64 hex characters")
    return errors

REQUIRED_DATASET = {"status", "repository", "revision", "split", "license"}

def check_datasets(entries, names):
    errors = []
    for name in names:
        entry = entries.get(name)
        if entry is None:
            errors.append(f"{name}: absent from registry")
            continue
        missing = REQUIRED_DATASET - entry.keys()
        if missing:
            errors.append(f"{name}: missing {sorted(missing)}")
        if entry.get("status") != "approved":
            errors.append(f"{name}: status is {entry.get('status')}")
    return errors
```

CLI exits 0 only with no errors. `--scope submission` also requires every release boolean to be true and prints exact false fields.

- [ ] **Step 6: Run compliance tests**

Run: `pixi run --platform linux-64-cpu pytest tests/test_compliance.py -q`

Expected: PASS.

- [ ] **Step 7: Verify intended gates**

```bash
pixi run --platform linux-64-cpu compliance -- --models all --dataset sid_set
pixi run --platform linux-64-cpu python scripts/compliance.py check --scope benchmark --dataset ntire_2026_train
pixi run --platform linux-64-cpu python scripts/compliance.py check --scope submission --models all --dataset sid_set
```

Expected: SID benchmark passes; NTIRE fails on `status is review`; submission fails on unresolved model and release fields.

- [ ] **Step 8: Commit compliance gate**

```bash
git add models.toml datasets.toml compliance.toml scripts/compliance.py tests/test_compliance.py
git commit -m "feat: enforce benchmark compliance registry"
```

### Task 3: Build deterministic local manifests

**Files:**
- Create: `scripts/data_manifest.py`
- Create: `tests/fixtures/real.ppm`
- Create: `tests/fixtures/ai.ppm`
- Create: `tests/test_data_manifest.py`

**Interfaces:**
- Produces: `validate_manifest(payload: dict, root: Path) -> list[str]`, `build_sid(output: Path, per_class: int) -> Path`, and `build_ntire(shard_dir: Path, output: Path) -> Path`.
- Manifest row keys: `sample_id`, `base_id`, `label`, `truth`, `path`, `sha256`, `source_family`, `generator_family`, `license`.

- [ ] **Step 1: Add team-created PPM fixtures**

`real.ppm`:

```text
P3
2 2
255
10 20 30 20 30 40
30 40 50 40 50 60
```

`ai.ppm`:

```text
P3
2 2
255
240 220 200 230 210 190
220 200 180 210 190 170
```

- [ ] **Step 2: Write failing manifest validation test**

```python
def test_fixture_manifest_validates(tmp_path):
    samples = []
    for label, name in enumerate(("real.ppm", "ai.ppm")):
        shutil.copy(Path("tests/fixtures") / name, tmp_path / name)
        digest = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        samples.append({
            "sample_id": name, "base_id": name, "label": label,
            "truth": "ai" if label else "real", "path": name,
            "sha256": digest, "source_family": "fixture",
            "generator_family": "fixture", "license": "CC0-1.0",
        })
    payload = {"dataset": "fixture", "revision": "1", "samples": samples}
    assert validate_manifest(payload, tmp_path) == []
    payload["samples"][1]["sha256"] = "0" * 64
    assert "hash mismatch" in " ".join(validate_manifest(payload, tmp_path))
```

Also assert duplicate `sample_id`, label outside `{0, 1}`, absolute path, absent rights, absent file, and cross-label duplicate hash each produce an error.

- [ ] **Step 3: Run test and confirm missing module**

Run: `pixi run --platform linux-64-cpu pytest tests/test_data_manifest.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement validation**

```python
def validate_manifest(payload: dict, root: Path) -> list[str]:
    errors, ids, hashes = [], set(), {}
    for row in payload.get("samples", []):
        sample_id = str(row.get("sample_id", ""))
        relative = Path(str(row.get("path", "")))
        if not sample_id or sample_id in ids:
            errors.append(f"{sample_id}: duplicate or empty sample_id")
        ids.add(sample_id)
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"{sample_id}: path must stay relative")
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f"{sample_id}: absent file")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row.get("sha256"):
            errors.append(f"{sample_id}: hash mismatch")
        label = row.get("label")
        if label not in (0, 1):
            errors.append(f"{sample_id}: invalid label")
        if not row.get("license"):
            errors.append(f"{sample_id}: missing rights")
        if digest in hashes and hashes[digest] != label:
            errors.append(f"{sample_id}: same bytes have conflicting labels")
        hashes[digest] = label
    return errors
```

- [ ] **Step 5: Implement SID builder**

Call compliance first. Stream `validation` at the pinned revision with image decoding disabled. Skip label 2. Save first `per_class` rows of labels 0 and 1 under `work/data/sid_set/images/`, using opaque SHA-based filenames. Write relative paths, hashes, source row, and class metadata to `work/manifests/sid_set_1000x2.json`; write through a temporary file then `os.replace`.

```python
dataset = load_dataset(
    "saberzl/SID_Set", split="validation", streaming=True,
    revision="dc03ead57929879319ce30a82bfcfb8d317b10bd",
).cast_column("image", HFImage(decode=False))
```

Set real `source_family = "OpenImages_V7"`; set generated `generator_family = "SID_Set_undisclosed_mixture"`; retain the limitation in manifest metadata.

- [ ] **Step 6: Implement local NTIRE builder without downloader**

Read `labels.csv` with `csv.DictReader`; resolve every image under `images/`; sort by image name; store relative paths and hashes. CLI checks `ntire_2026_train` approval before this function, so current invocation fails before reading files.

- [ ] **Step 7: Run tests and manifest smoke**

```bash
pixi run --platform linux-64-cpu pytest tests/test_data_manifest.py -q
```

Expected: tests pass.

- [ ] **Step 8: Commit manifest builder**

```bash
git add scripts/data_manifest.py tests/fixtures tests/test_data_manifest.py
git commit -m "feat: add deterministic data manifests"
```

### Task 4: Fetch verified checkpoints and expose unified adapters

**Files:**
- Create: `scripts/fetch_models.py`
- Create: `scripts/model_adapters.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Produces: `fetch_model(name: str, cache: Path) -> Path`, `sha256_file(path: Path) -> str`, `load_model(name: str, device: str, cache: Path) -> ModelAdapter`, and `ModelAdapter.score_batch(images: list[Image.Image]) -> list[tuple[float, float]]`.
- `ModelAdapter` properties: `name`, `threshold`, `parameter_count`, `revision`, `weight_sha256`.

- [ ] **Step 1: Write failing hash and parameter tests**

```python
def test_sha256_file(tmp_path):
    path = tmp_path / "weight.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

def test_parameter_limit_is_strict():
    assert_parameter_limit(1_999_999_999)
    with pytest.raises(RuntimeError, match="2B"):
        assert_parameter_limit(2_000_000_000)
```

- [ ] **Step 2: Run tests and confirm missing module**

Run: `pixi run --platform linux-64-cpu pytest tests/test_models.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement revision-pinned fetch**

Use `snapshot_download(repo_id=..., revision=..., cache_dir=..., allow_patterns=...)`. Include model file, `config.json`, and `preprocessor_config.json` where used. Verify every primary and auxiliary SHA-256 after download; delete only the mismatched file; raise without retrying a different revision.

```python
def fetch_model(name: str, cache: Path) -> Path:
    entry = load_registry(Path("models.toml"), "models")[name]
    errors = check_models({name: entry}, [name], "benchmark")
    if errors:
        raise RuntimeError("\n".join(errors))
    snapshot = Path(snapshot_download(
        repo_id=entry["repository"],
        revision=entry["revision"],
        cache_dir=cache,
        allow_patterns=[
            entry["file"], *entry.get("auxiliary_files", []),
            "config.json", "preprocessor_config.json",
        ],
    ))
    weight = snapshot / entry["file"]
    if sha256_file(weight) != entry["sha256"]:
        weight.unlink()
        raise RuntimeError(f"{name}: checkpoint hash mismatch")
    return snapshot
```

- [ ] **Step 4: Promote existing loader logic**

Move the loader-kind branches from `CandidateDetector` in `work/baseline-spike/expanded.py` into `ModelAdapter`. Preserve:

- Hugging Face label-token detection for Ateeqq;
- single AI logit for Community Forensics;
- timm ViT-S/16 384 AI logit for Frontier;
- timm CLIP real-logit inversion and temperature 0.595 for wkaandemir;
- torchvision architectures and real-logit inversion for Divine;
- `aidetector` HF backends for Steganograph and CapCheck;
- `aidetector` UnivFD backend with both verified files cached.

All Hugging Face loads pass exact `revision`, cache directory, and `local_files_only=True`. Benchmark execution sets `HF_HUB_OFFLINE=1`.

```python
def assert_parameter_limit(count: int) -> None:
    if count >= 2_000_000_000:
        raise RuntimeError(f"Model exceeds 2B parameter limit: {count}")

def score_batch(self, images):
    if self.backend_supports_batch:
        values = self.preprocess(images).to(self.device)
        with torch.inference_mode():
            logits = self.model(values)
        return self.to_ai_scores(logits)
    return [self.score_one(image) for image in images]
```

After construction, compare actual parameter count with registry count; fail on mismatch rather than silently updating the registry.

- [ ] **Step 5: Run model unit tests**

Use a fake one-parameter `torch.nn.Module` and synthetic logits to test AI direction, real-logit inversion, threshold retention, and count mismatch without downloading weights.

Run: `pixi run --platform linux-64-cpu pytest tests/test_models.py -q`

Expected: PASS.

- [ ] **Step 6: Fetch and smoke-test one cached model**

```bash
pixi run fetch -- --model ateeqq_siglip --cache work/hf-cache
HF_HUB_OFFLINE=1 pixi run --platform linux-64-cpu python scripts/model_adapters.py smoke --model ateeqq_siglip --image tests/fixtures/ai.ppm
```

Expected: verified checkpoint hash prints; one finite AI probability prints; parameter count is 92,885,762.

- [ ] **Step 7: Commit model layer**

```bash
git add scripts/fetch_models.py scripts/model_adapters.py tests/test_models.py
git commit -m "feat: add verified model adapters"
```

### Task 5: Run deterministic, resumable transformation inference

**Files:**
- Create: `scripts/benchmark.py`
- Create: `tests/test_benchmark.py`

**Interfaces:**
- Produces: `apply_condition(image: Image.Image, condition: str, sample_id: str) -> Image.Image`, `score_with_backoff(adapter, images, batch_size) -> tuple[list[tuple[float, float]], int]`, `write_shard(path: Path, rows: list[dict]) -> None`, `validate_shard(path: Path, expected: set[str]) -> list[str]`, and `run_panel(...) -> list[Path]`.
- Consumes: manifest rows from Task 3 and `ModelAdapter` from Task 4.

- [ ] **Step 1: Write failing transform tests**

```python
def test_conditions_are_complete_and_deterministic():
    assert len(CONDITIONS) == 15
    image = Image.new("RGB", (100, 80), (10, 20, 30))
    assert apply_condition(image, "resize_0.25", "x").size == image.size
    assert apply_condition(image, "center_crop_80", "x").size == (80, 64)
    first = np.asarray(apply_condition(image, "noise_sigma0.05", "x"))
    second = np.asarray(apply_condition(image, "noise_sigma0.05", "x"))
    assert np.array_equal(first, second)
```

Add tests proving all conditions receive identical parameters for real and AI rows sharing the same `sample_id`, unknown conditions fail, atomic shards leave no temporary file, complete shards resume without rescoring, and conflicting duplicates fail.

- [ ] **Step 2: Run test and confirm missing module**

Run: `pixi run --platform linux-64-cpu pytest tests/test_benchmark.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement canonical fifteen-condition transform**

Promote `CONDITIONS`, `rng_for`, and `transformed` from `outputs/spark-a916-evaluation/run.py`. Keep exact condition names and parameters. Seed RNG with SHA-256 of `"20260829:{sample_id}"`. Always EXIF-transpose and convert to RGB before transformation.

- [ ] **Step 4: Implement OOM backoff**

```python
def score_with_backoff(adapter, images, batch_size):
    size = min(batch_size, len(images))
    while size >= 1:
        try:
            scored = []
            for start in range(0, len(images), size):
                scored.extend(adapter.score_batch(images[start:start + size]))
            return scored, size
        except torch.cuda.OutOfMemoryError:
            if size == 1:
                raise
            torch.cuda.empty_cache()
            size = max(1, size // 2)
    raise AssertionError("unreachable")
```

No rows are persisted until the full condition succeeds.

- [ ] **Step 5: Implement atomic condition shards**

One file represents one model/dataset/condition. Each row includes model/revision/hash, dataset/revision, sample/base ID, content hash, label/cohort, condition, exact condition parameters, raw score, AI probability, threshold, decision, device, effective batch size, elapsed seconds, Git commit, config hash, and seed.

```python
def write_shard(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
```

Identity is SHA-256 of model revision/hash, dataset revision, image hash, condition, Git commit, and registry/config hash. Resume skips only a shard whose exact identity set and row count validate.

- [ ] **Step 6: Implement CLI and invalid-image policy**

```bash
python scripts/benchmark.py \
  --models ateeqq_siglip,community_forensics \
  --dataset sid_set \
  --manifest work/manifests/sid_set_1000x2.json \
  --conditions all \
  --device cuda:0 \
  --batch-size 32 \
  --output work/predictions
```

Preflight calls compliance and manifest validation before loading a model. Log corrupt IDs; fail when invalid count exceeds `floor(0.001 * selected_count)`; record attempted, valid, excluded, and per-class counts.

- [ ] **Step 7: Run benchmark tests**

Use a deterministic dummy adapter returning mean-red-channel probability. Run two fixture images through clean and JPEG, rerun, assert adapter call count does not increase, then corrupt one shard and assert resume rejects it.

Run: `pixi run --platform linux-64-cpu pytest tests/test_benchmark.py -q`

Expected: PASS.

- [ ] **Step 8: Commit benchmark runner**

```bash
git add scripts/benchmark.py tests/test_benchmark.py
git commit -m "feat: add resumable robustness benchmark"
```

### Task 6: Report errors, confidence intervals, contamination, and ranking

**Files:**
- Create: `scripts/report.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Produces: `metrics(rows: list[dict]) -> dict`, `grouped_bootstrap(rows, statistic, seed=20260829, replicates=2000) -> tuple[float, float]`, `validate_coverage(...) -> list[str]`, `rank_models(summary, contamination) -> list[str]`, and `render_report(...) -> str`.
- Consumes: atomic JSONL shards and model/dataset registries.

- [ ] **Step 1: Write failing exact-metric tests**

```python
def test_metrics_and_grouped_bootstrap():
    rows = [
        row("r1", 0, 0.1), row("r2", 0, 0.9),
        row("a1", 1, 0.8), row("a2", 1, 0.2),
    ]
    result = metrics(rows, threshold=0.5)
    assert result["error_rate"] == 0.5
    assert result["fpr"] == 0.5
    assert result["fnr"] == 0.5
    assert result["balanced_accuracy"] == 0.5
    assert result["roc_auc"] == 0.5
    statistic = lambda sampled: metrics(sampled)["balanced_accuracy"]
    assert grouped_bootstrap(rows, statistic) == grouped_bootstrap(rows, statistic)
```

Add tests for all-correct AUROC 1, reversed AUROC 0, paired clean/transformed flip rate, duplicate identity, missing expected row, base-ID resampling, contaminated-pair exclusion, and tie-break order.

- [ ] **Step 2: Run test and confirm missing module**

Run: `pixi run --platform linux-64-cpu pytest tests/test_report.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement metrics without a new statistics dependency**

```python
def metrics(rows, threshold=None):
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    scores = np.asarray([row["probability_ai"] for row in rows], dtype=float)
    cutoff = float(rows[0]["threshold"] if threshold is None else threshold)
    predicted = scores >= cutoff
    tp = int(((labels == 1) & predicted).sum())
    tn = int(((labels == 0) & ~predicted).sum())
    fp = int(((labels == 0) & predicted).sum())
    fn = int(((labels == 1) & ~predicted).sum())
    fpr, fnr = fp / (fp + tn), fn / (fn + tp)
    return {
        "n": len(rows), "error_rate": (fp + fn) / len(rows),
        "fpr": fpr, "fnr": fnr,
        "balanced_accuracy": ((1 - fpr) + (1 - fnr)) / 2,
        "roc_auc": rank_auc(labels, scores),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
```

`rank_auc` uses stable average ranks for ties, matching the frozen Spark evaluator.

- [ ] **Step 4: Implement paired/grouped bootstrap and transform deltas**

Group controlled rows by `base_id`; sample group IDs with replacement 2,000 times. NTIRE uses `sample_id` when no lineage exists. Return 2.5th and 97.5th percentiles. Join transformed rows to clean rows by model and sample identity to compute mean score delta and binary decision-flip rate.

- [ ] **Step 5: Enforce coverage and preregistered ranking**

Expected controlled coverage is selected sample count times fifteen per completed model. Wild coverage is selected sample count times clean only. Fail report generation on conflicting duplicate identities or missing rows.

Rank eligible models by:

1. descending worst-condition balanced accuracy;
2. ascending worst real-source FPR;
3. ascending worst AI-source FNR;
4. descending aggregate AUROC;
5. model name for deterministic final ordering.

Exclude model-dataset pairs whose `contaminated_datasets` contains the dataset key. Report them with `ranking_eligible = false`.

- [ ] **Step 6: Render all output formats**

```bash
python scripts/report.py \
  --predictions work/predictions \
  --manifest work/manifests/sid_set_1000x2.json \
  --json work/reports/summary.json \
  --csv work/reports/metrics.csv \
  --markdown outputs/public-baseline-robustness-report.md
```

Markdown contains per-condition metrics/CIs, worst cohorts, score deltas, flip rates, throughput, invalid counts, contamination/exclusion reasons, top-three rule, and scope limitations. It uses opaque IDs only.

- [ ] **Step 7: Run report tests**

Run: `pixi run --platform linux-64-cpu pytest tests/test_report.py -q`

Expected: PASS.

- [ ] **Step 8: Commit reporting**

```bash
git add scripts/report.py tests/test_report.py
git commit -m "feat: report robustness errors and ranking"
```

### Task 7: Add exact inference output and safe competition cleanup

**Files:**
- Create: `scripts/infer.py`
- Create: `scripts/cleanup.py`
- Create: `tests/test_infer_cleanup.py`

**Interfaces:**
- Produces: `infer_directory(adapter, input_dir: Path) -> tuple[list[dict], list[str]]`, `cleanup_targets(root: Path) -> list[Path]`, and `delete_targets(root: Path, targets: list[Path]) -> None`.
- Consumes: `load_model` from Task 4.

- [ ] **Step 1: Write failing output-schema and cleanup tests**

```python
def test_submission_rows_have_exact_schema(tmp_path):
    shutil.copy("tests/fixtures/real.ppm", tmp_path / "image.ppm")
    rows, invalid = infer_directory(DummyAdapter(), tmp_path)
    assert invalid == []
    assert list(rows[0]) == ["image_path", "pred"]
    assert rows[0]["image_path"] == "image.ppm"
    assert 0.0 <= rows[0]["pred"] <= 1.0

def test_cleanup_preview_and_confined_delete(tmp_path):
    target = tmp_path / "work" / "data"
    target.mkdir(parents=True)
    (target / "x").write_text("x")
    planned = cleanup_targets(tmp_path)
    assert target in planned and target.exists()
    delete_targets(tmp_path, planned)
    assert not target.exists()
    with pytest.raises(ValueError):
        delete_targets(tmp_path, [tmp_path.parent])
```

- [ ] **Step 2: Run test and confirm missing modules**

Run: `pixi run --platform linux-64-cpu pytest tests/test_infer_cleanup.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement directory inference**

Sort regular files by relative POSIX path. Decode with EXIF transpose/RGB. Score in batches. Reject non-finite or out-of-range scores. Log corrupt inputs to stderr and return their opaque relative paths separately. JSON output is an array; each row is constructed exactly as:

```python
{"image_path": relative_path.as_posix(), "pred": float(probability_ai)}
```

CLI requires an approved model, verified local weights, input directory, and output path. It records extended diagnostics in a separate optional file, never in required rows.

- [ ] **Step 4: Implement confined cleanup**

Only these repository-relative roots are legal targets:

```python
SAFE_TARGETS = (
    Path("work/data"),
    Path("work/manifests"),
    Path("work/hf-cache"),
    Path("work/model-cache"),
    Path("work/predictions"),
    Path("work/reports"),
)
```

`preview` writes `work/cleanup-inventory.json` with path, byte count, file count, and timestamp; it deletes nothing. `delete --confirm` resolves every target, verifies it is inside repository and exactly equals a safe target, deletes it, and writes a deletion attestation outside deleted roots at `outputs/data-deletion-attestation.json`.

- [ ] **Step 5: Run tests**

Run: `pixi run --platform linux-64-cpu pytest tests/test_infer_cleanup.py -q`

Expected: PASS; only pytest temporary directories are removed.

- [ ] **Step 6: Run fixture inference**

```bash
HF_HUB_OFFLINE=1 pixi run --platform linux-64-cpu infer -- \
  --model ateeqq_siglip \
  --input tests/fixtures \
  --output work/reports/fixture-predictions.json
python -c "import json; rows=json.load(open('work/reports/fixture-predictions.json')); assert all(set(x)=={'image_path','pred'} for x in rows)"
pixi run cleanup
```

Expected: exact schema assertion passes; cleanup prints inventory only.

- [ ] **Step 7: Commit inference and cleanup**

```bash
git add scripts/infer.py scripts/cleanup.py tests/test_infer_cleanup.py
git commit -m "feat: add submission inference and safe cleanup"
```

### Task 8: Document, verify, and run the allowed public gate

**Files:**
- Modify: `README.md`
- Create: `outputs/public-baseline-robustness-report.md`
- Modify: `compliance.toml`

**Interfaces:**
- Consumes: all preceding tasks.
- Produces: reproducible allowed-gate report and unresolved compliance list.

- [ ] **Step 1: Run complete network-free verification**

```bash
pixi run --platform linux-64-cpu test
pixi run --platform linux-64-cpu python work/baseline-spike/verify_outputs.py
pixi run --platform linux-64-cpu python outputs/spark-a916-evaluation/run.py self-test
pixi run --platform linux-64-cpu python -m compileall -q scripts tests outputs work/baseline-spike
git diff --check
```

Expected: all pytest tests pass; existing checks print success; compile and diff checks are silent.

- [ ] **Step 2: Document exact Pixi workflow**

README must include:

- CPU and CUDA install/run commands;
- model cache location and one-model-at-a-time policy;
- SID_Set controlled-gate selection;
- fifteen conditions;
- fixed thresholds and no fine-tuning;
- metrics and ranking rule;
- current Community Forensics and NTIRE blockers;
- model/data revisions and license inventory links;
- directory inference command;
- cleanup preview and end-of-competition confirmed cleanup;
- claim limits and prohibition on identity inference;
- final-release blockers from `pixi run compliance -- --scope submission`.

- [ ] **Step 3: Build legal 2,000-image SID manifest**

```bash
pixi run --platform linux-64-cpu python scripts/data_manifest.py build-sid \
  --per-class 1000 \
  --output work/manifests/sid_set_1000x2.json
pixi run --platform linux-64-cpu python scripts/data_manifest.py validate \
  work/manifests/sid_set_1000x2.json
```

Expected: exactly 2,000 valid rows, 1,000 per class, zero duplicate IDs, zero hash errors.

- [ ] **Step 4: Fetch all ten approved technical baselines**

```bash
pixi run fetch -- --model all --cache work/hf-cache
```

Expected: every primary/auxiliary file matches its registry SHA-256; no unpinned revision downloads.

- [ ] **Step 5: Run CUDA gate one model at a time**

Run exact registry list sequentially:

```bash
for model in \
  ateeqq_siglip community_forensics frontier_community_forensics \
  wkaandemir_clip divine_resnet50 divine_efficientnet divine_convnext \
  steganograph capcheck univfd
do
  HF_HUB_OFFLINE=1 pixi run benchmark -- \
    --models "$model" \
    --dataset sid_set \
    --manifest work/manifests/sid_set_1000x2.json \
    --conditions all \
    --device cuda:0 \
    --batch-size 32 \
    --output work/predictions
done
```

Expected per model: 30,000 prediction rows across fifteen complete atomic shards. OOM backoff may lower effective batch size but must not reduce coverage.

- [ ] **Step 6: Generate and validate report**

```bash
pixi run report -- \
  --predictions work/predictions \
  --manifest work/manifests/sid_set_1000x2.json \
  --json work/reports/summary.json \
  --csv work/reports/metrics.csv \
  --markdown outputs/public-baseline-robustness-report.md
pixi run --platform linux-64-cpu python scripts/report.py validate \
  --predictions work/predictions \
  --manifest work/manifests/sid_set_1000x2.json \
  --models all
```

Expected: 300,000 unique predictions, no missing/conflicting identities, ten model summaries, fifteen condition summaries per model, fixed-threshold metrics/CIs, and top three selected by the preregistered ranking.

- [ ] **Step 7: Record compliance state without bypassing blockers**

Set `third_party_inventory_complete = true` after README inventory review. Keep `problem_brief_verified`, `repository_public`, and `free_judging_access_verified` false until humans verify them. Do not change Community Forensics or NTIRE status without written permission.

Run:

```bash
pixi run --platform linux-64-cpu python scripts/compliance.py check \
  --scope submission --models all --dataset sid_set
```

Expected: nonzero exit listing remaining human release blockers.

- [ ] **Step 8: Review generated report for prohibited claims**

Search:

```bash
rg -n -i 'proves authentic|detects all|universal|generator-unseen' README.md outputs/public-baseline-robustness-report.md
```

Expected: no matches except explicit limitation statements prefixed by `does not` or `not`.

- [ ] **Step 9: Commit reproducible code and summary only**

```bash
git status --short
git add README.md compliance.toml outputs/public-baseline-robustness-report.md
git commit -m "docs: report public robustness baseline"
git status --short --branch
```

Expected: ignored data, weights, predictions, caches, and local reports do not appear; branch is clean and ahead of `origin/xuan`.

## Authorized Large-Panel Continuation Gate

Do not run Community Forensics Eval or NTIRE commands under current registry state. If organizers provide written permission:

1. Commit permission source, scope, date, and authorized uses to `datasets.toml`.
2. Change only the covered dataset status to `approved`.
3. Run compliance and preserve its passing output.
4. Build the pinned full manifest.
5. Run the 2,000-image as-distributed NTIRE gate where authorized.
6. Run the top three eligible models on the complete authorized panel.
7. Generate a new versioned report; never rewrite the SID report or reuse its claim.

No permission means no download, transformation, benchmark, demo, or submission use.
