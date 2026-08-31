# Demo script: TRACE LENS

Target duration: 2–4 minutes. Use an approved local artifact set only. This
script demonstrates the current local prototype; it must not imply a public
service, a final competition-approved model, or image authentication.

## Before recording

- [ ] Verify the local gallery has the approved canonical manifest, images,
  predictions, and derivatives; do not show missing-artifact errors unless that
  is intentional.
- [ ] Select a recorded model/condition combination with visible gallery and
  analytics data.
- [ ] If showing upload inference, confirm the local CUDA environment and cached
  model work first. Use a non-sensitive, rights-cleared image and state that its
  truth is unknown and excluded from the benchmark.
- [ ] Hide terminals, browser tabs, paths, hostnames, usernames, credentials,
  API tokens, and all Tailnet/private-network details. Do not show ignored
  `work/` paths in screen recordings.
- [ ] Do not show restricted datasets, weights, private images, or raw manifests.
- [ ] Keep on-screen text and audio legible; record a local fallback copy.

## 0:00–0:20 — Problem and framing

**Screen:** Title card, then the TRACE LENS gallery landing page.

**Narration:** “AI-generated-image detectors can behave differently after the
same transformations images receive in real feeds: compression, blur, resizing,
noise, color changes, and cropping. TRACE LENS is a local evidence browser that
helps us inspect those changes instead of relying on one clean-image score.”

**Safety note:** Say “helps inspect” rather than “detects every fake” or
“authenticates images.”

## 0:20–0:40 — Solution and architecture

**Screen:** Gallery, then the browser’s model/condition controls and an image
detail panel.

**Narration:** “The benchmark uses pinned model and dataset registries. Local
prediction shards feed the gallery and analytics views. Each image can show its
source and rights metadata, transformation context, model probability, fixed
threshold, and prediction outcome. The browser runs locally; it is not a hosted
moderation system.”

**On-screen callouts:** `Pinned revisions`, `Fixed thresholds`, `Local evidence`.

## 0:40–1:15 — Gallery, metadata, and transformation chain

**Screen:** Select one gallery image, open its metadata/provenance area, then
change from Clean to a recorded transformation such as JPEG q30 or Resize 0.25x.

**Narration:** “Here, I can inspect an individual benchmark record. TRACE LENS
shows the recorded dataset metadata and the selected transformation condition.
Changing the condition updates the local derivative and lets us compare the
same image under a controlled, deterministic transformation.”

**Action:** Point out the truth label is benchmark ground truth, the score is
probability, and the threshold is fixed for that model.

**Safety note:** Do not read opaque IDs, file paths, or private source details
aloud. Do not claim the transformation chain proves real-world provenance.

## 1:15–1:45 — Model outputs and mismatch investigation

**Screen:** Choose a model and condition; filter the gallery to False Positive,
False Negative, or Mismatch. Open an example.

**Narration:** “Rather than hiding errors, the gallery can filter to mismatches.
This makes it possible to inspect whether a false positive or false negative is
linked to a particular transformation and compare model scores under the same
condition. These are benchmark outcomes, not proof that a new image is real or
AI-generated.”

## 1:45–2:15 — Analytics and measured evidence

**Screen:** Navigate to Analytics. Compare at least two conditions for one
available model.

**Narration:** “The analytics view recomputes confusion counts, false-positive
and false-negative rates, balanced accuracy, and mismatch behavior from the
local prediction shards. In the full 2,000-image-per-condition report, the
displayed top three are wkaandemir_clip, ateeqq_siglip, and
frontier_community_forensics—but the winner is explicitly unresolved because
the required confidence-interval comparisons are not all conclusive.”

**Optional callout:** “For the full-gate Ateeqq SigLIP result, 0.25x resizing
produced its worst real-source FPR of 0.525.”

**Safety note:** Do not call any model the winner or make claims outside the
pinned SID_Set conditions.

## 2:15–2:40 — Test a new image (optional)

**Screen:** Upload panel with a pre-approved non-sensitive PNG/JPEG/WebP,
then show the returned probability. If local CUDA inference is unavailable,
skip this segment rather than recording an error.

**Narration:** “For an exploratory local check, I can submit one supported image
to a cached detector. The result is an AI probability, not authentication. Its
truth is unknown, and this image is excluded from benchmark, training, and
calibration.”

**Action:** Clear the session after showing the result.

**Safety note:** Never upload personal, confidential, restricted, or copyrighted
material without permission. Do not show file names that reveal private details.

## 2:40–3:10 — Close with scope and next step

**Screen:** Return to the report/analytics summary or a simple closing slide.

**Narration:** “TRACE LENS makes robustness evidence inspectable: condition by
condition, model by model, with visible errors and provenance controls. It is a
technical baseline, not an image-authentication service. Our next step is to
complete human verification of the final model, data rights, public access, and
track-specific requirements before submission.”

## Recording acceptance checks

- [ ] Duration is between 2 and 4 minutes.
- [ ] The recorded UI matches the submitted code and uses approved local data.
- [ ] All reported figures match `outputs/public-baseline-robustness-report.md`.
- [ ] “Winner unresolved” is retained if the full-gate ranking is mentioned.
- [ ] No screenshots, audio, URLs, terminals, logs, or metadata reveal secrets,
  tokens, user names, private paths, Tailnet details, or restricted assets.
- [ ] The video is publicly accessible while logged out before its URL is added
  to the submission.
