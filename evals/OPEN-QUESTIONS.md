# Open questions and recommended next steps

Written for whoever takes this over. Ranked by expected value.

---

## 1. Ship the memory branch, or re-fit the fusion gate  — *cheap, concrete, high value*

TRACE-RX-Parallel's fused output tracks its global branch to three decimals on every dataset tested,
i.e. the learned gate effectively ignores the memory branch. In-distribution that is correct (global
is better there). Out-of-distribution it costs **0.12 AUROC**:

| dataset | fused (shipped) | memory alone |
|---|---|---|
| WildFake chains | 0.4730 | **0.5960** |
| eval-subset `default` | 0.4260 | **0.5686** |
| eval-subset `diverse` | 0.5223 | **0.6471** |

The memory branch alone also beats **TRACE-RX-M's** 0.5721 on WildFake. The best detector measured in
this whole exercise is a component the shipped model discards.

**Next step:** re-fit the two fusion weights on any out-of-distribution sample, or ship memory alone
as a fallback path. This does not require retraining anything — the branch already exists and its
logits are already exposed as `memory_logit`.

## 2. Generator coverage, not augmentation  — *the actual bottleneck*

Six stacked transforms cost <= 0.005 AUROC. One unseen generator costs 0.66. Roughly 130x.

Also note their own docs: the pipeline **warns when fewer than eight generator families are
available**. Training used four. The warning was correct.

**Next step:** more generator families in training. My earlier holdout work on `data_draft` pointed
the same way from the other direction — removing `gan_based` cost more external AUROC (−0.041) than
removing any other generator, while removing `adm` cost nothing (+0.005). Weight acquisition toward
**GAN-family and architecturally distinct generators**, not more diffusion variants.

## 3. Why is the failure *below* chance rather than at chance?  — *unresolved, and diagnostic*

On the held-out generator both models score 0.24–0.34, not ~0.50. On `data_draft`, Gemini-equivalent
AI images score a median logit of **−7.29** against real photos at **−5.55** — the models are *more
confident these AI images are authentic than they are about actual photographs*.

That is not the signature of a model that has run out of signal. It is the signature of a model
reading a present cue backwards. Given TRACE-RX's architecture, the natural hypothesis is that these
images **reconstruct unusually well from the authentic-prototype memory** — they look more like the
reference photos than photos do.

**Next step:** dump `residual`, `s_max` and `s_ent` (all already returned by both models' forward
pass) for held-out-generator images versus real images and compare distributions. If reconstruction
error is genuinely lower for those AI images, that localises the failure to the memory and suggests
the prototype set is the thing to fix.

**And the transform evidence supports this hypothesis:** `adm` and `gan_based` get *better* with more
transforms (+0.096, +0.054). Degrading the image is destroying the misleading cue.

## 4. Does transform ORDER matter?  — *data collected, analysis not run*

The chain runs record the full ordered chain per row (`chain` column in
`acai-project/runs/chains_*.parquet`), and all 120 orderings of 3-family chains occur. Nobody has
yet tested whether e.g. `jpeg -> blur` differs from `blur -> jpeg`.

**Next step:** group `acai-project/runs/chains_*.parquet` by the `chain` string and compare AUROC across permutations
of the same family set. No re-running required — the predictions are already there.

## 5. Unfinished from my own earlier baseline work

- **H2, the dense forensic-token fusion arm**, was never run. It injects per-patch feature maps
  inside the backbone so attention can localise evidence, which a global scalar cannot express. It
  needs a full finetune (cannot use cached embeddings). It is the only fusion hypothesis untested.
- **Feature calibration gate is OPEN.** Of the three low-level forensic features, only
  `wavelet_hf_kurtosis` reproduces the prior study's numbers (0.621, CI [0.584, 0.657] contains their
  0.597). `residual_kurtosis` (0.532) and `phase_neighbor_coherence` (0.521) do **not** — both sit at
  chance where theirs were 0.462 and 0.589. So my finding that "these features add nothing to DINOv3"
  is conditional: it shows *my* features add nothing, not that the originals do. Resolving this needs
  the original implementation.

## 6. Housekeeping that will bite

- `acai/transforms.py` was replaced mid-project with the team's official-policy version. The older
  modules (`dataset.py`, `train.py`, `compose.py`, `overnight.py`, `build_transformed.py`) still
  import the pre-rewrite API (`Chain`, `apply_chain`, `conditions`) and **will not import**. The six
  scripts in `scripts/` all use the current API and run fine.
- The two model branches cannot share a `sys.path`. `feat/trace-rx-parallel` dropped
  `optimizer.mixed_precision`, which trace-rx-m's shipped config still contains. **One process per
  model.**

## 7. Things I tested that are dead ends — do not repeat

| approach | result |
|---|---|
| preprocessing geometry (crop/resize/native-crop/bottleneck) | thoroughly tested, no effect on transfer |
| domain-adversarial / removing the source axis | disproved — stripping 16 source dims changed transfer by <0.03 |
| training-composition tuning (43 mixtures) | no effect, entire range 0.905–0.915 |
| feature fusion H0/H1/H3 | +0.003 AUROC, and every arm had a *worse* worst-generator fold |
