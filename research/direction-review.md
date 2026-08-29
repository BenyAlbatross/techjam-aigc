# Direction research review

> Historical snapshot recovered from the referenced research conversation. Internal citation markers refer to that conversation; validate sources before external publication.

# Re-review: updated direction

Building on the prior research, I would **keep the overall project thesis**, but adjust the architecture and research hierarchy in three important ways.

Your current plan is fundamentally strong: it defines a narrow binary task, separates partial/ambiguous AI cases, explicitly controls lineage leakage, evaluates held-out generators and real-source families, measures post-processing robustness, and freezes the method before external evaluation. Those decisions are more important than small architectural optimisations. fileciteturn2file0L1-L20

The main update is that the evidence now points toward **representation choice + training prior + robustness objective** being at least as important as the classifier architecture itself.

## 1. Revised core hypothesis

Previously, I recommended:

> DINOv3-L + spatial patch aggregation + clean/degraded consistency.

I would now broaden that slightly to:

> **Can a modern pretrained representation expose generator-independent provenance information, and can spatial/global evidence aggregation plus degradation-consistency preserve that information under realistic distribution shifts?**

This formulation is better because newer research suggests there are **two promising and somewhat competing representation regimes**:

- DINO-style self-supervised visual features, where distributed patch evidence appears highly useful;
- language-aligned Perception Encoder features, where explicit forensic-semantic prototypes may expose provenance structure more effectively.

The August 2026 PE-SPC paper reports that ordinary PE linear probing actually trails DINOv3-linear by 4.1 points on its in-the-wild evaluation, but that **Semantic Prototype Calibration on PE reverses this and surpasses DINOv3 across cross-generator, post-processing, and in-the-wild benchmarks**. citeturn171075academia50

So I would no longer hard-code “DINOv3 is the final model” into the research story.

---

# 2. Architecture I would investigate now

The cleanest experimental architecture is a **shared evaluation framework with two representation branches**, not an ensemble.

```text
                           INPUT IMAGE
                               │
               ┌───────────────┴────────────────┐
               │                                │
        canonical/native policy           provenance metadata
               │                          (diagnostic only)
               ▼
       REPRESENTATION EXPERIMENT
               │
       ┌───────┴──────────┐
       │                  │
   DINOv3-L            PE-Core-L
  visual SSL          vision-language
       │                  │
       ▼                  ▼
  global token       semantic prototype
      baseline          baseline
       │                  │
       ├──────────┐       │
       ▼          ▼       ▼
 Gaussian       spatial   SPC-style
 head           head      classifier
                 │
                 ▼
       degradation consistency
                 │
                 ▼
       ───────────────────────
          COMMON EVALUATOR
       ───────────────────────
       held-out generator
       held-out real source
       clean
       single degradation
       ordered compound chains
       score bias
       worst-group metrics
```

The objective is **not** to combine these models.

The objective is to determine where the transferable signal actually exists.

---

# 3. Highest-value experiment: DINOv3 versus PE

This is now more informative than comparing DINOv2 versus OpenCLIP or several traditional forensic models.

## Candidate A — DINOv3-L

DINOv3 remains extremely strong as the default vision representation. Meta positions DINOv3 as a universal frozen visual backbone, and its representations have already become a major baseline in AIGI detection. citeturn639658search1

Advantages:

- approximately 300M parameters;
- strong dense spatial tokens;
- much lower risk under the <2B constraint;
- naturally supports PatchHead-style experiments;
- relatively simple inference path.

## Candidate B — PE-Core-L14

PE-Core-L has a **0.32B vision tower plus a 0.31B text tower**. If only the vision encoder is loaded for inference, it is similar in scale to DINOv3-L. Even if both towers count, it remains well below 2B. citeturn171075search0turn171075search2

The interesting characteristic is not merely that it is a VLM. PE-SPC argues that its feature space contains **local organisation associated with provenance semantics**, but that a conventional linear classifier does not exploit it well. citeturn171075academia50

This gives you a very strong research question:

> Is robust AI-image provenance encoded predominantly as spatial visual irregularity, or as a higher-level semantic/provenance structure in modern pretrained representations?

That is substantially more interesting than simply comparing two model accuracies.

---

# 4. Keep PatchHead, but reinterpret what it proves

PatchHead is still one of the strongest architectural leads.

It reports:

- +3.0 points average balanced accuracy over its strongest prior method;
- worst-case accuracy increasing from 82.4% to 89.4%;
- only 8.6% more trainable parameters;
- negligible additional FLOPs.

Most importantly, it reports that preserving the **2D spatial organisation of patch tokens** improves cross-dataset transfer compared with relying on a global CLS token. citeturn171075academia49

The earlier interpretation was:

> patch evidence is better than global evidence.

I would now avoid that conclusion.

The more defensible interpretation is:

> **pure global compression can discard useful forensic evidence.**

That distinction matters because GlobalForge provides evidence in the opposite direction: detectors that rely excessively on local artifacts become fragile under JPEG, blur and propagation. It improves robustness by suppressing local shortcuts and encouraging long-range structural reasoning. citeturn639658academia49

These findings are not contradictory.

They suggest a spectrum:

```text
bad extreme                                   bad extreme
─────────────                                 ─────────────
single CLS                           isolated forensic patches
loses spatial signal                overfit fragile local artifacts

                    ↓ likely useful region ↓

     distributed spatial evidence + global contextual integration
```

Therefore, if you extend PatchHead at all, the most principled modification would be a **very lightweight global-context pathway**, not frequency-domain branches or complex forensic experts.

For example:

\[
z_p = \mathrm{PatchHead}(P)
\]

\[
z_g = \mathrm{CLS}(P)
\]

\[
z = z_p + \alpha\, Wz_g
\]

where \(\alpha\) is learned or fixed small.

That gives a clean three-way ablation:

1. global only;
2. spatial only;
3. spatial + global context.

That experiment directly reconciles the two current research directions.

---

# 5. Add one very cheap but important baseline: Gaussian discriminants

This should now be mandatory.

The new ECCV 2026 Prior-Conditioned Gaussian Discriminants work evaluated **39 datasets / 7.1 million images** and found that simple closed-form heads can often rival learned detector heads when the **training prior and encoder are matched**. citeturn171075academia48

This changes how you should interpret model improvements.

If:

```text
DINOv3 + PatchHead > DINOv3 + linear
```

that does **not necessarily** mean PatchHead has discovered substantially better forensic evidence.

You should compare:

```text
DINOv3 + logistic
DINOv3 + LDA
DINOv3 + regularised QDA/Gaussian
DINOv3 + PatchHead
```

If the Gaussian classifier closes most of the gap, then your actual finding is:

> the pretrained representation is already strongly separable, and modelling its class distribution matters more than architectural complexity.

That would still be an excellent TechJam result.

It also means you should record every result as the tuple:

\[
(\text{training prior},\text{encoder},\text{head})
\]

rather than reporting only “model name.”

---

# 6. Data strategy is becoming more important than model size

One of the strongest findings from the large 2026 open-detector benchmark is that **training-data alignment causes 20–60% performance variation even within architecturally identical detector families**. It also found that no detector is universally best and that modern generators can sharply reduce detector performance. citeturn639658academia48

This strongly validates the original plan’s emphasis on dataset composition.

I would make your training prior a deliberate experimental variable.

Instead of simply:

```text
Real / AI = 50 / 50
```

define a balanced hierarchy:

```text
AI
├─ autoregressive / multimodal generators
├─ diffusion / flow generators
├─ proprietary generators
└─ open-weight generators

Real
├─ camera photographs
├─ processed photographs
├─ CGI / renders
├─ digital art
├─ screenshots
└─ web-compressed images
```

Sample first at the group level, then at the image level.

Otherwise large generator datasets will dominate the learned feature boundary.

---

# 7. Hard-negative real data should become a first-class workstream

I would increase the priority of real negative diversity.

False positives on:

- Blender renders;
- video-game screenshots;
- digital illustrations;
- HDR imagery;
- heavily denoised mobile photographs;
- sharpened images;
- synthetic-looking architecture;
- macro photography;
- astrophotography;

are arguably more damaging than missing a subset of AI images.

The project should report:

\[
FPR_{\text{CGI}},\quad
FPR_{\text{digital-art}},\quad
FPR_{\text{screenshot}},\quad
FPR_{\text{edited-photo}}
\]

not only aggregate real-image FPR.

This also improves the judging narrative because you can show that the system is designed not merely to “catch AI,” but to **avoid incorrectly accusing legitimate content**.

---

# 8. The transform programme should move closer to real propagation

The original six synthetic transformations are still useful, but newer evidence from RRBench and GlobalForge reinforces that **transmission and re-digitisation pipelines are disproportionately difficult**. RRBench found substantial detector degradation after internet transmission and re-digitisation. citeturn639658search46

Therefore, I would change the transformation hierarchy to:

### Tier 1 — primitive operations

- JPEG;
- resize;
- blur;
- crop;
- noise;
- colour modification.

### Tier 2 — propagation pipelines

Model actual usage:

```text
generated
 → app resize
 → JPEG
 → screenshot
 → crop
 → JPEG
```

or

```text
uploaded image
 → server transcoding
 → downscale
 → display
 → screenshot
```

### Tier 3 — physical recapture

Post-competition:

```text
display → camera → crop → resize
```

For TechJam, Tier 2 is much more valuable than exhaustively testing every mathematical pair.

---

# 9. Add transformation *bias* as distinct from transformation *damage*

Your current direction already measures accuracy after transformations. Keep that, but explicitly distinguish two failure types.

### Loss of discriminative information

Example:

```text
real scores: 0.1 → 0.45
AI scores:   0.9 → 0.55
```

Classes collapse toward each other.

### Transformation-induced bias

Example:

```text
real scores: 0.1 → 0.4
AI scores:   0.9 → 1.2
```

Ranking may remain good, but all images shift toward AI.

This is important operationally because AUROC may remain high in the second case while a fixed deployed threshold becomes badly calibrated.

For every transform \(T\), calculate:

\[
\Delta_{\text{real},T}
=
E[s(T(x))-s(x)\mid y=0]
\]

and

\[
\Delta_{\text{AI},T}
=
E[s(T(x))-s(x)\mid y=1].
\]

This is an extremely cheap addition and produces useful visualisations.

---

# 10. Hard-example generation is interesting, but not for the hackathon core

PROBE, accepted at ICML 2026, takes an active approach: use the detector to identify difficult regions of a generator’s latent/manifold space, then generate hard examples around detector boundaries. It reports improved unseen-generator generalisation. citeturn639658search9

This is conceptually powerful because it says:

> simply adding more random AI images is inefficient; generate examples specifically where your detector is uncertain or wrong.

However, this should be a **future extension**, unless the team already controls an open generator and latent manipulation pipeline.

For now, you can approximate the idea more cheaply:

1. train baseline;
2. score a large candidate pool;
3. retain false negatives and near-boundary generated images;
4. rebalance them into the next training set.

That preserves the active-learning intuition without requiring generator-level optimisation.

---

# 11. I would not yet add frequency-domain branches

Several current papers continue to report gains from combinations of spatial, DCT/frequency, residual and semantic features. For example, the newly published *All-around Forgery Clues* combines spatial/frequency residual evidence with a fine-tuned vision-language representation and reports strong generalisation across 30 sub-datasets. citeturn639658search0

SCA-Det, SDA-Det and MS-DCL similarly continue to find useful low-level artifact structure. citeturn639658search4turn639658search8turn639658search6

So frequency evidence is **not obsolete**.

But it remains a poor hackathon-core choice because it adds another family of assumptions:

```text
visual foundation features
        +
frequency features
        +
residual features
        +
multi-scale fusion
```

If performance improves, it becomes difficult to explain why.

I would only add a spectral probe diagnostically:

- radial FFT profile;
- DCT block statistics;
- high-frequency energy.

Use them to analyse **what transformations destroy**, not initially as classifier input.

---

# 12. Revised experiment hierarchy

I would now freeze the research ladder approximately like this:

| Priority | Experiment | Why |
|---|---|---|
| **P0** | DINOv3-L + logistic | foundational baseline |
| **P0** | DINOv3-L + Gaussian/LDA | determine representation separability |
| **P0** | DINOv3-L + PatchHead | test spatial evidence |
| **P0** | above + clean/degraded consistency | central robustness intervention |
| **P0** | PE-L + linear | encoder comparison |
| **P1** | PE-L + SPC-style classifier | strongest emerging challenger |
| **P1** | PatchHead + global-context fusion | reconcile spatial/global evidence |
| **P1** | LoRA on best branch | only if frozen features plateau |
| **P2** | hard-negative mining | data improvement |
| **P2** | diagnostic FFT/DCT probes | explanatory analysis |
| **Defer** | multi-expert frequency fusion | excessive scope |
| **Defer** | generative adversarial/probe sample search | infrastructure-heavy |
| **Defer** | multi-backbone inference ensemble | weak causal story |

---

# 13. Updated architecture recommendation

If I had to choose **one likely final system today**, before experiments:

```text
DINOv3-L
   │
patch tokens + CLS
   │
┌──┴────────────────────┐
│                       │
3×3 spatial aggregator  global projection
│                       │
└──────────┬────────────┘
           │
       fused feature
           │
      binary head
           │
 clean/degraded
 consistency training
           │
 temperature calibration
           │
 prediction + uncertainty diagnostics
```

But I would explicitly reserve one research gate:

> **PE-L + SPC must be tested before the final backbone is frozen.**

If PE-SPC materially beats DINOv3 on **held-out generators plus degraded images**, use PE.

If it only wins on clean data, retain DINOv3.

---

# 14. Model selection should prioritise the minimum failure mode

Do not rank architectures using average accuracy.

Use something closer to:

\[
S =
0.35R_{\text{unseen-gen}}
+
0.25R_{\text{degraded}}
+
0.20R_{\text{real-source}}
+
0.20R_{\text{worst-group}}
\]

with a hard constraint such as:

\[
\max_g FPR_g \le \tau
\]

for difficult real-image groups.

An even simpler hackathon metric would be:

```text
Primary:
    minimum AUROC across
    {held-out generator,
     degradation,
     real-source shift}

Tie-break:
    average robust AUROC

Guardrail:
    hard-negative real FPR
```

This prevents one spectacular easy-set result from masking catastrophic failure elsewhere.

---

# 15. Better final product framing

The product should not be marketed primarily as:

> “AI image detector.”

That space is crowded and difficult to make credible.

A stronger framing is:

> **AI image detection under distribution shift.**

The UI can demonstrate:

```text
Original image
AI probability: 0.91

After JPEG:
0.88

After resize → JPEG:
0.82

After screenshot simulation:
0.76

Stability:
HIGH

Validated operating envelope:
✓ generator-generalisation set
✓ web compression
✓ resizing
△ physical recapture unvalidated
```

That directly exposes what the research contributes.

You could also show a **robustness fingerprint**:

```text
JPEG            █████████  stable
Resize          ██████████ stable
Blur            ███████    moderate
Screenshot      ██████     moderate
Heavy crop      ████       unstable
```

That would be substantially more differentiated than a single “93% AI” meter.

---

# Overall assessment after the second review

The original research direction remains valid. The new evidence actually strengthens its central assumptions:

- **unseen-generator evaluation is necessary**; large contemporary benchmarks continue to show severe generalisation instability. citeturn639658academia48
- **distributed patch evidence matters**, but it should not become local-artifact shortcut learning. citeturn171075academia49turn639658academia49
- **training prior/data composition is a first-order variable**, not merely a dataset implementation detail. citeturn171075academia48
- **modern foundation representations are the right starting point**, but DINOv3 now has a credible challenger in PE + semantic prototype calibration. citeturn171075academia50
- **robustness must be trained explicitly**, rather than expected to emerge from clean-data training.
- **real-world propagation chains are more valuable than exhaustive artificial transform grids**. citeturn639658search46turn639658academia49

The strongest project therefore becomes a controlled study of **where transferable provenance evidence exists, how it should be aggregated, and whether it survives real distribution shift**, rather than an attempt to accumulate every known forensic cue into one detector.
