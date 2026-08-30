 Do not choose them from literature popularity or clean AUPRC alone. Choose each component based on its incremental,
 out-of-domain contribution to the full system.

 Also, these are three different kinds of choices:

 1. Evidence inputs: RGB, residuals, DCT, Fourier phase, neighbour relations.
 2. Sampling: one or multiple patch scales.
 3. Decision machinery: aggregation and uncertainty.

 Select them in that order. Otherwise you face a large, uninterpretable combinatorial search.

 The central selection question

 For every component, ask:

 │ When added to the simplest valid TRACE-RX baseline, does this component improve performance on unseen generators or
 │ transformations, without creating authentic-image false positives or unacceptable latency?

 A component can have good standalone AUPRC but still be useless if it makes exactly the same mistakes as the global model.

 Conversely, a relatively weak component may be valuable if it correctly identifies images on which the global model fails.

 ────────────────────────────────────────────────────────────────────────────────

 My current prior for each component

 ┌──────────────────┬─────────────────────────────────────────────┬────────────────────────────────────┬────────────────────┐
 │ Component        │ Why it might help                           │ Main danger                        │ Current priority   │
 ├──────────────────┼─────────────────────────────────────────────┼────────────────────────────────────┼────────────────────┤
 │ Native RGB       │ Learns local texture, edges and rendering   │ Semantic/content and source        │ Required forensic  │
 │                  │ artifacts that global resizing removes      │ shortcuts                          │ baseline           │
 ├──────────────────┼─────────────────────────────────────────────┼────────────────────────────────────┼────────────────────┤
 │ Residuals        │ Exposes sensor, demosaicing, denoising and  │ Destroyed by blur, noise and       │ High               │
 │                  │ decoder traces                              │ strong compression                 │                    │
 ├──────────────────┼─────────────────────────────────────────────┼────────────────────────────────────┼────────────────────┤
 │ Block DCT        │ Exposes quantization, periodicity and local │ Can become a JPEG/PNG or quality   │ Medium             │
 │                  │ frequency structure                         │ detector                           │                    │
 ├──────────────────┼─────────────────────────────────────────────┼────────────────────────────────────┼────────────────────┤
 │ Fourier phase    │ May expose spatial organization ignored by  │ Very sensitive to crop,            │ Medium-low,        │
 │                  │ magnitude features                          │ translation and resizing           │ diagnostic         │
 ├──────────────────┼─────────────────────────────────────────────┼────────────────────────────────────┼────────────────────┤
 │ Neighbour        │ May expose decoder upsampling and local     │ Often generator-specific and       │ Low initially      │
 │ relations        │ pixel dependencies                          │ fragile under JPEG/resize          │                    │
 ├──────────────────┼─────────────────────────────────────────────┼────────────────────────────────────┼────────────────────┤
 │ Multiple patch   │ Covers both fine artifacts and larger       │ Extra latency and more             │ Test after         │
 │ scales           │ structural inconsistencies                  │ opportunities for shortcuts        │ modalities freeze  │
 ├──────────────────┼─────────────────────────────────────────────┼────────────────────────────────────┼────────────────────┤
 │ Aggregation      │ Converts patch evidence into one image      │ Learned attention can select       │ Simple aggregation │
 │                  │ score                                       │ content instead of provenance      │ required           │
 ├──────────────────┼─────────────────────────────────────────────┼────────────────────────────────────┼────────────────────┤
 │ Learned          │ May identify when forensic evidence was     │ Can merely learn transform or      │ Stretch component  │
 │ uncertainty      │ erased                                      │ codec identity                     │                    │
 └──────────────────┴─────────────────────────────────────────────┴────────────────────────────────────┴────────────────────┘

 Your existing scalar EDA gives residuals the strongest preliminary support. Phase-neighbour coherence also survived the
 exploratory screen, but with low confidence. The current scalar DCT features were weak or shortcut-prone, although that does
 not prove a learned DCT representation will fail.

 ────────────────────────────────────────────────────────────────────────────────

 Recommended baseline

 Start with this:

   Global encoder
       │
       └── global score

   Native forensic branch
       ├── fixed-size native RGB patches
       ├── corresponding high-pass residual patches
       └── lightweight CNN
              │
              └── patch scores
                     │
              mean + standard deviation + upper quantile
                     │
              forensic score

   global score + forensic score
                │
         logistic stacking
                │
             AI score

 Specifically:

 - One patch scale initially, perhaps 192 or 224 pixels.
 - Several spatially distributed patches.
 - RGB and residual inputs.
 - Fixed aggregation:
     - mean;
     - standard deviation;
     - perhaps the 90th percentile.
 - Patch disagreement as a simple uncertainty proxy.
 - No learned uncertainty head yet.
 - No Fourier phase or neighbour branch initially.

 That gives you a credible minimal system against which everything else can be measured.

 ────────────────────────────────────────────────────────────────────────────────

 A staged ablation plan

 Stage 0: Make the experiment valid

 Before selecting modalities:

 - Use licensed, lineage-safe data.
 - Match or re-encode real and generated images to comparable formats and quality.
 - Hold out complete generators and authentic sources.
 - Apply transformations symmetrically to both labels.
 - Freeze the splits, metrics, training budget, and patch sampler.
 - Keep the organizer demonstration set out of all selection.

 Otherwise you may choose DCT because it recognizes PNG versus JPEG rather than AI provenance.

 Stage 1: Test each evidence family separately

 Train comparable forensic models:

 - F0: RGB only.
 - F1: residual only.
 - F2: DCT only.
 - F3: Fourier phase only.
 - F4: neighbour relations only.

 For each one, evaluate:

 1. Standalone performance.
 2. Performance after logistic stacking with the global model.
 3. Error correlation with the global model.
 4. Performance by generator.
 5. Performance under every official transformation.
 6. Authentic false-positive concentration by subtype/source.
 7. Latency and memory.

 The most important comparison is not:

   AUPRC(F1) versus AUPRC(F2)

 It is:

   Global + F1 versus Global
   Global + F2 versus Global

 This measures conditional value.

 Stage 2: Forward-select modalities

 Avoid evaluating all 2⁵ modality combinations.

 Start from RGB and add one component at a time:

   RGB
   RGB + residual
   RGB + residual + DCT
   RGB + residual + phase
   RGB + residual + neighbour relations

 Choose the best third component, if any. Then perform at most one final combination check.

 A likely sequence is:

 1. RGB.
 2. Add residuals.
 3. Compare DCT against phase as the third input.
 4. Test neighbour relations only if their isolated errors are clearly complementary.

 Initially, giving each modality its own score and combining scores through logistic stacking makes ablation easier. Once the
 winning modalities are known, you can test whether earlier feature-level fusion produces an additional gain.

 Stage 3: Select patch scale

 After modalities are frozen, compare:

 - One fixed native patch scale.
 - Two scales, such as fine and medium.
 - Whole-image canonical view plus native patches.

 Multiple scales should be kept only if they improve:

 - resize robustness;
 - crop robustness;
 - performance across small and large source images;
 - unseen-generator performance.

 Do not keep multiple scales merely because they improve clean in-domain accuracy.

 Stage 4: Select aggregation

 Test in increasing complexity:

 1. Mean patch score.
 2. Mean plus variance.
 3. Mean, variance, and upper quantile.
 4. Learned attention pooling.

 For purely generated images, every region should generally share the same origin. Therefore, simple statistical aggregation
 may be enough. Complex attention is more important for localized manipulation, which is outside this competition’s primary
 scope.

 Keep learned attention only if it improves over mean/variance pooling on unseen data. Inspect which patches it selects;
 otherwise it may focus on faces, text, or dataset-specific backgrounds.

 Stage 5: Add uncertainty last

 Begin with cheap uncertainty signals:

 - Variance between patch scores.
 - Difference between global and forensic scores.
 - Prediction changes under mild JPEG or blur.
 - Entropy of the fused score.

 Only then test a learned heteroscedastic uncertainty or availability head.

 Evaluate uncertainty using:

 - Calibration error.
 - Risk-versus-coverage curves.
 - Ability to predict forensic-branch failures.
 - Ability to improve fusion on unseen transforms.
 - Authentic FPR after degradation.

 Do not judge uncertainty by AUPRC alone. Its job is to identify when a score is unreliable.

 The learned availability target must also be cross-fitted and derived from evidence survival. It must not simply learn:

   JPEG detected → forensic unavailable

 That would be another codec shortcut.

 ────────────────────────────────────────────────────────────────────────────────

 Keep/kill rules

 Estimate ordinary seed variance before defining a numerical threshold. Then keep a component only if:

 1. Its gain is larger than normal run-to-run variation.
 2. The gain is consistent across finalist seeds or grouped folds.
 3. It improves unseen-generator or worst-transformation performance.
 4. It does not materially worsen worst-authentic-subtype FPR.
 5. It remains useful after matched re-encoding.
 6. Its errors are complementary to the existing experts.
 7. Its latency and memory fit the deployment budget.

 Use paired bootstrap intervals by master lineage for metric differences.

 Examples:

 ### Keep residuals if

 - global + RGB + residual beats global + RGB;
 - improvement survives generator holdout;
 - it particularly helps clean/resize/JPEG conditions;
 - the reliability mechanism successfully discounts it under destructive blur or noise.

 ### Kill DCT if

 - it performs well only when real images are JPEG and generated images are PNG;
 - the benefit vanishes after matched JPEG re-encoding;
 - it raises false positives on screenshots or previously compressed authentic images.

 ### Kill Fourier phase if

 - minor translation, crop, or resize reverses its predictions;
 - it provides no incremental fused benefit;
 - its apparent gain is unstable across generators.

 ### Kill neighbour relations if

 - they identify known decoder families but fail on held-out DiT generators;
 - JPEG and resizing make the branch misleading;
 - its errors are highly correlated with the RGB/residual branch.

 ### Kill learned uncertainty if

 - patch variance performs equally well;
 - it identifies transform type rather than expert failure;
 - calibration or risk-coverage does not improve on unseen journeys.

 ────────────────────────────────────────────────────────────────────────────────

 A manageable experiment ladder

 A practical ladder might contain:

 ┌────┬──────────────────────────────────┬────────────────────────────────────┐
 │ ID │ Forensic configuration           │ Purpose                            │
 ├────┼──────────────────────────────────┼────────────────────────────────────┤
 │ A0 │ No forensic branch               │ Global baseline                    │
 ├────┼──────────────────────────────────┼────────────────────────────────────┤
 │ A1 │ Native RGB, simple pooling       │ Minimum forensic branch            │
 ├────┼──────────────────────────────────┼────────────────────────────────────┤
 │ A2 │ Native RGB + residual            │ Highest-priority modality addition │
 ├────┼──────────────────────────────────┼────────────────────────────────────┤
 │ A3 │ A2 + DCT                         │ Compression/frequency hypothesis   │
 ├────┼──────────────────────────────────┼────────────────────────────────────┤
 │ A4 │ A2 + phase                       │ Phase complementarity hypothesis   │
 ├────┼──────────────────────────────────┼────────────────────────────────────┤
 │ A5 │ A2 + neighbour relations         │ Decoder-dependence hypothesis      │
 ├────┼──────────────────────────────────┼────────────────────────────────────┤
 │ A6 │ Best modality set + two scales   │ Scale hypothesis                   │
 ├────┼──────────────────────────────────┼────────────────────────────────────┤
 │ A7 │ A6 + learned aggregation         │ Pooling hypothesis                 │
 ├────┼──────────────────────────────────┼────────────────────────────────────┤
 │ A8 │ A6 + simple reliability features │ Minimal TRACE-RX                   │
 ├────┼──────────────────────────────────┼────────────────────────────────────┤
 │ A9 │ A6 + learned uncertainty         │ Full reliability ablation          │
 └────┴──────────────────────────────────┴────────────────────────────────────┘

 Run A1–A5 cheaply on discovery data. Promote only the best one or two configurations to complete training and multiple
 seeds.

 My expected minimal final system

 Unless experiments strongly contradict it, I would expect the best feasibility/novelty trade-off to be:

 - One global representation model.
 - Native RGB patches.
 - High-pass residual patches.
 - Possibly DCT if it survives matched-format controls.
 - One or two patch scales at most.
 - Mean/variance/upper-quantile aggregation.
 - Patch disagreement as simple uncertainty.
 - Logistic stacking.
 - At most two controlled interventions.

 Fourier phase, neighbour relations, learned attention, and learned heteroscedastic uncertainty should have to earn their
 place through clear incremental ablations. They should not be included merely to make TRACE-RX appear more sophisticated.
