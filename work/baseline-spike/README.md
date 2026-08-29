# TechJam zero-training baseline spike

Disposable research harness for Track 5. It evaluates three licensed pretrained detectors and five native-image shortcut heuristics on a fixed balanced sample of the CC BY 4.0 SID-Set validation split. It performs no fitting or threshold calibration.

The seven conditions are clean input plus the six severe settings from the official Track 5 transformation list: JPEG quality 30, Gaussian blur sigma 2, resize to 0.25x and upscale, Gaussian noise sigma 0.10, brightness/contrast/saturation jitter of 20%, and center crop to 80%.

This is a feasibility spike. The 24-image sample is deliberately small because the available Torch build is CPU-only; results are directional, not leaderboard estimates.
