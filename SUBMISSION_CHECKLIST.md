# TikTok TechJam 2026 Submission Checklist

Official deadline: **1 September 2026, 12:00 PM SGT**  
Submission portal: <https://tiktoktechjam2026.devpost.com/>

## Submission status

This checklist keeps its planning intent, but the status labels below are now
anchored to repository evidence as of 2026-09-01. "Not evidenced" means the
artifact or confirmation is not present in this checkout; it does not prove it
does not exist elsewhere. See [`docs/submission-handoff.md`](docs/submission-handoff.md)
for runnable scope, release blockers, and local-access requirements.

Draft the written entry with [`docs/devpost-draft.md`](docs/devpost-draft.md) and record the walkthrough using [`docs/demo-script.md`](docs/demo-script.md).

| Deliverable | Required | Status | Owner | Link / notes |
|---|---:|---|---|---|
| Devpost written description | Yes | Not evidenced | — | No description export or submission receipt is committed |
| Public code repository | Yes | Blocked: visibility unverified | — | `compliance.toml` records `repository_public = false`; verify logged-out access after the public audit |
| Comprehensive README | Yes | In progress | — | Setup/evaluation are documented; project license, team attribution, and clean-checkout proof remain unverified |
| Public YouTube demo | Yes | Not evidenced | — | The repository has no video URL or recording; the official brief says 2–4 minutes |
| Track-specific deliverables | Possibly | Blocked: human verification | — | `docs/problem-statement.md` is documented, while `problem_brief_verified = false` |
| Final Devpost submission | Yes | Not evidenced | — | Ensure the entry is submitted, not merely saved as a draft |

## 1. Devpost written description

- [ ] Project name
- [ ] Short tagline
- [ ] Selected TechJam track
- [ ] Problem statement and why it matters
- [ ] Intended users and stakeholders
- [ ] Solution overview
- [ ] End-to-end user workflow
- [ ] Technical architecture
- [ ] Model approach and fixed-threshold evaluation
- [ ] Technology stack
- [ ] Innovation and differentiation
- [ ] Benchmark results and robustness findings
- [ ] Practical impact and feasibility
- [ ] Challenges encountered
- [ ] Accomplishments
- [ ] Limitations and responsible-use considerations
- [ ] Future work
- [ ] Team members and contributions
- [ ] Public repository URL
- [ ] Public YouTube URL

### Project story to cover

- Detection of AI-generated images
- Robustness under JPEG compression, blur, resizing, noise, cropping, and color changes
- Image metadata, provenance, and transformation-chain visualization
- Model comparison and mismatch investigation
- True/false positive and negative analytics
- Condition-linked mismatch analysis
- Rights-aware dataset governance
- Interactive testing of new images

## 2. Public repository

- [ ] Repository visibility set to public
- [ ] Repository opens in a logged-out/incognito browser
- [ ] No credentials, tokens, private URLs, or Tailnet details committed
- [ ] No restricted datasets or improperly licensed assets published
- [ ] Source code is organized and understandable
- [ ] Build and run instructions tested from a clean checkout
- [ ] Evaluation instructions included
- [ ] Model and dataset configuration documented
- [ ] Prediction and metric schemas documented
- [ ] Architecture diagram included
- [ ] Representative results table included
- [ ] Dataset provenance and licenses documented
- [ ] Pretrained models and external APIs disclosed
- [ ] Known limitations documented
- [ ] Project license included
- [ ] Team attribution included

### README sections

- [ ] Overview
- [ ] Problem and motivation
- [ ] Product walkthrough
- [ ] Architecture
- [ ] Model methodology
- [ ] Dataset policy and provenance
- [ ] Installation
- [ ] Running the application
- [ ] Running evaluation
- [ ] Results
- [ ] Repository structure
- [ ] Limitations
- [ ] Responsible use
- [ ] License and acknowledgements

## 3. Three-minute YouTube demo

- [ ] Script completed
- [ ] Application prepared with approved local example images and benchmark artifacts
- [ ] Recording completed
- [ ] Final video is 2–4 minutes, matching `docs/problem-statement.md`
- [ ] Video uploaded publicly to YouTube
- [ ] Video plays while logged out
- [ ] Audio and on-screen text are legible
- [ ] No secrets or private user information appear in the recording
- [ ] YouTube URL added to Devpost

### Suggested timeline

| Time | Segment |
|---|---|
| 0:00–0:20 | Problem and motivation |
| 0:20–0:40 | Solution and architecture |
| 0:40–1:15 | Gallery, metadata, and transformation chain |
| 1:15–1:45 | Model and condition outputs |
| 1:45–2:15 | Confusion matrix, mismatches, and condition analytics |
| 2:15–2:40 | Test a new image |
| 2:40–3:00 | Results, impact, limitations, and closing |

## 4. Track-specific requirements

The general Devpost page says additional track-specific deliverables may apply. Verify the released AI Safety/AIGC problem statement before submission.

- [ ] Obtain and review the official track brief
- [ ] Record the exact required prediction/output schema
- [ ] Confirm required benchmark datasets and evaluation split
- [ ] Confirm required metrics
- [ ] Confirm whether model weights or an inference endpoint are required
- [ ] Confirm whether a report, notebook, or slide deck is required
- [ ] Confirm runtime, hardware, or resource-efficiency reporting requirements
- [ ] Confirm restrictions on external data, pretrained models, and APIs
- [ ] Confirm reproducibility requirements
- [ ] Add every confirmed track deliverable to the status table above

### Technical evidence to prepare

- [ ] Reproducible inference command
- [ ] Reproducible evaluation command
- [ ] Prediction files in the prescribed schema
- [ ] Overall benchmark metrics
- [ ] Per-condition robustness metrics
- [ ] TP, TN, FP, and FN counts
- [ ] Error and mismatch analysis
- [ ] Runtime and hardware measurements
- [ ] Methodology summary
- [ ] Dataset and model declarations

## 5. Final submission audit

- [ ] Devpost entry uses the correct track
- [ ] All team members are added
- [ ] Written description is complete and proofread
- [ ] Repository URL works while logged out
- [ ] YouTube URL works while logged out
- [ ] Track-specific files are attached or linked
- [ ] Claims in the description match measured results
- [ ] Demo matches the submitted code
- [ ] All public artifacts use consistent project naming
- [ ] Devpost entry has been formally submitted
- [ ] Submission confirmation captured

## 6. Finalist preparation

Finalists are scheduled to be announced on **8 September 2026**. The Grand Final is scheduled for **11 September 2026** at TikTok Singapore.

- [ ] Live demo prepared
- [ ] Offline demo recording available as fallback
- [ ] Pitch deck prepared
- [ ] Architecture slide prepared
- [ ] Benchmark and robustness slide prepared
- [ ] Impact and feasibility slide prepared
- [ ] Limitations and responsible-use slide prepared
- [ ] Q&A ownership assigned
- [ ] Demo tested on venue-compatible hardware and network

## Judging alignment

- [ ] **Technical Execution:** architecture, code quality, reliability, and deliberate model choices
- [ ] **Innovation & Problem Insight:** clear problem framing and differentiated approach
- [ ] **Impact & Relevance:** tangible value to users and stakeholders
- [ ] **Feasibility & Practicality:** sustainable and realistic deployment approach
- [ ] **Presentation & Communication:** coherent story and confident technical Q&A

## Open questions

- [ ] What is the exact official name of our selected track?
- [ ] What additional artifacts does the track brief require?
- [ ] Which benchmark result should headline the submission?
- [ ] Will the application be publicly deployed, or shown only through the video?
- [ ] Who owns the description, README, video, and final submission?
