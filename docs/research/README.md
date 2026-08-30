# Research index and review contract

This directory contains decision-oriented research for the AIGC detector. The
challenge brief remains authoritative; research reports evaluate implementation
choices rather than create new requirements.

## Reports

- [SOTA AIGC image-detection report](aigc-detection-sota.html) — broad methods,
  benchmarks, implementation recipes, and an offline paper library.
- [Architecture reading of seven generalizable AIGC detection papers](seven-paper-architecture-reading.html)
  — deep paper arguments, model data flows, protocol-aware critiques, and
  implications for TRACE-RX.
- [Idea validation: watermark-like cues and resolution-aware inference](idea-validation-watermarks-resolution.html)
  — focused evidence review and proposed falsification experiments for the
  team's 29 August 2026 brainstorm.
- [Offline paper library](papers/) — primary PDFs archived by the broader SOTA
  review. A report may link to public primary sources without duplicating the
  PDF locally.

## Review contract

Use one focused HTML report per decision or tightly related set of hypotheses.
Every report should be understandable without reading chat history and include:

1. **Frozen question and terminology.** State exactly what is being evaluated
   and separate easily confused concepts.
2. **Decision-first verdict.** Use `pursue`, `pursue with changes`, `experiment
   only`, or `do not pursue`; state confidence and the main reason.
3. **Evidence ledger.** For every important source, record venue/status,
   protocol or population, result, relevance, and the largest validity limit.
4. **Evidence labels.** Mark claims as `direct`, `adjacent`, `general
   precedent`, `author-reported`, or `inference`. Never present analogy as
   validation.
5. **Falsification plan.** Define counterfactuals, baselines, held-out axes,
   acceptance criteria, and what result would kill the idea.
6. **Challenge fit.** Check data licensing, the sub-2B limit, watermark rules,
   pure-generation scope, transformations, reproducibility, and inference
   interface implications.
7. **Primary citations.** Link the paper or official proceedings page beside
   the claim it supports. Cite every number. Do not merge results from
   incompatible protocols into one ranking.
8. **Search audit.** Record the search date, query families, inclusion boundary,
   and known gaps. Write `not found in this search` rather than claiming that no
   literature exists.

## Recommended workflow

For now, use this contract and the focused report as the manual template. After
two or three reports expose which fields and review checks are genuinely stable,
encode the workflow as a repository skill. A skill is the appropriate eventual
automation unit because the task is a repeatable research procedure using
existing browsing and file tools. A plugin would add packaging and integration
overhead without solving the current style-quality problem.

A future skill should produce both:

- a readable HTML report from a shared template; and
- a small machine-readable evidence ledger (JSON or YAML) containing source,
  evidence class, claim, protocol, limitation, and decision impact.

It should stop or visibly downgrade confidence when it cannot reach primary
sources, when protocols are incomparable, or when a claimed research gap is
based only on keyword search.
