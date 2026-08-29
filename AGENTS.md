# Agent Guide

## Repository purpose

This repository contains the implementation for a hackathon project. The authoritative challenge brief belongs in [`docs/problem-statement.md`](docs/problem-statement.md).

The project is a robust binary detector for purely AI-generated versus authentic images. It must generalize across generator families and common real-world transformations.

## Before making changes

1. Read `docs/problem-statement.md` in full.
2. Inspect the existing repository structure and working tree before editing.
3. Treat explicit requirements and constraints in the problem statement as authoritative.
4. Do not invent missing product requirements. Record assumptions in the relevant documentation when they materially affect the implementation.
5. Review the brief's "Open questions and ambiguities" section before making interface or evaluation assumptions.

## Non-negotiable challenge guardrails

- Use only public or properly licensed data; never train on test labels or the demonstration-only validation split.
- Keep every final model below 2 billion parameters.
- Public pretrained backbones are allowed, but merely using or replicating an existing pretrained AIGC detector is disallowed. The solution must contain a clear original technical contribution.
- Treat the target as binary classification of purely generated images versus authentic images. AI-edited or partially composited images are outside the stated focus.
- Do not rely on SynthID or other watermarks.
- The inference entry point must accept an image directory and produce JSON records containing `image_path` and an AIGC confidence score in `pred`.
- Keep augmentation, training, evaluation, and inference reproducible and suitable for public release.

## Working conventions

- Keep changes focused on the requested task.
- Preserve unrelated user changes in the working tree.
- Prefer simple, maintainable solutions suitable for a hackathon prototype.
- Keep credentials, API keys, tokens, and other secrets out of the repository. Document required environment variables in an example environment file when applicable.
- Add or update tests for behavior that can reasonably be tested.
- Run the most relevant available checks before handing off work. If checks cannot be run, explain why.
- Update documentation when setup steps, architecture, commands, or user-visible behavior change.

## Documentation

- `docs/problem-statement.md`: source of truth for the challenge, judging criteria, constraints, and required deliverables.
- `docs/README.md`: documentation index and status.

As the project grows, put durable architecture and setup notes under `docs/` and link them from `docs/README.md`.

## Scope and decision-making

- Optimize first for satisfying the documented hackathon requirements.
- Clearly distinguish requirements from implementation choices.
- When multiple interpretations are possible, choose the smallest reversible approach that remains consistent with the brief and record the assumption.
- Ask for clarification before making an irreversible or materially scope-changing decision.
- Keep formal rules, organizer clarifications, and non-binding workshop suggestions distinct in code and documentation. Do not present a suggested hybrid architecture as a mandatory requirement.

