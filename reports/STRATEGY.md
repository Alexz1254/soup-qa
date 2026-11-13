# Test Strategy (Pre-Execution) — Soup CLI v0.77.2

## 1. Main Objective & Risk Prioritization
The primary release risk is a **silent failure**: a run that exits with code `0` (or passes release gates) while producing corrupted weights, ignoring config parameters, or training on broken data.

Risks are prioritized from highest to lowest impact:
- **P0 (Critical — Silent Data/Model Corruption):** CLI reports success (`rc=0`), but ignores hyperparameters, trains with `0` parameters, produces `NaN` weights, or overwrites existing artifacts without warning.
- **P1 (High — Nondeterminism & Reproducibility):** Identical runs with a fixed `seed` diverge beyond expected floating-point tolerances, or checkpoint resumption breaks training continuity.
- **P2 (Medium — Weak Input/Schema Validation):** Invalid types (e.g., `bool` coerced to `int`), out-of-range values (`inf`, `0`), or empty required strings pass `--dry-run` validation silently.
- **P3 (Low — Unhandled Exceptions / UX):** Invalid inputs fail fast (`rc != 0`), but expose raw Python tracebacks instead of user-friendly CLI error messages.

## 2. Execution Order (What We Test First & Why)
1. **Environment & Baseline Smoke Check:** Verify CLI diagnostics (`soup doctor`) and validate a minimal synthetic pipeline (`--dry-run`) before running heavy training.
2. **Generative Config & CLI Fuzzing (Property-Based):** Use `Hypothesis` against `--dry-run` to rapidly test 6 classes of malformed inputs (typos, wrong types, out-of-range values, empty fields, conflicting flags, extreme lengths) and detect silent acceptances early.
3. **Determinism & Reproducibility:** Run two identical CPU training jobs with `seed: 42` on a lightweight model (`Qwen/Qwen2.5-0.5B-Instruct`) and compare step-by-step loss, adapter tensor weights, and inference outputs.
4. **Targeted Silent-Failure Hunting:** Probe edge cases suggested by fuzzing (e.g., `lora.r: 0`, `lr: .inf`, `epochs: true`, empty/malformed dataset rows, `soup ship` / `soup diagnose` gates).

## 3. Out of Scope (What We Deliberately Do Not Test)
- **Large-scale model benchmark quality:** Evaluating downstream LLM accuracy across 100+ epochs is an ML research task, not CLI release verification.
- **Upstream library internals:** We do not test internal math of `PyTorch`, `Transformers`, or `PEFT` — only how `soup-cli` configures and invokes them.
- **Multi-node distributed clusters:** Testing is scoped to a single-node environment (Windows 11, Python 3.13, CPU backend).
