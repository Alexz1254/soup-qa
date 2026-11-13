# Soup CLI v0.77.2 QA Evaluation Suite

## Quick Start (One Command to Run)

1. Install dependencies:
   ```bash
   pip install "soup-cli[train]" pytest hypothesis
   ```

2. Run the fast config/CLI property-based test suite (~45 seconds, no model download required):
   ```bash
   pytest tests/test_config_fuzzing.py -v
   ```

3. Run the full suite including reproducibility & silent-failure training checks (downloads `Qwen/Qwen2.5-0.5B-Instruct`):
   ```bash
   pytest tests/ -v -s
   ```

## Repository Structure
- `configs/` — Base and mutated YAML configurations used in test runs.
- `data/` — Synthetic JSONL datasets (`train_small.jsonl`, `repro_prompts.jsonl`).
- `logs/` — Unedited raw logs of all runs (`doctor.log`, `fuzzing_raw*.jsonl`, `repro_run*.log`, `silent_*.log`).
- `reports/` — Pre-test strategy (`STRATEGY.md`), fuzzing table (`STEP2_FUZZING_REPORT.md`), reproducibility report (`STEP3_REPRODUCIBILITY.md`), bug reports (`BUG_REPORTS.md`), and final release note (`FINAL_NOTE.md`).
- `tests/` — Automated `pytest` + `Hypothesis` test suites.
