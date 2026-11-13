# Bug Reports — Soup CLI v0.77.2 Release Candidate

## Environment (Common to All Reports)
- **Product:** `soup-cli` `v0.77.2`
- **OS:** Windows 11 (AMD64)
- **Python:** 3.13.2
- **Key Dependencies:** `torch 2.15.0+cpu`, `pydantic 2.14.3`, `transformers 5.22.2`, `peft 0.21.2`, `trl 0.29.1`, `datasets 5.2.0`
- **Hardware:** CPU mode (no CUDA driver active)

---

### BUG-01: `training.lora.r: 0` silently falls back to Full Fine-Tuning and exits with `rc=0` (Silent Failure)
- **Severity:** Critical (P0 — Silent Failure / Resource & Artifact Corruption)
- **Summary:** Setting `training.lora.r: 0` passes `--dry-run` validation and executes a full-parameter fine-tune instead of LoRA, saving a 1.97 GB `model.safetensors` + 3.95 GB `optimizer.pt` with zero `adapter_model.safetensors`, while reporting `Training Complete!` (`rc=0`).
- **Steps to Reproduce:**
  1. Create a valid SFT config based on `soup init --template chat` with `base: Qwen/Qwen2.5-0.5B-Instruct` and `training.lora.r: 0`.
  2. Run `soup train --config configs/silent_lora_r_zero.yaml --dry-run --yes` -> exits `0` (`Config valid. Ready to train!`).
  3. Run `soup train --config configs/silent_lora_r_zero.yaml --yes`.
- **Expected Result:** Schema validation rejects `training.lora.r: 0` (`r` must be `>= 1` when LoRA is configured), failing fast before training.
- **Actual Result:** Schema allows `r >= 0` (confirmed via `r: -8` error message: `Input should be greater than or equal to 0`). The trainer treats `r=0` as falsy and silently trains all base model weights in full fine-tuning mode, exiting with `rc=0`. On a target 8B / 4 GB VRAM setup, this causes unexpected OOM or massive disk writes instead of producing a LoRA adapter.
- **Evidence:** `logs/silent_lora_r_zero.log`, `output_silent_lora_r_zero/soup_run.json` (`run_id: run_20261112_161342_1e4d6a8b`).

---

### BUG-02: `training.lr: .inf` passes schema validation and leaves `NaN`-corrupted checkpoint on disk
- **Severity:** High (P1 — Incomplete Input Validation / Corrupted Checkpoint)
- **Summary:** Positive infinity (`lr: .inf` in YAML) satisfies Pydantic's `gt=0` check during `--dry-run` (`rc=0`), causing immediate `NaN` weight corruption at step 1.
- **Steps to Reproduce:**
  1. Set `training.lr: .inf` in `soup.yaml`.
  2. Run `soup train --config configs/silent_inf_lr.yaml --dry-run --yes` -> exits `0` (`Config valid. Ready to train!`).
  3. Run `soup train --config configs/silent_inf_lr.yaml --yes`.
- **Expected Result:** Config validation rejects non-finite `lr` (`math.isfinite(lr)` check) during config load / `--dry-run`.
- **Actual Result:** `--dry-run` succeeds. Real training runs step 1, corrupts all 96 adapter tensors to `NaN`, writes `checkpoint-1` to disk, and only then aborts with `RuntimeError: non-finite training metric eval_loss=nan at step 1`.
- **Evidence:** `logs/fuzzing_raw.jsonl` (`tag: out_of_range:training.lr=inf`), `logs/silent_inf_lr.log`.

---

### BUG-03: Boolean values (`true`/`false`) are silently coerced into numeric fields (`epochs`, `val_split`)
- **Severity:** Medium (P2 — Silent Type Coercion)
- **Summary:** In YAML, `training.epochs: true` is silently accepted as `1` epoch, and `data.val_split: true` is coerced to `1.0` (failing only on the `<= 0.5` range bound rather than type mismatch).
- **Steps to Reproduce:**
  1. Set `training.epochs: true` in `soup.yaml`.
  2. Run `soup train --config soup.yaml --dry-run --yes`.
- **Expected Result:** Strict type validation rejects `bool` for integer/float fields (`Input should be a valid integer/number`).
- **Actual Result:** Exits `0` (`Config valid. Ready to train!`), silently training for `1` epoch.
- **Evidence:** `logs/fuzzing_raw.jsonl` (`wrong_type:training.epochs=True`, `wrong_type:data.val_split=True`).

---

### BUG-04: Empty string `base: ""` passes `--dry-run` config validation
- **Severity:** Medium (P2 — Missing `min_length=1` Constraint)
- **Summary:** While `base: null` is rejected (`Input should be a valid string`), an empty string `base: ""` passes `--dry-run` (`rc=0`).
- **Steps to Reproduce:**
  1. Set `base: ""` in `soup.yaml`.
  2. Run `soup train --config soup.yaml --dry-run --yes`.
- **Expected Result:** Validation fails with a clear error indicating `base` must be a non-empty model ID or local path.
- **Actual Result:** Exits `0` (`Config valid. Ready to train!`), deferring failure until HuggingFace Hub download/resolution.
- **Evidence:** `logs/fuzzing_raw.jsonl` (`empty:empty_base`).

---

### BUG-05: Empty `data.train: ""` bypasses schema validation and triggers an unhandled exception
- **Severity:** Low (P3 — Poor Error Handling / UX)
- **Summary:** Setting `data.train: ""` passes schema validation and attempts to open the current working directory as a dataset file, failing with an unhandled traceback prompt instead of a validation error.
- **Steps to Reproduce:**
  1. Set `data.train: ""` in `soup.yaml`.
  2. Run `soup train --config soup.yaml --dry-run --yes`.
- **Expected Result:** Schema validator rejects empty string `data.train` with a clean validation error.
- **Actual Result:** Exits `1` with `Run with --verbose for the full traceback.` during data loading.
- **Evidence:** `logs/fuzzing_raw.jsonl` (`empty:empty_train_path`).
