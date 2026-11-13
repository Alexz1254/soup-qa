# Release Readiness Assessment: Soup CLI v0.77.2

## Release Verdict: **DON'T SHIP**

**Recommendation:** Block release of `v0.77.2` until **BUG-01** (`lora.r: 0` silent fallback to full fine-tuning with `rc=0`) and **BUG-02** (`lr: .inf` passing validation) are fixed. While Soup's core reproducibility and post-v0.75 unknown-key rejection work well, silent mode-switching from LoRA to full fine-tuning violates Soup's core safety promise and can silently corrupt workflows or trigger unexpected OOMs.

---

## 1. Test Strategy (Pre-Execution Summary)
Full pre-test strategy is documented in `reports/STRATEGY.md`. We prioritized risks by blast radius:
- **P0 (Critical):** Silent failures (`rc=0` with wrong weights, ignored settings, or unintended training modes).
- **P1 (High):** Nondeterminism across identical seeds and corrupted checkpoints.
- **P2/P3 (Medium/Low):** Schema coercion (`bool -> int`), non-finite floats, and unhandled tracebacks.
- **Deliberately Out of Scope:** Large-scale benchmark convergence (ML research scope), upstream PyTorch/PEFT internal math, and multi-node distributed clusters (tested on single-node Windows 11, Python 3.13.2, `torch 2.15.0+cpu`).

## 2. Generative Config & CLI Testing (`Hypothesis`)
We ran property-based fuzzing (`tests/test_config_fuzzing.py`) across 6 input classes against `soup train --dry-run --yes` (raw logs in `logs/fuzzing_raw.jsonl`, full table in `reports/STEP2_FUZZING_REPORT.md`):
- **Typos in keys:** All tested typos (`quantizaton`, `lerning_rate`, `epocs`, `model`, `val_splt`) were cleanly rejected (`rc=1`) with helpful suggestions.
- **Wrong types:** Strings/dicts/lists were rejected, **but `training.epochs: true` was silently accepted (`rc=0`)** due to non-strict Pydantic coercion.
- **Out-of-range values:** Negative values and `NaN` were rejected, **but `training.lora.r: 0` and `training.lr: .inf` were silently accepted (`rc=0`)**.
- **Empty fields:** Empty/comment-only YAML and `null` fields were rejected, **but `base: ""` was silently accepted (`rc=0`)**, and `data.train: ""` caused an unhandled exception (`Run with --verbose...`).
- **Conflicting flags & extreme lengths:** Mutually exclusive tracker flags, `--hf-resume` without `--push-as`, `val_split: 0.99` on 4 rows, and 5,000–10,000 char strings were all cleanly rejected (`rc=1`).

## 3. Reproducibility Analysis (`seed: 42`)
We executed two identical training runs (`tests/test_reproducibility.py`) on `Qwen/Qwen2.5-0.5B-Instruct` (2 epochs, 3 train samples, `batch_size=1`, `lr=2e-4`, CPU backend):
- **Loss curves:** Exact match (`[1.489572286605835, 1.489572286605835]`, max diff = `0.0`). *Note:* Step 1 executed with `learning_rate=0` due to default warmup, keeping weights unchanged before step 2 (`eval_loss` dropped from `2.045` to `1.76` after step 2).
- **Adapter weights (`adapter_model.safetensors`):** **Bit-exact match** (`SHA-256: 2c570d685d87bd67ed57c5af332715e6ca9f75d7ce94e69eebeff79253aea133`, `max_abs_diff = 0.0`, `mean_abs_diff = 0.0`).
- **Inference (`soup infer --temperature 0`):** 100% character-for-character identical outputs across all prompts.
- **Source of determinism & normal tolerance:** On CPU (`torch 2.15.0+cpu`), single-process thread scheduling and Soup's global seed propagation (`random`, `numpy`, `torch`, `transformers`) yield deterministic floating-point reduction order (`tolerance = 0.0` expected on identical CPU hardware). On CUDA GPUs with non-deterministic atomic additions or `fp16`/`bf16` mixed precision, a tolerance of `1e-5` to `1e-4` in max weight diff is normal.

## 4. Silent Failures & Blind Spots
- **Confirmed Silent Failure (`lora.r: 0`):** Passing `lora.r: 0` exited with `rc=0` (`Training Complete!`) while silently disabling LoRA and running full fine-tuning (`1.97 GB` `model.safetensors` saved instead of `adapter_model.safetensors`).
- **Where else we looked:** Checked `lr: .inf` (caught mid-run at step 1 with `eval_loss=nan`, though `checkpoint-1` with `NaN` weights remained on disk) and 1-epoch warmup behavior (confirmed 48/96 LoRA tensors updated non-zero).
- **What our tests would NOT catch:** Multi-GPU distributed deadlocks (FSDP/DeepSpeed), CUDA-specific kernel non-determinism or layer-streaming NF4 dequantization drift on >4 GB models, corrupted/adversarial Unicode inside large datasets, and `soup ship` / `soup merge` downstream artifact validation.

## 5. What Surprised Me & What Still Concerns Me
What surprised me most was the sharp contrast between Soup's strong runtime safeguards (such as catching `eval_loss=nan` mid-training and rejecting unknown YAML keys with "did you mean" hints) and its boundary gaps in schema validation—specifically how `lora.r: 0` silently morphs a lightweight LoRA job into a multi-gigabyte full fine-tune with exit code `0`. What still concerns me is how other falsy or boundary config values (such as `alpha: 0`, `max_steps: 0`, or zero-row filtered datasets) behave across non-SFT tasks (DPO, GRPO, RLHF) and whether `soup ship` or `soup merge` properly validates adapter integrity before pushing artifacts to production hubs.

## 6. AI Tools Disclosure
- **Where used:** Used an AI assistant to structure the pre-test risk matrix, scaffold `Hypothesis` property-based test templates, write PowerShell log-parsing one-liners, and format Markdown reports.
- **Accepted vs. Verified/Changed:** Accepted the 6-class fuzzing structure and tensor comparison helper. Manually verified every CLI flag against `soup train --help`, corrected initial schema assumptions (`model`/`data.path` -> `base`/`data.train` after running `soup init --template chat`), preserved failed run logs instead of overwriting them, and independently investigated the `output_silent_lora_r_zero` directory contents to uncover the silent full fine-tuning fallback.
