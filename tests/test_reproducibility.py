import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT_DIR / "logs"
REPORTS_DIR = ROOT_DIR / "reports"
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

TRAIN_FILE = str((DATA_DIR / "train_small.jsonl").resolve())
PROMPTS_FILE = DATA_DIR / "repro_prompts.jsonl"


def create_prompts_file():
    prompts = [
        {"prompt": "Say hello"},
        {"prompt": "What is 2+2?"},
        {"prompt": "Translate cat to Russian"},
    ]
    with PROMPTS_FILE.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def run_cmd(cmd: list[str], log_path: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["SOUP_NO_AUDIT_LOG"] = "1"
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    full_out = f"CMD: {' '.join(cmd)}\nEXIT CODE: {proc.returncode}\n\n--- STDOUT ---\n{proc.stdout}\n--- STDERR ---\n{proc.stderr}\n"
    log_path.write_text(full_out, encoding="utf-8")
    return proc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_weights(output_dir: Path) -> tuple[Path | None, dict[str, torch.Tensor]]:
    st_files = list(output_dir.rglob("adapter_model.safetensors"))
    if st_files:
        from safetensors.torch import load_file
        return st_files[0], load_file(str(st_files[0]))
    bin_files = list(output_dir.rglob("adapter_model.bin"))
    if bin_files:
        return bin_files[0], torch.load(str(bin_files[0]), map_location="cpu", weights_only=True)
    return None, {}


def extract_losses(output_dir: Path, log_text: str) -> list[float]:
    # 1. Пробуем достать из trainer_state.json (стандарт HuggingFace Trainer)
    state_files = list(output_dir.rglob("trainer_state.json"))
    if state_files:
        data = json.loads(state_files[0].read_text(encoding="utf-8"))
        losses = [float(entry["loss"]) for entry in data.get("log_history", []) if "loss" in entry]
        if losses:
            return losses
    # 2. Фоллбэк: парсим из текстового лога
    return [float(m) for m in re.findall(r"['\"]?loss['\"]?\s*[:=]\s*([0-9]+\.[0-9]+)", log_text)]


def test_reproducibility_same_seed():
    create_prompts_file()

    out1 = ROOT_DIR / "output_repro_1"
    out2 = ROOT_DIR / "output_repro_2"
    for d in (out1, out2):
        if d.exists():
            shutil.rmtree(d)

    base_cfg = {
        "base": "Qwen/Qwen2.5-0.5B-Instruct",
        "task": "sft",
        "data": {
            "train": TRAIN_FILE,
            "format": "alpaca",
            "val_split": 0.25,
            "max_length": 128,
        },
        "training": {
            "epochs": 2,
            "lr": 2e-4,
            "batch_size": 1,
            "seed": 42,
            "logging_steps": 1,
            "save_steps": 100,
            "lora": {
                "r": 8,
                "alpha": 16,
                "target_modules": "auto",
            },
            "quantization": "none",
        },
        "output": str(out1),
    }

    cfg1_path = ROOT_DIR / "configs" / "repro_run1.yaml"
    cfg2_path = ROOT_DIR / "configs" / "repro_run2.yaml"

    base_cfg["output"] = str(out1)
    cfg1_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

    base_cfg["output"] = str(out2)
    cfg2_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

    # Запуск 1 и Запуск 2
    p1 = run_cmd([sys.executable, "-m", "soup_cli", "train", "--config", str(cfg1_path), "--yes"], LOGS_DIR / "repro_run1.log")
    assert p1.returncode == 0, f"Run 1 failed! Check logs/repro_run1.log\n{p1.stderr}"

    p2 = run_cmd([sys.executable, "-m", "soup_cli", "train", "--config", str(cfg2_path), "--yes"], LOGS_DIR / "repro_run2.log")
    assert p2.returncode == 0, f"Run 2 failed! Check logs/repro_run2.log\n{p2.stderr}"

    # 1. Сравнение кривых loss
    losses1 = extract_losses(out1, p1.stdout + "\n" + p1.stderr)
    losses2 = extract_losses(out2, p2.stdout + "\n" + p2.stderr)
    loss_diffs = [abs(a - b) for a, b in zip(losses1, losses2)]
    max_loss_diff = max(loss_diffs) if loss_diffs else float("nan")

    # 2. Сравнение весов адаптера
    w_path1, weights1 = load_weights(out1)
    w_path2, weights2 = load_weights(out2)
    assert w_path1 is not None and w_path2 is not None, "Adapter weight files not found!"

    hash1 = sha256_file(w_path1)
    hash2 = sha256_file(w_path2)
    byte_identical = hash1 == hash2

    max_weight_diff = 0.0
    total_abs_diff = 0.0
    total_elems = 0
    for k in weights1:
        t1 = weights1[k].float()
        t2 = weights2[k].float()
        diff = torch.abs(t1 - t2)
        max_weight_diff = max(max_weight_diff, float(diff.max().item()))
        total_abs_diff += float(diff.sum().item())
        total_elems += diff.numel()
    mean_weight_diff = total_abs_diff / max(total_elems, 1)

    # 3. Сравнение результатов инференса (greedy decoding, temperature=0)
    res1_path = out1 / "infer_results.jsonl"
    res2_path = out2 / "infer_results.jsonl"

    inf1 = run_cmd(
        [sys.executable, "-m", "soup_cli", "infer", "--model", str(out1), "--input", str(PROMPTS_FILE), "--output", str(res1_path), "--temperature", "0"],
        LOGS_DIR / "repro_infer1.log",
    )
    inf2 = run_cmd(
        [sys.executable, "-m", "soup_cli", "infer", "--model", str(out2), "--input", str(PROMPTS_FILE), "--output", str(res2_path), "--temperature", "0"],
        LOGS_DIR / "repro_infer2.log",
    )

    out_texts_1 = res1_path.read_text(encoding="utf-8").splitlines() if res1_path.exists() else []
    out_texts_2 = res2_path.read_text(encoding="utf-8").splitlines() if res2_path.exists() else []
    outputs_exact_match = (out_texts_1 == out_texts_2) and len(out_texts_1) > 0

    # Запись подробного отчета
    report = f"""# Step 3: Reproducibility Report (Same Config, Same Seed = 42)

## 1. Measurements Summary
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (CPU backend, `torch {torch.__version__}`)
- **Seed:** `42` (2 epochs, 3 train samples, `batch_size=1`, `lr=2e-4`)
- **Loss curve Run 1:** `{losses1}`
- **Loss curve Run 2:** `{losses2}`
- **Max Loss Divergence:** `{max_loss_diff:.12e}`
- **Adapter SHA-256 Identical:** `{byte_identical}`
  - Run 1 (`{w_path1.name}`): `{hash1}`
  - Run 2 (`{w_path2.name}`): `{hash2}`
- **Max Weight Tensor Abs Diff:** `{max_weight_diff:.12e}`
- **Mean Weight Tensor Abs Diff:** `{mean_weight_diff:.12e}`
- **Inference Outputs Match (`temperature=0`):** `{outputs_exact_match}` (infer exit codes: `{inf1.returncode}`, `{inf2.returncode}`)

## 2. Inference Comparison
- **Run 1 outputs:** `{out_texts_1}`
- **Run 2 outputs:** `{out_texts_2}`
"""
    (REPORTS_DIR / "STEP3_REPRODUCIBILITY.md").write_text(report, encoding="utf-8")
    print("\n" + report)

    # Допуск (tolerance) для float32 на CPU: <= 1e-5
    assert max_weight_diff <= 1e-5, f"Weight divergence too high: {max_weight_diff}"
