import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT_DIR / "logs"
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

TRAIN_FILE = str((DATA_DIR / "train_small.jsonl").resolve())


def run_train_case(name: str, cfg: dict, timeout_sec: int = 400) -> tuple[int, str, Path]:
    cfg_path = ROOT_DIR / "configs" / f"silent_{name}.yaml"
    out_dir = ROOT_DIR / f"output_silent_{name}"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cfg["output"] = str(out_dir)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["SOUP_NO_AUDIT_LOG"] = "1"

    cmd = [sys.executable, "-m", "soup_cli", "train", "--config", str(cfg_path), "--yes"]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
        env=env,
    )
    combined = f"EXIT CODE: {proc.returncode}\n--- STDOUT ---\n{proc.stdout}\n--- STDERR ---\n{proc.stderr}\n"
    (LOGS_DIR / f"silent_{name}.log").write_text(combined, encoding="utf-8")
    return proc.returncode, combined, out_dir


def base_fast_config() -> dict:
    return {
        "base": "Qwen/Qwen2.5-0.5B-Instruct",
        "task": "sft",
        "data": {
            "train": TRAIN_FILE,
            "format": "alpaca",
            "val_split": 0.25,
            "max_length": 64,
        },
        "training": {
            "epochs": 1,
            "lr": 2e-4,
            "batch_size": 1,
            "seed": 42,
            "lora": {"r": 8, "alpha": 16, "target_modules": "auto"},
            "quantization": "none",
        },
    }


def inspect_adapter_weights(out_dir: Path) -> dict:
    st_files = list(out_dir.rglob("adapter_model.safetensors"))
    if not st_files:
        return {"found": False}
    from safetensors.torch import load_file
    weights = load_file(str(st_files[0]))
    has_nan = any(torch.isnan(t).any().item() for t in weights.values())
    has_inf = any(torch.isinf(t).any().item() for t in weights.values())
    nonzero = sum(1 for t in weights.values() if torch.count_nonzero(t).item() > 0)
    max_abs = max((float(t.abs().max().item()) for t in weights.values()), default=0.0)
    return {
        "found": True,
        "tensors": len(weights),
        "nonzero_tensors": nonzero,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "max_abs": max_abs,
    }


def test_silent_failure_inf_lr():
    """Кейс 1: lr: .inf проходит dry-run. Что произойдет при реальном обучении?"""
    cfg = base_fast_config()
    cfg["training"]["lr"] = float("inf")
    rc, out, out_dir = run_train_case("inf_lr", cfg)
    stats = inspect_adapter_weights(out_dir)
    print(f"\n[CASE inf_lr] rc={rc} | weights={stats}")
    # Если rc == 0 и веса содержат NaN/Inf или все нули — это 100% Silent Failure!
    assert not (rc == 0 and (stats.get("has_nan") or stats.get("has_inf") or stats.get("nonzero_tensors", 0) == 0)), (
        f"SILENT FAILURE FOUND (lr=.inf): rc=0, но веса испорчены! {stats}"
    )


def test_silent_failure_lora_r_zero():
    """Кейс 2: lora.r: 0 проходит dry-run. Что произойдет при реальном обучении?"""
    cfg = base_fast_config()
    cfg["training"]["lora"]["r"] = 0
    rc, out, out_dir = run_train_case("lora_r_zero", cfg)
    stats = inspect_adapter_weights(out_dir)
    print(f"\n[CASE lora_r_zero] rc={rc} | weights={stats}")
    assert not (rc == 0 and stats.get("nonzero_tensors", 0) == 0), (
        f"SILENT FAILURE FOUND (lora.r=0): rc=0, но адаптер пустой/нулевой! {stats}"
    )


def test_silent_failure_one_epoch_warmup_noop():
    """Кейс 3: 1 эпоха на малом датасете — обучается ли модель или шаг проходит с lr=0?"""
    cfg = base_fast_config()
    rc, out, out_dir = run_train_case("one_epoch_noop", cfg)
    stats = inspect_adapter_weights(out_dir)
    print(f"\n[CASE one_epoch_noop] rc={rc} | weights={stats}")
    assert not (rc == 0 and stats.get("nonzero_tensors", 0) == 0), (
        f"SILENT FAILURE FOUND (1 epoch warmup no-op): rc=0, но веса не обновились (все нули)! {stats}"
    )
