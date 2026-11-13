import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

ROOT_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
RAW_LOG_FILE = LOGS_DIR / "fuzzing_raw.jsonl"
DATA_FILE = str((ROOT_DIR / "data" / "train_small.jsonl").resolve())

# Базовый валидный конфиг (на базе soup init --template chat, с маленькой моделью)
VALID_CONFIG = {
    "base": "Qwen/Qwen2.5-0.5B-Instruct",
    "task": "sft",
    "data": {
        "train": DATA_FILE,
        "format": "alpaca",
        "val_split": 0.25,
        "max_length": 128,
    },
    "training": {
        "epochs": 1,
        "lr": 2e-5,
        "batch_size": 1,
        "seed": 42,
        "lora": {
            "r": 8,
            "alpha": 16,
            "target_modules": "auto",
        },
        "quantization": "none",
    },
    "output": "./output_fuzz",
}


def run_soup_cli(config_payload: str | dict | None, extra_args: list[str], tmp_path: Path, case_tag: str):
    """Запускает soup train --dry-run и логирует сырой результат."""
    cfg_path = tmp_path / "soup_fuzz.yaml"
    if isinstance(config_payload, str):
        cfg_path.write_text(config_payload, encoding="utf-8")
    elif config_payload is None:
        cfg_path.write_text("", encoding="utf-8")
    else:
        cfg_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    cmd = [sys.executable, "-m", "soup_cli", "train", "--config", str(cfg_path), "--dry-run", "--yes", *extra_args]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["SOUP_NO_AUDIT_LOG"] = "1"

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=env,
        )
        rc = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc = -999
        stdout = str(exc.stdout or "")
        stderr = "TIMEOUT_EXPIRED"

    combined = f"{stdout}\n{stderr}"
    if rc == 0:
        behavior = "SILENT_ACCEPTANCE"
    elif "Traceback (most recent call last):" in combined:
        behavior = "TRACEBACK_CRASH"
    else:
        behavior = "CLEAR_ERROR"

    record = {
        "tag": case_tag,
        "returncode": rc,
        "behavior": behavior,
        "extra_args": extra_args,
        "config": config_payload,
        "stdout": stdout[-1000:],
        "stderr": stderr[-1000:],
    }
    with RAW_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return rc, behavior, combined


# ---------------------------------------------------------
# 0. Контрольный Smoke-тест (базовый конфиг должен проходить)
# ---------------------------------------------------------
def test_valid_baseline_dry_run(tmp_path):
    rc, behavior, out = run_soup_cli(VALID_CONFIG, [], tmp_path, "baseline_valid")
    assert rc == 0, f"Базовый конфиг не прошел dry-run! Вывод:\n{out}"


# ---------------------------------------------------------
# 1. Typos in keys (Опечатки в ключах)
# ---------------------------------------------------------
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    typo_key=st.sampled_from([
        "quantizaton",      # известный пример из README
        "lerning_rate",     # опечатка в training
        "epocs",            # опечатка в training
        "model",            # старое/чужое имя вместо base
        "val_splt",         # опечатка в data
        "unknown_top_key",  # совершенно левый ключ
    ])
)
def test_fuzz_typos_in_keys(typo_key, tmp_path):
    cfg = copy.deepcopy(VALID_CONFIG)
    if typo_key in ("lerning_rate", "epocs", "quantizaton"):
        cfg["training"][typo_key] = 123
    elif typo_key == "val_splt":
        cfg["data"][typo_key] = 0.2
    else:
        cfg[typo_key] = "unexpected_value"

    rc, behavior, out = run_soup_cli(cfg, [], tmp_path, f"typo:{typo_key}")
    # По документации v0.75+ неизвестные ключи ОБЯЗАНЫ отвергаться
    assert behavior != "SILENT_ACCEPTANCE", f"Тихое принятие опечатки '{typo_key}'!\n{out}"


# ---------------------------------------------------------
# 2. Wrong types (Неверные типы данных)
# ---------------------------------------------------------
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    field_and_val=st.sampled_from([
        ("training.epochs", "three"),
        ("training.epochs", True),          # В Python isinstance(True, int) == True! Ловушка!
        ("training.epochs", 1.5),
        ("training.lr", [0.001]),
        ("training.lr", {"val": 1e-4}),
        ("data.val_split", "ten_percent"),
        ("data.val_split", True),           # Еще одна проверка на bool -> float/int coercion
        ("training.lora.r", "sixty_four"),
        ("base", 12345),
        ("training", "not_a_dict"),
    ])
)
def test_fuzz_wrong_types(field_and_val, tmp_path):
    path, bad_val = field_and_val
    cfg = copy.deepcopy(VALID_CONFIG)
    parts = path.split(".")
    target = cfg
    for p in parts[:-1]:
        target = target[p]
    target[parts[-1]] = bad_val

    rc, behavior, out = run_soup_cli(cfg, [], tmp_path, f"wrong_type:{path}={bad_val!r}")
    assert behavior == "CLEAR_ERROR", f"Ожидалась понятная ошибка для {path}={bad_val!r}, получено: {behavior}\n{out}"


# ---------------------------------------------------------
# 3. Out-of-range values (Значения вне допустимого диапазона)
# ---------------------------------------------------------
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    field_and_val=st.sampled_from([
        ("training.epochs", 0),
        ("training.epochs", -5),
        ("training.lr", 0.0),
        ("training.lr", -1e-4),
        ("training.lr", float("nan")),
        ("training.lr", float("inf")),
        ("data.val_split", -0.1),
        ("data.val_split", 1.0),
        ("data.val_split", 1.5),
        ("data.max_length", 0),
        ("data.max_length", -128),
        ("training.lora.r", 0),
        ("training.lora.r", -8),
        ("training.batch_size", 0),
        ("training.batch_size", -2),
    ])
)
def test_fuzz_out_of_range(field_and_val, tmp_path):
    path, bad_val = field_and_val
    cfg = copy.deepcopy(VALID_CONFIG)
    parts = path.split(".")
    target = cfg
    for p in parts[:-1]:
        target = target[p]
    target[parts[-1]] = bad_val

    rc, behavior, out = run_soup_cli(cfg, [], tmp_path, f"out_of_range:{path}={bad_val!r}")
    assert behavior == "CLEAR_ERROR", f"Некорректная реакция на {path}={bad_val!r}: {behavior}\n{out}"


# ---------------------------------------------------------
# 4. Empty fields (Пустые поля и пустые конфиги)
# ---------------------------------------------------------
@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    case=st.sampled_from([
        "empty_file",
        "comments_only",
        "empty_base",
        "null_base",
        "empty_train_path",
        "null_training_block",
        "empty_dict",
    ])
)
def test_fuzz_empty_fields(case, tmp_path):
    if case == "empty_file":
        payload = ""
    elif case == "comments_only":
        payload = "# only comments\n   \n"
    elif case == "empty_dict":
        payload = {}
    else:
        payload = copy.deepcopy(VALID_CONFIG)
        if case == "empty_base":
            payload["base"] = ""
        elif case == "null_base":
            payload["base"] = None
        elif case == "empty_train_path":
            payload["data"]["train"] = ""
        elif case == "null_training_block":
            payload["training"] = None

    rc, behavior, out = run_soup_cli(payload, [], tmp_path, f"empty:{case}")
    assert behavior == "CLEAR_ERROR", f"Пустой случай '{case}' вызвал {behavior}\n{out}"


# ---------------------------------------------------------
# 5. Conflicting parameters (Конфликтующие параметры CLI и YAML)
# ---------------------------------------------------------
@pytest.mark.parametrize(
    "case_name,cfg_patch,cli_args",
    [
        ("wandb_and_tracker", {}, ["--wandb", "--tracker", "mlflow"]),
        ("hf_resume_without_push_as", {}, ["--hf-resume"]),
        ("echo_trap_without_config_flag", {}, ["--echo-trap-tokenizer-aware"]),
        ("val_split_leaves_zero_train_rows", {"data": {"train": DATA_FILE, "val_split": 0.99}}, []),
        ("resume_nonexistent_dir", {}, ["--resume", "./non_existent_checkpoint_dir_xyz"]),
    ],
)
def test_conflicting_parameters(case_name, cfg_patch, cli_args, tmp_path):
    cfg = copy.deepcopy(VALID_CONFIG)
    for k, v in cfg_patch.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v

    rc, behavior, out = run_soup_cli(cfg, cli_args, tmp_path, f"conflict:{case_name}")
    assert behavior == "CLEAR_ERROR", f"Конфликт '{case_name}' обработан как {behavior}\n{out}"


# ---------------------------------------------------------
# 6. Very long values (Очень длинные строки и глубокая вложенность)
# ---------------------------------------------------------
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    long_case=st.sampled_from([
        "long_output_path_5000",
        "long_base_name_10000",
        "long_experiment_name_5000",
        "null_byte_in_string",
    ])
)
def test_fuzz_very_long_values(long_case, tmp_path):
    cfg = copy.deepcopy(VALID_CONFIG)
    cli_args = []
    if long_case == "long_output_path_5000":
        cfg["output"] = "./" + ("a" * 5000)
