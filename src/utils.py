"""
utils.py — Shared utilities for the Lantana ML pipeline.
Handles config loading, logging setup, file hashing, and reproducibility seeding.
"""

import hashlib
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "C:\\Users\\rudsi\\Desktop\\Expo-2026\\config\\config.yaml") -> dict:
    """Load and return the pipeline configuration as a dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def hash_file(filepath: str, algorithm: str = "sha256") -> str:
    """Return the hex digest of a file's contents."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_string(s: str, algorithm: str = "sha256") -> str:
    """Return the hex digest of a string."""
    h = hashlib.new(algorithm)
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def hash_dict(d: dict, algorithm: str = "sha256") -> str:
    """Deterministically hash a dict by serialising it to sorted JSON."""
    serialised = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hash_string(serialised, algorithm)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(module_name: str, log_dir: str = "logs", debug: bool = False) -> logging.Logger:
    """
    Return a logger that writes to both stdout and a timestamped log file.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"{module_name}_{timestamp}.log"

    level = logging.DEBUG if debug else logging.INFO

    logger = logging.getLogger(module_name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Module wrapper: log start/end + hashes
# ---------------------------------------------------------------------------

class ModuleRunner:
    """
    Context manager that logs module start/end, config hash, and
    input/output file hashes for full auditability.
    """

    def __init__(
        self,
        module_name: str,
        config: dict,
        input_files: list[str] = None,
        output_files: list[str] = None,
        log_dir: str = "logs",
        debug: bool = False,
    ):
        self.module_name = module_name
        self.config = config
        self.input_files = input_files or []
        self.output_files = output_files or []
        self.logger = get_logger(module_name, log_dir, debug)
        self._start = None

    def __enter__(self):
        self._start = time.time()
        self.logger.info(f"=== MODULE START: {self.module_name} ===")
        self.logger.info(f"Config hash: {hash_dict(self.config)}")
        for f in self.input_files:
            if Path(f).exists():
                self.logger.info(f"Input  [{f}] hash: {hash_file(f)}")
            else:
                self.logger.warning(f"Input  [{f}] does not exist yet")
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(f"MODULE FAILED: {exc_type.__name__}: {exc_val}")
            return False  # re-raise
        for f in self.output_files:
            if Path(f).exists():
                self.logger.info(f"Output [{f}] hash: {hash_file(f)}")
            else:
                self.logger.warning(f"Output [{f}] was not produced")
        elapsed = time.time() - self._start
        self.logger.info(f"=== MODULE COMPLETE: {self.module_name} ({elapsed:.1f}s) ===")
        return False


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class InsufficientDataError(ValueError):
    """Raised when the dataset has too few rows for reliable modelling."""


class DataLeakageError(RuntimeError):
    """Raised if experimental data appears in any training code path."""


class PredictionTamperingError(RuntimeError):
    """Raised if the locked prediction file has been modified post-locking."""


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def ensure_dirs(*paths: str) -> None:
    """Create directories if they do not exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    """Save a JSON-serialisable object to path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Any:
    """Load a JSON file and return the parsed object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
