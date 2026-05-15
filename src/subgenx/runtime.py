from __future__ import annotations

import gc
import logging
import warnings

import torch

from subgenx.errors import SubgenxError


def get_device(preferred: str = "auto") -> str:
    normalized = preferred.strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise SubgenxError(
            f"Unsupported device '{preferred}'. Use 'auto', 'cpu', or 'cuda'."
        )
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise SubgenxError(
                "CUDA was requested but is not available on this system."
            )
        return "cuda"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_whisperx_runtime(preferred_device: str = "auto") -> tuple[str, str]:
    device = get_device(preferred_device)
    if device == "cuda":
        return device, "float16"
    return device, "float32"


def configure_runtime_noise() -> None:
    for logger_name in (
        "whisperx",
        "whisperx.asr",
        "whisperx.vads",
        "whisperx.vads.pyannote",
        "lightning",
        "lightning.pytorch",
        "lightning.pytorch.utilities.migration",
        "lightning.pytorch.utilities.migration.utils",
        "pytorch_lightning",
        "pytorch_lightning.utilities.migration",
        "pytorch_lightning.utilities.migration.utils",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    warnings.filterwarnings(
        "ignore",
        message=r"TensorFloat-32 \(TF32\) has been disabled.*",
        module=r"pyannote\.audio\.utils\.reproducibility",
    )


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
