from __future__ import annotations

import gc

import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_whisperx_runtime() -> tuple[str, str]:
    device = get_device()
    if device == "cuda":
        return device, "float16"
    return device, "float32"


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
