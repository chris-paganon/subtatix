from __future__ import annotations

import argparse
from pathlib import Path

import torch
import whisperx
from whisperx.utils import get_writer


DEFAULT_MODEL = "large-v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subgenx",
        description="Transcribe an audio file to SRT with WhisperX.",
    )
    parser.add_argument("audio_file", type=Path, help="Path to the input audio file.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Whisper model name to use. Default: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for Whisper inference. Reduce this if you run out of GPU memory.",
    )
    return parser


def detect_runtime() -> tuple[str, str]:
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "float32"


def transcribe_to_srt(audio_file: Path, model_name: str, batch_size: int) -> Path:
    audio_file = audio_file.expanduser().resolve()
    if not audio_file.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    device, compute_type = detect_runtime()
    model = whisperx.load_model(model_name, device, compute_type=compute_type)

    audio = whisperx.load_audio(str(audio_file))
    result = model.transcribe(audio, batch_size=batch_size)

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device,
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    writer = get_writer("srt", str(audio_file.parent))
    writer(
        result,
        str(audio_file),
        {
            "highlight_words": False,
            "max_line_count": None,
            "max_line_width": None,
        },
    )
    return audio_file.with_suffix(".srt")


def main() -> None:
    args = build_parser().parse_args()
    output_path = transcribe_to_srt(
        audio_file=args.audio_file,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    print(output_path)
