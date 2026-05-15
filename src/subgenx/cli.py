from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import whisperx
from whisperx.utils import get_writer


DEFAULT_MODEL = "large-v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subgenx",
        description="Transcribe an audio or video file to SRT with WhisperX.",
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input audio or video file.",
    )
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
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output SRT path. If a directory is provided, the default SRT filename is used inside it.",
    )
    return parser


def detect_runtime() -> tuple[str, str]:
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "float32"


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required but was not found on PATH. Install ffmpeg and try again."
        )


def resolve_output_path(input_file: Path, output_file: Path | None) -> Path:
    default_output_path = input_file.with_suffix(".srt")
    if output_file is None:
        return default_output_path

    output_file = output_file.expanduser()
    if output_file.exists() and output_file.is_dir():
        return (output_file / default_output_path.name).resolve()
    if output_file.suffix:
        return output_file.resolve()
    return output_file.with_suffix(".srt").resolve()


def transcribe_to_srt(
    input_file: Path,
    model_name: str,
    batch_size: int,
    output_file: Path | None = None,
) -> Path:
    input_file = input_file.expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    output_path = resolve_output_path(input_file, output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device, compute_type = detect_runtime()
    whisper_model = whisperx.load_model(model_name, device, compute_type=compute_type)

    audio = whisperx.load_audio(str(input_file))
    transcription = whisper_model.transcribe(audio, batch_size=batch_size)
    language = transcription["language"]

    align_model, align_metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
    )
    aligned_transcription = whisperx.align(
        transcription["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    aligned_transcription["language"] = language

    temporary_output_path = input_file.with_suffix(".srt")
    writer = get_writer("srt", str(input_file.parent))
    writer(
        aligned_transcription,
        str(input_file),
        {
            "highlight_words": False,
            "max_line_count": None,
            "max_line_width": None,
        },
    )
    if temporary_output_path != output_path:
        temporary_output_path.replace(output_path)
    return output_path


def main() -> None:
    require_ffmpeg()
    args = build_parser().parse_args()
    output_path = transcribe_to_srt(
        input_file=args.input_file,
        model_name=args.model,
        batch_size=args.batch_size,
        output_file=args.output,
    )
    print(output_path)
