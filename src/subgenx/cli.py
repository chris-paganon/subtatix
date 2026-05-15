from __future__ import annotations

import argparse
from pathlib import Path

from subgenx.subtitles import DEFAULT_MODEL, require_ffmpeg, transcribe_to_srt


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
