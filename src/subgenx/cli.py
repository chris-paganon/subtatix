from __future__ import annotations

import argparse
from pathlib import Path

from subgenx.subtitles import DEFAULT_MODEL, require_ffmpeg, transcribe_to_srt
from subgenx.translation import translate_subtitles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subgenx",
        description="Transcribe an audio or video file to SRT with WhisperX and optionally translate it.",
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
    parser.add_argument(
        "-t",
        "--target-language",
        help="Target language for a translated subtitle file, for example 'spanish' or 'es'.",
    )
    parser.add_argument(
        "--save-intermediary-srt",
        action="store_true",
        help="When translating, also save the original untranslated SRT file.",
    )
    return parser


def main() -> None:
    require_ffmpeg()
    args = build_parser().parse_args()
    write_original_srt = args.target_language is None or args.save_intermediary_srt
    document = transcribe_to_srt(
        input_file=args.input_file,
        model_name=args.model,
        batch_size=args.batch_size,
        output_file=args.output,
        write_output=write_original_srt,
    )
    if write_original_srt:
        print(document.subtitle_path)

    if args.target_language:
        translated_path = translate_subtitles(
            document=document,
            target_language=args.target_language,
        )
        print(translated_path)
