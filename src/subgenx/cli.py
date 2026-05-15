from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from subgenx.subtitles import (
    DEFAULT_MODEL,
    require_ffmpeg,
    transcribe_to_srt,
)
from subgenx.translation import SUPPORTED_TARGET_LANGUAGE_CODES, translate_subtitles

app = typer.Typer(
    add_completion=False,
    help=(
        "Transcribe an audio or video file to SRT with WhisperX. "
        "Without --to, the tool only transcribes. Passing --to also translates the "
        "subtitles; add --save-intermediary-srt to keep the original transcribed SRT."
    ),
)

TARGET_LANGUAGE_CODES_HELP = ", ".join(SUPPORTED_TARGET_LANGUAGE_CODES)


@app.command()
def run(
    input_file: Annotated[
        Path,
        typer.Argument(help="Path to the input audio or video file."),
    ],
    model: Annotated[
        str,
        typer.Option(help=f"Whisper model name to use. Default: {DEFAULT_MODEL}."),
    ] = DEFAULT_MODEL,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Batch size for Whisper inference. Reduce this if you run out of GPU memory.",
        ),
    ] = 16,
    source_language: Annotated[
        str | None,
        typer.Option(
            "--source-language",
            "-s",
            help=(
                "Optional Whisper source language code to skip language detection, "
                "for example 'en', 'es', or 'fr'."
            ),
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output SRT path. If a directory is provided, the default SRT filename is used inside it.",
        ),
    ] = None,
    target_language: Annotated[
        str | None,
        typer.Option(
            "--to",
            "--target-language",
            "-t",
            help=(
                "Translate to this language. Use one of the mapped Whisper language "
                f"codes ({TARGET_LANGUAGE_CODES_HELP}) or a raw NLLB code like "
                "'spa_Latn'. If omitted, the tool only transcribes."
            ),
        ),
    ] = None,
    save_intermediary_srt: Annotated[
        bool,
        typer.Option(
            "--save-intermediary-srt",
            help=(
                "When used with --to, also save the original untranslated SRT file. "
                "Without --to, the transcribed SRT is already written."
            ),
            is_flag=True,
        ),
    ] = False,
) -> None:
    require_ffmpeg()
    write_original_srt = target_language is None or save_intermediary_srt
    document = transcribe_to_srt(
        input_file=input_file,
        model_name=model,
        batch_size=batch_size,
        output_file=output,
        write_output=write_original_srt,
        source_language=source_language,
    )
    if write_original_srt:
        typer.echo(document.subtitle_path)

    if target_language:
        translated_path = translate_subtitles(
            document=document,
            target_language=target_language,
        )
        typer.echo(translated_path)


def main() -> None:
    app()
