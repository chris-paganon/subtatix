from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from subgenx.subtitles import (
    DEFAULT_MODEL,
    SUPPORTED_SOURCE_LANGUAGE_CODES,
    require_ffmpeg,
    transcribe_to_srt,
)
from subgenx.translation import (
    DEFAULT_TRANSLATION_BATCH_SIZE,
    SUPPORTED_TARGET_LANGUAGE_CODES,
    get_available_nllb_languages,
    translate_subtitles,
)

app = typer.Typer(
    add_completion=False,
    help=(
        "Transcribe an audio or video file to SRT with WhisperX. "
        "Without --to, the tool only transcribes. Passing --to also translates the "
        "subtitles; add --save-intermediary-srt to keep the original transcribed SRT. "
        "Use the language listing flags to inspect supported source, mapped target, "
        "and raw NLLB codes."
    ),
)


class ProgressBar:
    def __init__(self, label: str, total: int, width: int = 28) -> None:
        self._label = label
        self._total = max(1, total)
        self._width = width
        self._current = 0

    def __enter__(self) -> "ProgressBar":
        self._render()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None and self._current < self._total:
            self._current = self._total
            self._render()
        sys.stderr.write("\n")
        sys.stderr.flush()

    def update_to(self, value: int) -> None:
        bounded = min(self._total, max(0, value))
        if bounded != self._current:
            self._current = bounded
            self._render()

    def reset(self) -> None:
        if self._current != 0:
            self._current = 0
            self._render()

    def _render(self) -> None:
        filled = round((self._current / self._total) * self._width)
        empty = self._width - filled
        percent = round((self._current / self._total) * 100)
        sys.stderr.write(
            f"\r{self._label} [{'#' * filled}{'.' * empty}] {percent:>3}%"
        )
        sys.stderr.flush()


@app.command()
def run(
    input_file: Annotated[
        Path | None,
        typer.Argument(help="Path to the input audio or video file."),
    ] = None,
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
    ] = 8,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help=(
                "Execution device for WhisperX: 'auto', 'cuda', or 'cpu'. "
                "Default: auto."
            ),
        ),
    ] = "auto",
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
                "Translate to this language. Use one of the mapped Whisper target "
                "codes or a raw NLLB code like 'spa_Latn'. If omitted, the tool only "
                "transcribes. Use --list-target-languages to inspect supported values."
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
    list_languages: Annotated[
        bool,
        typer.Option(
            "--list-languages",
            help=(
                "List source Whisper codes and the convenience target codes. "
                "Use --list-target-languages for the full raw NLLB target list."
            ),
            is_flag=True,
        ),
    ] = False,
    list_source_languages: Annotated[
        bool,
        typer.Option(
            "--list-source-languages",
            help="List supported Whisper source language codes.",
            is_flag=True,
        ),
    ] = False,
    list_target_languages: Annotated[
        bool,
        typer.Option(
            "--list-target-languages",
            help=(
                "List the convenience target codes for --to, followed by the full raw "
                "NLLB target language codes accepted by --to."
            ),
            is_flag=True,
        ),
    ] = False,
) -> None:
    if list_languages or list_source_languages or list_target_languages:
        if input_file is not None:
            raise typer.BadParameter(
                "INPUT_FILE cannot be used with language listing options."
            )
        if list_languages or list_source_languages:
            typer.echo("Source languages (Whisper codes):")
            typer.echo(", ".join(SUPPORTED_SOURCE_LANGUAGE_CODES))
            typer.echo()
        if list_languages or list_target_languages:
            typer.echo(
                "Convenience target languages (--to Whisper-style codes mapped to NLLB):"
            )
            typer.echo(", ".join(SUPPORTED_TARGET_LANGUAGE_CODES))
            typer.echo()
        if list_languages:
            typer.echo(
                "Use --list-target-languages to also show the full raw NLLB target list."
            )
        if list_target_languages:
            typer.echo("Raw NLLB target languages (--to NLLB codes):")
            typer.echo(", ".join(get_available_nllb_languages()))
        return

    if input_file is None:
        raise typer.BadParameter("Missing argument 'INPUT_FILE'.")

    require_ffmpeg()
    write_original_srt = target_language is None or save_intermediary_srt
    with ProgressBar("Transcription", 100) as transcription_progress:
        document = transcribe_to_srt(
            input_file=input_file,
            model_name=model,
            batch_size=batch_size,
            output_file=output,
            write_output=write_original_srt,
            source_language=source_language,
            device_preference=device,
            log=lambda message: typer.echo(message, err=True),
            progress_callback=lambda percent: transcription_progress.update_to(
                round(percent)
            ),
            progress_reset=transcription_progress.reset,
        )
    if write_original_srt:
        typer.echo(document.subtitle_path)

    if target_language:
        total_batches = max(
            1,
            (len(document.cues) + DEFAULT_TRANSLATION_BATCH_SIZE - 1)
            // DEFAULT_TRANSLATION_BATCH_SIZE,
        )
        with ProgressBar("Translation", total_batches) as translation_progress:
            translated_path = translate_subtitles(
                document=document,
                target_language=target_language,
                progress_callback=lambda batch_index, total: translation_progress.update_to(
                    batch_index
                ),
            )
        typer.echo(translated_path)


def main() -> None:
    app()
