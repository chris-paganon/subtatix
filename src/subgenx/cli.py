from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from subgenx.subtitles import DEFAULT_MODEL, require_ffmpeg, transcribe_to_srt
from subgenx.translation import translate_subtitles

app = typer.Typer(
    add_completion=False,
    help="Transcribe an audio or video file to SRT with WhisperX and optionally translate it.",
)


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
            "--target-language",
            "-t",
            help="Target language for a translated subtitle file, for example 'spanish' or 'es'.",
        ),
    ] = None,
    save_intermediary_srt: Annotated[
        bool,
        typer.Option(
            "--save-intermediary-srt",
            help="When translating, also save the original untranslated SRT file.",
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
