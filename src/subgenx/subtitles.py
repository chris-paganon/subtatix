from __future__ import annotations

from dataclasses import dataclass
import shutil
from pathlib import Path

import whisperx

from subgenx.runtime import get_whisperx_runtime, release_memory


DEFAULT_MODEL = "large-v2"


@dataclass(frozen=True)
class SubtitleCue:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SubtitleDocument:
    source_language: str
    subtitle_path: Path
    cues: list[SubtitleCue]


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


def format_srt_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def build_cues(aligned_transcription: dict) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    for segment in aligned_transcription["segments"]:
        text = segment["text"].strip()
        if not text:
            continue
        cues.append(
            SubtitleCue(
                start=float(segment["start"]),
                end=float(segment["end"]),
                text=text,
            )
        )
    return cues


def write_srt(subtitle_path: Path, cues: list[SubtitleCue]) -> None:
    subtitle_path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for index, cue in enumerate(cues, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    (
                        f"{format_srt_timestamp(cue.start)} --> "
                        f"{format_srt_timestamp(cue.end)}"
                    ),
                    cue.text,
                ]
            )
        )
    subtitle_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def transcribe_to_srt(
    input_file: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 16,
    output_file: Path | None = None,
    write_output: bool = True,
) -> SubtitleDocument:
    input_file = input_file.expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    output_path = resolve_output_path(input_file, output_file)

    device, compute_type = get_whisperx_runtime()
    whisper_model = None
    align_model = None
    audio = None
    transcription = None
    aligned_transcription = None

    try:
        whisper_model = whisperx.load_model(
            model_name,
            device,
            compute_type=compute_type,
        )

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
        cues = build_cues(aligned_transcription)
        if write_output:
            write_srt(output_path, cues)
        return SubtitleDocument(
            source_language=language,
            subtitle_path=output_path,
            cues=cues,
        )
    finally:
        del whisper_model
        del align_model
        del audio
        del transcription
        del aligned_transcription
        release_memory()
