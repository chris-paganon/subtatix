from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from subgenx.runtime import get_device, release_memory
from subgenx.subtitles import SubtitleCue, SubtitleDocument, write_srt


DEFAULT_TRANSLATION_MODEL = "facebook/nllb-200-1.3B"
DEFAULT_TRANSLATION_BATCH_SIZE = 16


@dataclass(frozen=True)
class LanguageSpec:
    nllb_code: str
    suffix: str


LANGUAGE_SPECS = {
    "ca": LanguageSpec("cat_Latn", "ca"),
    "de": LanguageSpec("deu_Latn", "de"),
    "en": LanguageSpec("eng_Latn", "en"),
    "es": LanguageSpec("spa_Latn", "es"),
    "fr": LanguageSpec("fra_Latn", "fr"),
    "it": LanguageSpec("ita_Latn", "it"),
    "ja": LanguageSpec("jpn_Jpan", "ja"),
    "ko": LanguageSpec("kor_Hang", "ko"),
    "nl": LanguageSpec("nld_Latn", "nl"),
    "pt": LanguageSpec("por_Latn", "pt"),
    "ru": LanguageSpec("rus_Cyrl", "ru"),
    "zh": LanguageSpec("zho_Hans", "zh"),
}
SUPPORTED_TARGET_LANGUAGE_CODES = tuple(sorted(LANGUAGE_SPECS))

_TRANSLATION_MODELS: dict[str, tuple[object, object, str]] = {}
_NLLB_LANGUAGE_CODES: tuple[str, ...] | None = None


def resolve_language(language: str) -> LanguageSpec:
    raw_language = language.strip()
    key = raw_language.lower()
    if key in LANGUAGE_SPECS:
        return LANGUAGE_SPECS[key]

    parts = raw_language.split("_", 1)
    if len(parts) == 2 and len(parts[0]) == 3 and parts[0].islower():
        return LanguageSpec(raw_language, parts[0])

    raise ValueError(
        f"Unsupported language '{language}'. Use a Whisper language code like 'es' "
        "or a full NLLB language code like 'spa_Latn'."
    )


def resolve_translation_output_path(
    subtitle_path: Path,
    target_language: LanguageSpec,
) -> Path:
    if subtitle_path.suffix != ".srt":
        raise ValueError(f"Expected an .srt subtitle file, got: {subtitle_path}")
    return subtitle_path.with_name(
        f"{subtitle_path.stem}.{target_language.suffix}{subtitle_path.suffix}"
    )


def get_translation_backend(
    model_name: str = DEFAULT_TRANSLATION_MODEL,
) -> tuple[object, object, str]:
    if model_name not in _TRANSLATION_MODELS:
        release_memory()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = get_device()
        _TRANSLATION_MODELS[model_name] = (tokenizer, model.to(device), device)
    return _TRANSLATION_MODELS[model_name]


def get_available_nllb_languages(
    model_name: str = DEFAULT_TRANSLATION_MODEL,
) -> tuple[str, ...]:
    global _NLLB_LANGUAGE_CODES
    if _NLLB_LANGUAGE_CODES is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        _NLLB_LANGUAGE_CODES = tuple(sorted(tokenizer.additional_special_tokens))
    return _NLLB_LANGUAGE_CODES


def translate_batch(
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    model_name: str = DEFAULT_TRANSLATION_MODEL,
    max_length: int = 400,
) -> list[str]:
    tokenizer, model, device = get_translation_backend(model_name)
    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_length=max_length,
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)


def translate_subtitles(
    document: SubtitleDocument,
    target_language: str,
    model_name: str = DEFAULT_TRANSLATION_MODEL,
    batch_size: int = DEFAULT_TRANSLATION_BATCH_SIZE,
    max_length: int = 400,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    source = resolve_language(document.source_language)
    target = resolve_language(target_language)
    output_path = resolve_translation_output_path(document.subtitle_path, target)
    cues = document.cues
    total_batches = max(1, math.ceil(len(cues) / batch_size))

    translated_texts: list[str] = []
    for batch_index, start in enumerate(range(0, len(cues), batch_size), start=1):
        batch = cues[start : start + batch_size]
        translated_texts.extend(
            translate_batch(
                [cue.text for cue in batch],
                src_lang=source.nllb_code,
                tgt_lang=target.nllb_code,
                model_name=model_name,
                max_length=max_length,
            )
        )
        if progress_callback is not None:
            progress_callback(batch_index, total_batches)

    translated_cues = [
        SubtitleCue(start=cue.start, end=cue.end, text=text)
        for cue, text in zip(cues, translated_texts, strict=True)
    ]
    write_srt(output_path, translated_cues)
    return output_path
