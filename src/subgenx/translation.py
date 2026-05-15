from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from subgenx.runtime import get_device, release_memory


DEFAULT_TRANSLATION_MODEL = "facebook/nllb-200-1.3B"


@dataclass(frozen=True)
class LanguageSpec:
    nllb_code: str
    suffix: str


@dataclass(frozen=True)
class SubtitleCue:
    index: str
    timing: str
    text: str


LANGUAGE_SPECS = {
    "ca": LanguageSpec("cat_Latn", "ca"),
    "catalan": LanguageSpec("cat_Latn", "ca"),
    "de": LanguageSpec("deu_Latn", "de"),
    "german": LanguageSpec("deu_Latn", "de"),
    "en": LanguageSpec("eng_Latn", "en"),
    "english": LanguageSpec("eng_Latn", "en"),
    "es": LanguageSpec("spa_Latn", "es"),
    "spanish": LanguageSpec("spa_Latn", "es"),
    "fr": LanguageSpec("fra_Latn", "fr"),
    "french": LanguageSpec("fra_Latn", "fr"),
    "it": LanguageSpec("ita_Latn", "it"),
    "italian": LanguageSpec("ita_Latn", "it"),
    "ja": LanguageSpec("jpn_Jpan", "ja"),
    "japanese": LanguageSpec("jpn_Jpan", "ja"),
    "ko": LanguageSpec("kor_Hang", "ko"),
    "korean": LanguageSpec("kor_Hang", "ko"),
    "nl": LanguageSpec("nld_Latn", "nl"),
    "dutch": LanguageSpec("nld_Latn", "nl"),
    "pt": LanguageSpec("por_Latn", "pt"),
    "portuguese": LanguageSpec("por_Latn", "pt"),
    "ru": LanguageSpec("rus_Cyrl", "ru"),
    "russian": LanguageSpec("rus_Cyrl", "ru"),
    "zh": LanguageSpec("zho_Hans", "zh"),
    "chinese": LanguageSpec("zho_Hans", "zh"),
}

_TRANSLATION_MODELS: dict[str, tuple[object, object, str]] = {}


def resolve_language(language: str) -> LanguageSpec:
    key = language.strip().lower()
    if key in LANGUAGE_SPECS:
        return LANGUAGE_SPECS[key]

    if re.fullmatch(r"[a-z]{3}_[A-Za-z][a-z]{3}", language):
        return LanguageSpec(language, key.split("_", 1)[0])

    raise ValueError(
        f"Unsupported language '{language}'. Use a supported alias like 'es' or "
        "'spanish', or pass a full NLLB language code like 'spa_Latn'."
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


def parse_srt(subtitle_path: Path) -> list[SubtitleCue]:
    content = subtitle_path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    cues: list[SubtitleCue] = []
    for block in re.split(r"\n\s*\n", content):
        lines = block.splitlines()
        if len(lines) < 3:
            raise ValueError(f"Malformed SRT block in {subtitle_path}: {block!r}")
        cues.append(
            SubtitleCue(
                index=lines[0],
                timing=lines[1],
                text="\n".join(lines[2:]),
            )
        )
    return cues


def write_srt(subtitle_path: Path, cues: list[SubtitleCue]) -> None:
    blocks = [f"{cue.index}\n{cue.timing}\n{cue.text}" for cue in cues]
    subtitle_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


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
    subtitle_path: Path,
    source_language: str,
    target_language: str,
    model_name: str = DEFAULT_TRANSLATION_MODEL,
    batch_size: int = 16,
    max_length: int = 400,
) -> Path:
    subtitle_path = subtitle_path.expanduser().resolve()
    if not subtitle_path.is_file():
        raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")

    source = resolve_language(source_language)
    target = resolve_language(target_language)
    output_path = resolve_translation_output_path(subtitle_path, target)
    cues = parse_srt(subtitle_path)

    translated_texts: list[str] = []
    for start in range(0, len(cues), batch_size):
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

    translated_cues = [
        SubtitleCue(index=cue.index, timing=cue.timing, text=text)
        for cue, text in zip(cues, translated_texts, strict=True)
    ]
    write_srt(output_path, translated_cues)
    return output_path
