# Subtatix

`Subtatix` is a small CLI for generating `.srt` subtitles from audio or video files with [WhisperX](https://github.com/m-bain/whisperX), with optional subtitle translation.

It transcribes the input, aligns subtitle timings with WhisperX, and can then translate the resulting subtitle lines into another language.

## Requirements

- Python 3.12+
- `ffmpeg` installed separately and available on your `PATH`
- Enough disk space for model downloads and caching

The first run will be slower because WhisperX and translation models need to be downloaded. Subsequent runs reuse the cached models and do not need to download them again unless the cache is cleared.

`ffmpeg` is an external system dependency. It is not installed by `pip`, `uvx`, or `uv tool install`.

## Installation

Run without installing:

```bash
uvx subtatix --help
```

Install as a tool with `uv`:

```bash
uv tool install subtatix
```

Install with `pip`:

```bash
pip install subtatix
```

## Usage

Run the CLI:

```bash
subtatix input.mp4
```

Transcribe to a specific output path:

```bash
subtatix input.mp4 --output some-path/some-file-name
```

`--output` is a base path, not a full `.srt` filename. This writes `some-path/some-file-name.srt`. If you also translate to Spanish, it writes `some-path/some-file-name.es.srt`.

Set the source language explicitly:

```bash
subtatix input.mp4 --source-language en
```

Translate after transcription:

```bash
subtatix input.mp4 --to es
```

This writes both the original transcription SRT and the translated SRT by default.

If CUDA runs out of memory on larger files, reduce the batch size or force CPU mode:

```bash
subtatix input.mp4 --batch-size 4
subtatix input.mp4 --device cpu
```

To discard the original transcription and only keep the translated output:

```bash
subtatix input.mp4 --to es --discard-transcription
```

Passing an `--output` value that ends in `.srt` is rejected. Use a base path such as `--output subtitles` instead.

List supported language codes:

```bash
subtatix --list-languages
subtatix --list-target-languages
```

## Models

By default, transcription uses WhisperX with the Whisper model `large-v2`. This is a good general default when you want higher transcription quality and aligned subtitle timings, but it is heavier and slower than smaller Whisper models.

Translation uses `facebook/nllb-200-1.3B`. The CLI accepts simple target codes such as `en`, `es`, `fr`, `de`, `pt`, `ja`, `ko`, `zh`, and also raw NLLB codes such as `spa_Latn`.

Other model options can also be used:

- For transcription, you can pass another Whisper model with `--model`, such as `small`, `medium`, or `large-v3`, depending on your speed and accuracy needs.
- For translation, the code currently defaults to the NLLB model above, but the translation layer is built around Hugging Face seq2seq models and could be adapted to use a different multilingual translation model if needed.
