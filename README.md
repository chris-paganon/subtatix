# subgenx

`subgenx` is a small CLI for generating `.srt` subtitles from audio or video files with [WhisperX](https://github.com/m-bain/whisperX), with optional subtitle translation.

It transcribes the input, aligns subtitle timings with WhisperX, and can then translate the resulting subtitle lines into another language.

## Requirements

- Python 3.12+
- `ffmpeg` available on your `PATH`
- Enough disk space for model downloads and caching
- Internet access on first run to download the required models

The first run will be slower because WhisperX and translation models need to be downloaded. Subsequent runs reuse the cached models and do not need to download them again unless the cache is cleared.

## Usage

Install project dependencies:

```bash
uv sync
```

Run the CLI:

```bash
uv run subgenx input.mp4
```

Transcribe to a specific output path:

```bash
uv run subgenx input.mp4 --output subtitles.srt
```

Set the source language explicitly:

```bash
uv run subgenx input.mp4 --source-language en
```

Translate after transcription:

```bash
uv run subgenx input.mp4 --to es
```

If CUDA runs out of memory on larger files, reduce the batch size or force CPU mode:

```bash
uv run subgenx input.mp4 --batch-size 4
uv run subgenx input.mp4 --device cpu
```

Keep both the original and translated subtitle files:

```bash
uv run subgenx input.mp4 --to es --save-intermediary-srt
```

List supported language codes:

```bash
uv run subgenx --list-languages
uv run subgenx --list-target-languages
```

## Models

By default, transcription uses WhisperX with the Whisper model `large-v2`. This is a good general default when you want higher transcription quality and aligned subtitle timings, but it is heavier and slower than smaller Whisper models.

Translation uses `facebook/nllb-200-1.3B`. The CLI accepts simple target codes such as `en`, `es`, `fr`, `de`, `pt`, `ja`, `ko`, `zh`, and also raw NLLB codes such as `spa_Latn`.

Other model options can also be used:

- For transcription, you can pass another Whisper model with `--model`, such as `small`, `medium`, or `large-v3`, depending on your speed and accuracy needs.
- For translation, the code currently defaults to the NLLB model above, but the translation layer is built around Hugging Face seq2seq models and could be adapted to use a different multilingual translation model if needed.
