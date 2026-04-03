# dataset_collector

Professional data collection toolkit for building Arabic (including Saudi dialect) ASR/TTS datasets from YouTube channels.

This toolkit is optimized for:
- **Fast channel audio ingestion** via `yt-dlp`.
- **Configurable speech-focused filtering** (optional audio classifier stage).
- **Random chunking** of long audio into short segments (2–20 seconds by default).
- **ASR transcription** with modern Arabic-capable models (default: `facebook/omniASR-LLM-7B`).
- **Scalable multi-run / multi-channel dataset organization**.
- **Dataset-level manifest merging** for immediate SILMA training usage.

---

## Folder structure

```text
dataset_collector/
  scripts/
    download_channel_audio.py
    process_audio_chunks.py
    transcribe_audio.py
    merge_manifests.py
    run_pipeline.py
  configs/
    pipeline.example.yaml
```

Generated data layout:

```text
data/
  channels/
    <channel_slug>/
      raw_audio/
      processed/
        chunks/
        transcripts/
      manifests/
        manifest_raw.jsonl
        manifest_chunks.jsonl
        manifest_transcribed.jsonl
      runs/
        <YYYYmmdd_HHMMSS>/
          run_config.json
  merged/
    dataset_manifest.jsonl
```

---

## Requirements

- Python 3.10+
- `ffmpeg` installed and available in PATH
- Python packages:

```bash
pip install yt-dlp librosa soundfile pyyaml transformers torchaudio tqdm numpy
```

Optional (recommended for faster inference):

```bash
pip install accelerate bitsandbytes
```

---

## 1) Download a channel as audio-only

```bash
python dataset_collector/scripts/download_channel_audio.py \
  --channel-url "https://www.youtube.com/@YourChannel/videos" \
  --channel-slug "your_channel" \
  --output-root data \
  --max-videos 100
```

### Notes
- Uses best audio stream + converts to WAV (16k mono by default).
- Stores full per-video metadata in `manifest_raw.jsonl`.
- Safe for multiple runs and incremental data growth.

---

## 2) Process long audio into random chunks (2–20s)

```bash
python dataset_collector/scripts/process_audio_chunks.py \
  --channel-slug "your_channel" \
  --output-root data \
  --min-chunk-sec 2 \
  --max-chunk-sec 20 \
  --chunks-per-file 120
```

### Optional speech-vs-music filtering

```bash
python dataset_collector/scripts/process_audio_chunks.py \
  --channel-slug "your_channel" \
  --output-root data \
  --enable-audio-filter \
  --audio-filter-model "MIT/ast-finetuned-audioset-10-10-0.4593" \
  --speech-threshold 0.45
```

> Tip: if you have a preferred speech/music separator (e.g., SAM-style audio model), replace the classifier model using `--audio-filter-model` and adjust threshold.

---

## 3) Transcribe chunks with omniASR

```bash
python dataset_collector/scripts/transcribe_audio.py \
  --channel-slug "your_channel" \
  --output-root data \
  --model-id "facebook/omniASR-LLM-7B" \
  --language ar
```

Outputs:
- `manifest_transcribed.jsonl`
- `transcripts/*.txt`

---

## 4) Merge all channel manifests into one dataset

```bash
python dataset_collector/scripts/merge_manifests.py \
  --output-root data \
  --input-manifest-name manifest_transcribed.jsonl \
  --output-manifest data/merged/dataset_manifest.jsonl
```

---

## 5) Full pipeline in one command

```bash
python dataset_collector/scripts/run_pipeline.py --config dataset_collector/configs/pipeline.example.yaml
```

---

## Manifest schema (training-ready)

Each line in `manifest_transcribed.jsonl` contains:

```json
{
  "id": "your_channel_abc123_000042",
  "audio_path": "data/channels/your_channel/processed/chunks/your_channel_abc123_000042.wav",
  "text": "...",
  "duration_sec": 6.31,
  "sample_rate": 16000,
  "channel_slug": "your_channel",
  "video_id": "abc123",
  "source_url": "https://www.youtube.com/watch?v=abc123",
  "language": "ar",
  "split": "train"
}
```

This format is ideal for downstream SILMA data loaders.
