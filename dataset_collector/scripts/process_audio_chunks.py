from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from transformers import pipeline

from common import append_jsonl, ensure_dir, read_jsonl, slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk long audio files into random short segments.")
    parser.add_argument("--channel-slug", required=True)
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--min-chunk-sec", type=float, default=2.0)
    parser.add_argument("--max-chunk-sec", type=float, default=20.0)
    parser.add_argument("--chunks-per-file", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-audio-filter", action="store_true")
    parser.add_argument("--audio-filter-model", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--speech-threshold", type=float, default=0.45)
    return parser.parse_args()


def is_speech_like(classifier, audio: np.ndarray, sample_rate: int, threshold: float) -> bool:
    if classifier is None:
        return True
    labels = classifier({"array": audio, "sampling_rate": sample_rate}, top_k=5)
    speech_score = 0.0
    music_score = 0.0
    for pred in labels:
        label = pred["label"].lower()
        score = float(pred["score"])
        if "speech" in label or "conversation" in label:
            speech_score = max(speech_score, score)
        if "music" in label or "singing" in label:
            music_score = max(music_score, score)
    return speech_score >= threshold and speech_score >= music_score


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    channel_slug = slugify(args.channel_slug)
    root = Path(args.output_root)
    channel_root = root / "channels" / channel_slug
    raw_manifest = channel_root / "manifests" / "manifest_raw.jsonl"
    chunk_manifest = channel_root / "manifests" / "manifest_chunks.jsonl"
    chunks_dir = ensure_dir(channel_root / "processed" / "chunks")
    runs_dir = ensure_dir(channel_root / "runs")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(runs_dir / run_id)

    rows = read_jsonl(raw_manifest)
    if not rows:
        raise SystemExit(f"No raw manifest found at: {raw_manifest}")

    classifier = None
    if args.enable_audio_filter:
        classifier = pipeline("audio-classification", model=args.audio_filter_model)

    out_rows = []
    for row in tqdm(rows, desc="chunking"):
        src_path = Path(row["audio_path"])
        if not src_path.exists():
            continue

        audio, sr = librosa.load(src_path, sr=args.sample_rate, mono=True)
        total_sec = len(audio) / args.sample_rate
        if total_sec < args.min_chunk_sec:
            continue

        for i in range(args.chunks_per_file):
            chunk_sec = random.uniform(args.min_chunk_sec, args.max_chunk_sec)
            if chunk_sec >= total_sec:
                start_sec = 0.0
                end_sec = total_sec
            else:
                start_sec = random.uniform(0, total_sec - chunk_sec)
                end_sec = start_sec + chunk_sec

            s = int(start_sec * args.sample_rate)
            e = int(end_sec * args.sample_rate)
            chunk = audio[s:e]
            if len(chunk) < int(args.min_chunk_sec * args.sample_rate):
                continue

            rms = float(np.sqrt(np.mean(np.square(chunk))))
            if rms < 0.004:
                continue

            if not is_speech_like(classifier, chunk, args.sample_rate, args.speech_threshold):
                continue

            chunk_id = f"{channel_slug}_{row['video_id']}_{i:06d}"
            out_path = chunks_dir / f"{chunk_id}.wav"
            sf.write(out_path, chunk, args.sample_rate)

            out_rows.append(
                {
                    "id": chunk_id,
                    "channel_slug": channel_slug,
                    "video_id": row["video_id"],
                    "source_url": row.get("source_url", ""),
                    "audio_path": str(out_path),
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "duration_sec": round((e - s) / args.sample_rate, 3),
                    "sample_rate": args.sample_rate,
                }
            )

    append_jsonl(chunk_manifest, out_rows)

    run_cfg = {
        "run_id": run_id,
        "stage": "chunk",
        "channel_slug": channel_slug,
        "output_root": str(root),
        "min_chunk_sec": args.min_chunk_sec,
        "max_chunk_sec": args.max_chunk_sec,
        "chunks_per_file": args.chunks_per_file,
        "enable_audio_filter": args.enable_audio_filter,
        "audio_filter_model": args.audio_filter_model,
        "speech_threshold": args.speech_threshold,
        "chunks_written": len(out_rows),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(out_rows)} chunks to {chunks_dir}")
    print(f"Chunk manifest: {chunk_manifest}")


if __name__ == "__main__":
    main()
