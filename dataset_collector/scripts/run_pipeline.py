from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full dataset collector pipeline from YAML config.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    scripts_dir = Path(__file__).resolve().parent
    channels = cfg.get("channels", [])
    output_root = cfg.get("output_root", "data")

    for ch in channels:
        channel_url = ch["channel_url"]
        channel_slug = ch["channel_slug"]

        run(
            [
                "python",
                str(scripts_dir / "download_channel_audio.py"),
                "--channel-url",
                channel_url,
                "--channel-slug",
                channel_slug,
                "--output-root",
                output_root,
                "--max-videos",
                str(ch.get("max_videos", 0)),
            ]
        )

        chunk_cfg = cfg.get("chunking", {})
        chunk_cmd = [
            "python",
            str(scripts_dir / "process_audio_chunks.py"),
            "--channel-slug",
            channel_slug,
            "--output-root",
            output_root,
            "--min-chunk-sec",
            str(chunk_cfg.get("min_chunk_sec", 2)),
            "--max-chunk-sec",
            str(chunk_cfg.get("max_chunk_sec", 20)),
            "--chunks-per-file",
            str(chunk_cfg.get("chunks_per_file", 80)),
        ]
        if chunk_cfg.get("enable_audio_filter", False):
            chunk_cmd += [
                "--enable-audio-filter",
                "--audio-filter-model",
                chunk_cfg.get("audio_filter_model", "MIT/ast-finetuned-audioset-10-10-0.4593"),
                "--speech-threshold",
                str(chunk_cfg.get("speech_threshold", 0.45)),
            ]
        run(chunk_cmd)

        asr_cfg = cfg.get("asr", {})
        run(
            [
                "python",
                str(scripts_dir / "transcribe_audio.py"),
                "--channel-slug",
                channel_slug,
                "--output-root",
                output_root,
                "--model-id",
                asr_cfg.get("model_id", "facebook/omniASR-LLM-7B"),
                "--language",
                asr_cfg.get("language", "ar"),
                "--split",
                asr_cfg.get("split", "train"),
            ]
        )

    merge_cfg = cfg.get("merge", {})
    run(
        [
            "python",
            str(scripts_dir / "merge_manifests.py"),
            "--output-root",
            output_root,
            "--input-manifest-name",
            merge_cfg.get("input_manifest_name", "manifest_transcribed.jsonl"),
            "--output-manifest",
            merge_cfg.get("output_manifest", "data/merged/dataset_manifest.jsonl"),
            "--dedupe-by-id",
        ]
    )


if __name__ == "__main__":
    main()
