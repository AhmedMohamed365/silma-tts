from __future__ import annotations

import argparse
from pathlib import Path

from common import ensure_dir, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge channel manifests into one dataset manifest.")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--input-manifest-name", default="manifest_transcribed.jsonl")
    parser.add_argument("--output-manifest", default="data/merged/dataset_manifest.jsonl")
    parser.add_argument("--dedupe-by-id", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    channels_root = root / "channels"
    out_manifest = Path(args.output_manifest)
    ensure_dir(out_manifest.parent)

    all_rows: list[dict] = []
    seen_ids: set[str] = set()

    for channel_dir in sorted(channels_root.glob("*")):
        manifest_path = channel_dir / "manifests" / args.input_manifest_name
        rows = read_jsonl(manifest_path)
        for row in rows:
            if args.dedupe_by_id:
                row_id = row.get("id", "")
                if row_id in seen_ids:
                    continue
                seen_ids.add(row_id)
            all_rows.append(row)

    with out_manifest.open("w", encoding="utf-8") as f:
        for row in all_rows:
            import json

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Merged {len(all_rows)} rows into {out_manifest}")


if __name__ == "__main__":
    main()
