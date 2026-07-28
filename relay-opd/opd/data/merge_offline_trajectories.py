#!/usr/bin/env python3
"""Merge trajectory-generation shards and materialize training parquet."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd


TEACHER_COLUMNS = [
    "messages",
    "prompt_token_ids",
    "response_token_ids",
    "sample_idx",
    "unique_id",
    "gt",
    "enable_thinking",
    "eos_reached",
    "finish_reason",
    "gen_len",
    "data_source",
]

TRD_COLUMNS = [
    "messages",
    "rewrite_messages",
    "prompt_token_ids",
    "teacher_prompt_token_ids",
    "response_token_ids",
    "teacher_response_token_ids",
    "y_o_response_token_ids",
    "sample_idx",
    "unique_id",
    "gt",
    "enable_thinking",
    "eos_reached",
    "finish_reason",
    "gen_len",
    "rewrite_prompt_truncated_tokens",
    "data_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["student", "teacher", "trd"], required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--shard-dir")
    source.add_argument("--input-jsonl")
    parser.add_argument("--jsonl")
    parser.add_argument("--parquet")
    parser.add_argument("--expected-shards", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    shard_summaries = []
    if args.shard_dir:
        if args.expected_shards is None or args.jsonl is None:
            raise ValueError("--shard-dir requires --expected-shards and --jsonl")
        shard_dir = Path(args.shard_dir)
        shard_paths = sorted(shard_dir.glob("shard_*.jsonl"))
        if len(shard_paths) != args.expected_shards:
            raise RuntimeError(f"expected {args.expected_shards} shards, found {len(shard_paths)}")
        for shard_path in shard_paths:
            summary_path = Path(f"{shard_path}.summary.json")
            if not summary_path.exists():
                raise RuntimeError(f"missing shard summary: {summary_path}")
            shard_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            with shard_path.open(encoding="utf-8") as shard_file:
                rows.extend(json.loads(line) for line in shard_file if line.strip())
        jsonl_path = Path(args.jsonl)
    else:
        jsonl_path = Path(args.input_jsonl)
        with jsonl_path.open(encoding="utf-8") as input_file:
            rows.extend(json.loads(line) for line in input_file if line.strip())

    if not rows:
        raise RuntimeError("no offline trajectories found")

    rows.sort(key=lambda row: int(row["original_index"]))
    indices = [int(row["original_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise RuntimeError("duplicate original_index values across generation shards")

    source_counts = {int(summary["source_rows"]) for summary in shard_summaries if "source_rows" in summary}
    if len(source_counts) > 1:
        raise RuntimeError(f"generation shards disagree on source row count: {sorted(source_counts)}")
    if source_counts and len(rows) != next(iter(source_counts)):
        source_rows = next(iter(source_counts))
        raise RuntimeError(f"expected one trajectory for each of {source_rows} source rows, got {len(rows)}")

    if args.jsonl:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="offline_trajectory_merge_") as tmp_dir:
        if args.shard_dir:
            tmp_jsonl = Path(tmp_dir) / jsonl_path.name
            with tmp_jsonl.open("w", encoding="utf-8") as output_file:
                for row in rows:
                    output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            shutil.copyfile(tmp_jsonl, jsonl_path)

        if args.parquet:
            columns = TEACHER_COLUMNS if args.mode == "teacher" else TRD_COLUMNS
            missing = sorted(set(columns) - set(rows[0] if rows else columns))
            if missing:
                raise RuntimeError(f"missing parquet columns: {missing}")
            frame = pd.DataFrame([{key: row[key] for key in columns} for row in rows])
            parquet_path = Path(args.parquet)
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_parquet = Path(tmp_dir) / parquet_path.name
            frame.to_parquet(tmp_parquet, index=False)
            if len(pd.read_parquet(tmp_parquet)) != len(frame):
                raise RuntimeError("parquet row-count verification failed")
            shutil.copyfile(tmp_parquet, parquet_path)

    total_tokens = sum(int(row.get("gen_len", 0)) for row in rows)
    summary = {
        "mode": args.mode,
        "rows": len(rows),
        "jsonl": str(jsonl_path),
        "parquet": args.parquet,
        "avg_gen_tokens": total_tokens / max(len(rows), 1),
        "max_gen_tokens": max((int(row.get("gen_len", 0)) for row in rows), default=0),
        "eos_rate": sum(bool(row.get("eos_reached")) for row in rows) / max(len(rows), 1),
        "shards": shard_summaries,
    }
    if args.shard_dir:
        Path(f"{jsonl_path}.summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.parquet:
        Path(f"{args.parquet}.summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
