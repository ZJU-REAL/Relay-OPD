#!/usr/bin/env python3
"""Generate one offline teacher trajectory per prompt for SFT and SeqKD."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def _token_ids(template_output) -> list[int]:
    if hasattr(template_output, "data") and "input_ids" in template_output:
        template_output = template_output["input_ids"]
    elif isinstance(template_output, dict):
        template_output = template_output["input_ids"]
    if hasattr(template_output, "tolist"):
        template_output = template_output.tolist()
    if template_output and isinstance(template_output[0], list):
        template_output = template_output[0]
    return [int(token_id) for token_id in template_output]


def _messages(value) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [{"role": str(item["role"]), "content": str(item["content"])} for item in value]


def _ground_truth(row: dict):
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth")
    return row.get("gt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--student-tokenizer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-new", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=18433)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=65536)
    parser.add_argument("--data-source", default="teacher_rollout_student_nothink_prompt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_parquet(args.data)
    source_rows = len(dataframe)
    tokenizer = AutoTokenizer.from_pretrained(args.student_tokenizer, trust_remote_code=True)
    rows = []
    for position, source in enumerate(dataframe.to_dict(orient="records")):
        if position % args.num_shards != args.shard_id:
            continue
        messages = _messages(source["prompt"])
        prompt_ids = _token_ids(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        rows.append(
            {
                "original_index": int(position),
                "messages": messages,
                "prompt_token_ids": prompt_ids,
                "gt": _ground_truth(source),
            }
        )

    print(
        json.dumps(
            {
                "event": "start",
                "teacher_model": args.teacher_model,
                "student_tokenizer": args.student_tokenizer,
                "data": args.data,
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "requests": len(rows),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    llm = LLM(
        model=args.teacher_model,
        tokenizer=args.student_tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
    )
    sampling_params = [
        SamplingParams(
            max_tokens=args.max_new,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        for _ in rows
    ]
    started = time.time()
    outputs = llm.generate(
        prompts=[{"prompt_token_ids": row["prompt_token_ids"]} for row in rows],
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    elapsed = time.time() - started

    total_tokens = 0
    eos_count = 0
    max_gen_len = 0
    with out_path.open("w", encoding="utf-8") as output_file:
        for row, request_output in zip(rows, outputs, strict=True):
            completion = request_output.outputs[0]
            response_ids = [int(token_id) for token_id in completion.token_ids]
            eos_reached = completion.finish_reason == "stop"
            total_tokens += len(response_ids)
            eos_count += int(eos_reached)
            max_gen_len = max(max_gen_len, len(response_ids))
            record = {
                "original_index": row["original_index"],
                "unique_id": row["original_index"],
                "messages": row["messages"] + [{"role": "assistant", "content": completion.text}],
                "prompt_token_ids": row["prompt_token_ids"],
                "response_token_ids": response_ids,
                "sample_idx": 0,
                "gt": row["gt"],
                "enable_thinking": False,
                "eos_reached": eos_reached,
                "finish_reason": completion.finish_reason,
                "gen_len": len(response_ids),
                "data_source": args.data_source,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "rows": len(rows),
        "source_rows": source_rows,
        "eos_rate": eos_count / max(len(rows), 1),
        "avg_gen_tokens": total_tokens / max(len(rows), 1),
        "max_gen_tokens": max_gen_len,
        "wall_seconds": elapsed,
        "tokens_per_second": total_tokens / max(elapsed, 1e-9),
        "settings": vars(args),
    }
    Path(f"{out_path}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
