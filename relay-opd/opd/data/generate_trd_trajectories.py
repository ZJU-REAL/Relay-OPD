#!/usr/bin/env python3
"""Generate the student and teacher-rewrite stages of the TRD baseline."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


DEFAULT_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


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


def _apply_student_template(tokenizer, messages) -> list[int]:
    return _token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def _problem_context(messages) -> tuple[str, str]:
    system = next((item["content"] for item in messages if item["role"] == "system"), DEFAULT_SYSTEM)
    problem = next((item["content"] for item in messages if item["role"] == "user"), "")
    return system, problem


def _rewrite_messages(system: str, problem: str, initial_response: str) -> list[dict[str, str]]:
    prompt = f"""Your task is to rewrite your mathematical solution.

**Problem:**
{problem}

**Your Initial Solution:**
{initial_response}

**Instructions:**
1. Preserve the overall structure and reasoning path of your original solution
2. Identify and fix errors in computation or logic
3. Keep correct intermediate steps and meaningful work
4. Output ONLY the rewritten solution"""
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def _fit_rewrite_prompt(tokenizer, system, problem, initial_response, max_prompt_tokens):
    messages = _rewrite_messages(system, problem, initial_response)
    prompt_ids = _apply_student_template(tokenizer, messages)
    if len(prompt_ids) <= max_prompt_tokens:
        return messages, prompt_ids, 0

    empty_len = len(_apply_student_template(tokenizer, _rewrite_messages(system, problem, "")))
    response_budget = max(max_prompt_tokens - empty_len - 16, 0)
    initial_ids = tokenizer.encode(initial_response, add_special_tokens=False)
    retained = tokenizer.decode(initial_ids[:response_budget], skip_special_tokens=True)
    retained += "\n[... initial solution truncated ...]"
    messages = _rewrite_messages(system, problem, retained)
    prompt_ids = _apply_student_template(tokenizer, messages)[:max_prompt_tokens]
    return messages, prompt_ids, max(len(initial_ids) - response_budget, 0)


def _load_student_rows(args, tokenizer) -> tuple[list[dict], int]:
    dataframe = pd.read_parquet(args.data)
    rows = []
    for position, source in enumerate(dataframe.to_dict(orient="records")):
        if position % args.num_shards != args.shard_id:
            continue
        messages = _messages(source["prompt"])
        rows.append(
            {
                "original_index": int(position),
                "messages": messages,
                "prompt_token_ids": _apply_student_template(tokenizer, messages),
                "gt": _ground_truth(source),
            }
        )
    return rows, len(dataframe)


def _load_rewrite_rows(args, tokenizer) -> tuple[list[dict], int]:
    rows = []
    source_rows = 0
    max_prompt_tokens = args.max_model_len - args.max_new
    with Path(args.data).open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            source_rows += 1
            source = json.loads(line)
            original_index = int(source["original_index"])
            if original_index % args.num_shards != args.shard_id:
                continue
            trajectory_messages = _messages(source["messages"])
            base_messages = [item for item in trajectory_messages if item["role"] != "assistant"]
            initial_response = trajectory_messages[-1]["content"]
            system, problem = _problem_context(base_messages)
            rewrite_messages, rewrite_prompt_ids, truncated = _fit_rewrite_prompt(
                tokenizer,
                system,
                problem,
                initial_response,
                max_prompt_tokens,
            )
            rows.append(
                {
                    "original_index": original_index,
                    "messages": base_messages,
                    "rewrite_messages": rewrite_messages,
                    "prompt_token_ids": rewrite_prompt_ids,
                    "student_prompt_token_ids": _apply_student_template(tokenizer, base_messages),
                    "y_o_text": initial_response,
                    "y_o_response_token_ids": [int(token_id) for token_id in source["response_token_ids"]],
                    "gt": source.get("gt"),
                    "rewrite_prompt_truncated_tokens": int(truncated),
                }
            )
    return rows, source_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["student", "rewrite"], required=True)
    parser.add_argument("--model", required=True)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.student_tokenizer, trust_remote_code=True)
    rows, source_rows = (
        _load_student_rows(args, tokenizer)
        if args.mode == "student"
        else _load_rewrite_rows(args, tokenizer)
    )

    print(
        json.dumps(
            {
                "event": "start",
                "mode": args.mode,
                "model": args.model,
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
        model=args.model,
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
    truncated_prompts = 0
    with out_path.open("w", encoding="utf-8") as output_file:
        for row, request_output in zip(rows, outputs, strict=True):
            completion = request_output.outputs[0]
            response_ids = [int(token_id) for token_id in completion.token_ids]
            eos_reached = completion.finish_reason == "stop"
            total_tokens += len(response_ids)
            eos_count += int(eos_reached)
            max_gen_len = max(max_gen_len, len(response_ids))
            truncated_prompts += int(row.get("rewrite_prompt_truncated_tokens", 0) > 0)

            if args.mode == "student":
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
                    "data_source": "trd_student_trajectory",
                }
            else:
                record = {
                    "original_index": row["original_index"],
                    "unique_id": row["original_index"],
                    "messages": row["messages"] + [{"role": "assistant", "content": completion.text}],
                    "rewrite_messages": row["rewrite_messages"]
                    + [{"role": "assistant", "content": completion.text}],
                    "prompt_token_ids": row["student_prompt_token_ids"],
                    "teacher_prompt_token_ids": row["prompt_token_ids"],
                    "response_token_ids": response_ids,
                    "teacher_response_token_ids": response_ids,
                    "y_o_response_token_ids": row["y_o_response_token_ids"],
                    "sample_idx": 0,
                    "gt": row["gt"],
                    "enable_thinking": False,
                    "eos_reached": eos_reached,
                    "finish_reason": completion.finish_reason,
                    "gen_len": len(response_ids),
                    "rewrite_prompt_truncated_tokens": row["rewrite_prompt_truncated_tokens"],
                    "data_source": "trd_rewrite_student_nothink_prompt",
                }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "mode": args.mode,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "rows": len(rows),
        "source_rows": source_rows,
        "eos_rate": eos_count / max(len(rows), 1),
        "avg_gen_tokens": total_tokens / max(len(rows), 1),
        "max_gen_tokens": max_gen_len,
        "prompt_truncated_rows": truncated_prompts,
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
