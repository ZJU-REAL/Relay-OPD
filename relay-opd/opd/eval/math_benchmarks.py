"""Evaluate language models on the math benchmarks used by this project.

Answers are checked by OpenMathInstruct parsing, math-verify, and OBJudge for
OlympiadBench. The runner supports independent sampling and avg/pass metrics.

Two modes:
  1. Plain eval (default): single model rollout.
       --model /path/to/ckpt
  2. Patched speculative rollout: target=--model and draft=--draft_model.
       --model /path/to/teacher --draft_model /path/to/student \
       --num_spec_tokens 4 --rollout_mode relay --trigger_topk 5

Grader / bench data dirs are provided through arguments or environment:
  MATH_GRADER_PATH (default: ``opd/reward/grader`` in this repository)

Usage:
    python math_benchmarks.py \
        --model /path/to/model --bench math500 \
        --data_dir /path/to/bench/parquets \
        --n_samples 1 --max_new 4096 --temperature 0.7 \
        --tag teacher_math500
"""
import argparse
import json
import os
import sys
import time

os.environ["VLLM_ALLREDUCE_USE_FLASHINFER"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Grader path (math_verify + openmathinst_utils + objudge). Override via env.
_DEFAULT_GRADER = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reward", "grader")
)
_GRADER_PATH = os.environ.get("MATH_GRADER_PATH") or os.environ.get("GRADER_PATH", _DEFAULT_GRADER)
sys.path.insert(0, _GRADER_PATH)
from math_verify import verify, parse  # type: ignore
from openmathinst_utils import process_results  # local file in $GRADER_PATH
from objudge import OBJudge  # local file in $GRADER_PATH


BENCH_PATHS = {
    "math500": "math500.parquet",
    "aime24": "aime-24.parquet",
    "aime25": "aime-2025.parquet",
    "aime26": "aime26.parquet",
    "amc": "amc.parquet",
    "amc23": "amc23.parquet",
    "olympiad": "olympiadBench.parquet",
    "olympiadbench": "olympiadbench.parquet",
    "minerva": "minerva.parquet",
    "minervamath": "minervamath.parquet",
    "gpqa": "gpqa.parquet",
    "gpqa_diamond": "gpqa_diamond.parquet",
    "hmmt_feb_2026": "hmmt_feb_2026.parquet",
    "hmmt_nov_2025": "hmmt_nov_2025.parquet",
    "dapo128": "dapo128.parquet",
}


def grade_math_answer(resp, gt_answer, bench):
    """Return whether a response matches the benchmark answer."""
    try:
        if bench == "olympiad":
            if isinstance(gt_answer, list) and len(gt_answer) > 0:
                gt = gt_answer[0]
            else:
                gt = gt_answer
            gt = str(gt) if gt is not None else ""
            scorer = OBJudge()
            return bool(
                process_results(resp, gt, response_extract_from_boxed=True)
                or process_results(resp, gt, response_extract_from_boxed=False, response_extract_regex=r"The answer is: (.+)$")
                or verify(parse(f"\\boxed{{${gt}}}$"), parse(resp))
                or scorer.judge(gt, resp if "</think>" not in resp else resp.split("</think>")[1].strip(), 1e-8)
            )
        else:
            gt = str(gt_answer) if gt_answer is not None else ""
            return bool(
                process_results(resp, gt, response_extract_from_boxed=True)
                or process_results(resp, gt, response_extract_from_boxed=False, response_extract_regex=r"The answer is: (.+)$")
                or verify(parse(f"\\boxed{{${gt}}}$"), parse(resp))
            )
    except Exception:
        return False


def load_unique_problems(path):
    """Return list of {msgs, gt} unique problems (dedup'd)."""
    df = pd.read_parquet(path)
    seen = set()
    out = []
    for _, row in df.iterrows():
        msgs = row["prompt"]
        if hasattr(msgs, "tolist"):
            msgs = msgs.tolist()
        msgs = [{"role": m["role"], "content": m["content"]} for m in msgs]
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        if user in seen:
            continue
        seen.add(user)
        rm = row.get("reward_model")
        if isinstance(rm, dict):
            gt = rm.get("ground_truth")
        else:
            gt = None
        out.append({"msgs": msgs, "gt": gt})
    return out


def to_prompt_token_ids(template_output):
    """Normalize tokenizer chat-template outputs to vLLM's list[int] input."""
    if hasattr(template_output, "data") and "input_ids" in template_output:
        template_output = template_output["input_ids"]
    elif isinstance(template_output, dict) and "input_ids" in template_output:
        template_output = template_output["input_ids"]
    if hasattr(template_output, "tolist"):
        template_output = template_output.tolist()
    if (isinstance(template_output, list) and template_output
            and isinstance(template_output[0], list)):
        template_output = template_output[0]
    return [int(x) for x in template_output]


def _setup_speculative_rollout(args):
    """Configure the OPD vLLM patch when a draft model is provided."""
    if not args.draft_model:
        return None

    # Find the vLLM patch directory. Override via OPD_PATCH_PATH.
    here = os.path.dirname(os.path.abspath(__file__))
    default_patch = os.path.normpath(os.path.join(here, "..", "patches", "vllm"))
    patch_path = os.environ.get("OPD_PATCH_PATH", default_patch)
    if not os.path.isdir(patch_path):
        raise FileNotFoundError(
            f"OPD patch directory not found at {patch_path}. "
            "Set OPD_PATCH_PATH to the opd/patches/vllm directory."
        )
    sys.path.insert(0, patch_path)

    # Set env BEFORE importing the patch so the patch's env reads pick them up.
    os.environ["VERL_OPD_VLLM_PATCH"] = "1"
    os.environ["VERL_OPD_ROLLOUT_MODE"] = args.rollout_mode
    os.environ["SKD_ACCEPTANCE_TOPK"] = str(args.skd_topk)
    os.environ["RELAY_OPD_TRIGGER_TOPK"] = str(args.trigger_topk)
    os.environ["RELAY_OPD_MAX_TAKEOVERS"] = str(args.max_takeovers)
    os.environ["RELAY_OPD_PARAGRAPHS_PER_TAKEOVER"] = str(args.paragraphs_per_takeover)
    os.environ["VERL_OPD_TOKENIZER_PATH"] = args.tokenizer_model or args.draft_model
    os.environ["VERL_OPD_DRAFT_SAMPLING"] = "1"
    os.environ["VERL_OPD_COLLECT_DRAFT_PROBS"] = "1"
    os.environ["VERL_OPD_DRAFT_TEMPERATURE"] = str(args.temperature)
    os.environ["VERL_OPD_DRAFT_TOP_P"] = str(args.top_p)
    os.environ["VERL_OPD_SOURCE_METRICS"] = "1"

    # Defensive: explicitly apply patches in case sitecustomize.py didn't fire
    # (e.g., if PYTHONPATH didn't include the patch dir). When PYTHONPATH IS
    # set correctly, apply_patches() is idempotent so this is a safe no-op.
    from speculative_decode import apply_patches
    apply_patches()

    return {
        "model": args.draft_model,
        "method": "draft_model",
        "num_speculative_tokens": args.num_spec_tokens,
        "draft_tensor_parallel_size": args.draft_tp or args.tp,
    }


def _truthy_env(name):
    value = os.environ.get(name)
    return value is not None and value.lower() in {"1", "true", "yes", "y", "on"}


def _resolve_rollout_trace_mode(args):
    mode = (args.trace_mode or "auto").lower()
    if mode != "auto":
        return mode
    env_mode = os.environ.get("VERL_OPD_TRACE_MODE")
    if env_mode:
        return env_mode.lower()
    return "full" if _truthy_env("VERL_OPD_TRACE") else "none"


def _prepare_rollout_trace_env(args):
    if not args.draft_model or not args.save_rollout_trace:
        return
    mode = _resolve_rollout_trace_mode(args)
    if mode == "none":
        return
    os.environ["VERL_OPD_TRACE"] = "1"
    os.environ["VERL_OPD_TRACE_MODE"] = mode
    if not os.environ.get("VERL_OPD_TRACE_JSONL"):
        os.environ["VERL_OPD_TRACE_JSONL"] = os.path.join(args.out_dir, "opd_trace_events.jsonl")


def _load_opd_trace_events(path):
    by_rid = {}
    if not path or not os.path.exists(path):
        return by_rid
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(event.get("request_id", ""))
            if not rid:
                continue
            positions = event.get("positions")
            if positions is not None:
                tokens = event.get("tokens") or []
                for pos, tok in zip(positions, tokens):
                    by_rid.setdefault(rid, []).append({
                        "response_pos": int(pos),
                        "emit_token_id": int(tok),
                        "has_logits": False,
                    })
                continue
            by_rid.setdefault(rid, []).append(event)
    return by_rid


def _load_stop_events(path):
    by_rid = {}
    if not path or not os.path.exists(path):
        return by_rid
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(event.get("request_id", ""))
            if not rid:
                continue
            by_rid.setdefault(rid, []).append(event)
    return by_rid


def _best_stop_event(events):
    if not events:
        return None
    usable = [e for e in events if e.get("response_pos") is not None]
    if not usable:
        return None
    return min(usable, key=lambda e: int(e.get("response_pos", 10**18)))


def _strip_internal_stop_token(token_ids, text, request_id, stop_events, tok):
    event = _best_stop_event(stop_events.get(request_id))
    if event is None:
        return token_ids, text, None
    pos = int(event["response_pos"])
    internal_stop = event.get("internal_stop_token_id")
    stripped = list(token_ids)
    removed = False
    if 0 <= pos < len(stripped):
        if internal_stop is None or int(stripped[pos]) == int(internal_stop):
            stripped = stripped[:pos]
            removed = True
    elif pos == len(stripped):
        # Some vLLM paths omit the terminal stop token from final token_ids.
        removed = True

    info = dict(event)
    info["raw_gen_len"] = len(token_ids)
    info["stripped_gen_len"] = len(stripped)
    info["forced_stop_token_removed"] = bool(removed)
    if removed:
        text = tok.decode(stripped, skip_special_tokens=True)
    else:
        info["strip_warning"] = "truncate event did not match final token_ids"
    return stripped, text, info


def _trim_rollout_trace(trace, keep_len):
    if not trace:
        return trace
    out = dict(trace)
    mask = out.get("teacher_mask")
    if isinstance(mask, list):
        out["teacher_mask"] = mask[:keep_len]
    positions = out.get("positions")
    tokens = out.get("tokens")
    if isinstance(positions, list):
        kept_positions = []
        kept_tokens = []
        token_iter = tokens if isinstance(tokens, list) else [None] * len(positions)
        for pos, tok_id in zip(positions, token_iter):
            try:
                pos_i = int(pos)
            except Exception:
                continue
            if 0 <= pos_i < keep_len:
                kept_positions.append(pos_i)
                kept_tokens.append(tok_id)
        out["positions"] = kept_positions
        if isinstance(tokens, list):
            out["tokens"] = kept_tokens
    events = out.get("events")
    if isinstance(events, list):
        out["events"] = [
            e for e in events
            if e.get("response_pos") is None or int(e.get("response_pos", -1)) < keep_len
        ]
    return out


def _rollout_trace_from_events(request_id, gen_len, events, include_events):
    mask = [False] * gen_len
    by_pos = {}
    for event in events or []:
        pos = event.get("response_pos")
        tok = event.get("emit_token_id")
        if pos is None:
            continue
        pos = int(pos)
        if 0 <= pos < gen_len:
            mask[pos] = True
            by_pos[pos] = int(tok) if tok is not None else None
    positions = sorted(by_pos)
    trace = {
        "request_id": str(request_id),
        "teacher_mask": mask,
        "positions": positions,
        "tokens": [by_pos[p] for p in positions],
    }
    if include_events:
        trace["events"] = events or []
    return trace


def run_one_bench(args, llm, tok, spec_config, bench, n_samples):
    """Eval ONE bench with an already-loaded llm/tok. Writes <tag>.jsonl /
    <tag>.summary.json where tag = bench (multi-bench mode) or args.tag (single)."""
    tag = bench if args.benches else (args.tag or bench)
    out_path = os.path.join(args.out_dir, f"{tag}.jsonl")
    summary_path = os.path.join(args.out_dir, f"{tag}.summary.json")
    if os.path.exists(summary_path):
        print(f"[skip] {bench}: {summary_path} exists")
        return

    bench_path = os.path.join(args.data_dir, BENCH_PATHS[bench])
    problems = load_unique_problems(bench_path)
    if args.n_problems > 0:
        problems = problems[:args.n_problems]
    total_problems_before_shard = len(problems)
    if args.num_shards > 1:
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            raise ValueError(
                f"shard_id must be in [0, {args.num_shards}), got {args.shard_id}"
            )
        problems = [
            p for idx, p in enumerate(problems)
            if idx % args.num_shards == args.shard_id
        ]
        print(
            f"[shard] {bench}: shard {args.shard_id}/{args.num_shards} "
            f"keeps {len(problems)}/{total_problems_before_shard} problems"
        )
    print(f"[ok] {bench}: {len(problems)} unique problems × {n_samples} samples = {len(problems)*n_samples} gens")

    prompt_token_ids = []
    problem_idx_map = []
    sample_idx_map = []
    for pi, p in enumerate(problems):
        template_kwargs = {"add_generation_prompt": True}
        if args.disable_thinking:
            template_kwargs["enable_thinking"] = False
        ids = to_prompt_token_ids(tok.apply_chat_template(p["msgs"], tokenize=True, **template_kwargs))
        for k in range(n_samples):
            prompt_token_ids.append(ids)
            problem_idx_map.append(pi)
            sample_idx_map.append(k)

    # Per-replica seed: the n_samples copies of a problem MUST be independent.
    # A single shared seed + identical prompt makes vLLM produce identical outputs,
    # collapsing mean@k to pass@1. seed = base + replica_idx keeps replicas distinct
    # yet reproducible (replica k of every problem uses base+k).
    sp_list = [
        SamplingParams(
            max_tokens=args.max_new, temperature=args.temperature,
            top_p=args.top_p, seed=args.seed + sample_idx_map[i],
        )
        for i in range(len(prompt_token_ids))
    ]
    trace_mode = _resolve_rollout_trace_mode(args)
    trace_file = None
    if spec_config is not None and args.save_rollout_trace and trace_mode != "none":
        trace_file = os.environ.get("VERL_OPD_TRACE_JSONL")
        if not trace_file:
            trace_name = "opd_trace_events.jsonl" if not args.benches else f"{tag}.opd_trace_events.jsonl"
            trace_file = os.path.join(args.out_dir, trace_name)
            os.environ["VERL_OPD_TRACE_JSONL"] = trace_file
        if os.path.exists(trace_file):
            os.remove(trace_file)
    stop_event_file = None
    if spec_config is not None and args.rollout_mode == "trigger_stop":
        stop_event_file = os.environ.get("VERL_OPD_STOP_EVENTS_JSONL")
        if not stop_event_file:
            stop_event_file = os.path.join(args.out_dir, "opd_stop_events.jsonl")
            os.environ["VERL_OPD_STOP_EVENTS_JSONL"] = stop_event_file
        if os.path.exists(stop_event_file):
            os.remove(stop_event_file)
    print(f"[generate] {bench}: {len(prompt_token_ids)} requests...")
    t0 = time.time()
    outputs = llm.generate(
        prompts=[{"prompt_token_ids": ids} for ids in prompt_token_ids],
        sampling_params=sp_list,
        use_tqdm=True,
    )
    elapsed = time.time() - t0
    total_tok = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"[ok] {bench} generated in {elapsed:.1f}s ({total_tok/elapsed:.1f} tok/s)")

    opd_stats = {}
    if spec_config is not None:
        try:
            from speculative_decode import pop_opd_stats
            opd_stats = pop_opd_stats()
            teacher_t = opd_stats.get("teacher_tokens", 0)
            student_t = opd_stats.get("student_tokens", 0)
            total_t = teacher_t + student_t
            if total_t > 0:
                print(f"[opd-eval] teacher_token_ratio={teacher_t/total_t:.3f} "
                      f"({teacher_t}/{total_t}) accept_rate={student_t/total_t:.3f}")
            if opd_stats:
                print(f"[opd-eval] stats={json.dumps(opd_stats, ensure_ascii=False, sort_keys=True)}")
        except Exception as e:
            print(f"[opd-eval] stats error: {e}")

    side_trace_events = _load_opd_trace_events(trace_file)
    if side_trace_events:
        print(
            f"[opd-eval] loaded side trace: mode={trace_mode} "
            f"requests={len(side_trace_events)} path={trace_file}"
        )
    stop_events = _load_stop_events(stop_event_file)
    if stop_events:
        print(
            f"[opd-eval] loaded trigger-stop events: requests={len(stop_events)} "
            f"path={stop_event_file}"
        )

    fout = open(out_path, "w")
    per_problem_scores = [[] for _ in problems]
    trace_getter = None
    if spec_config is not None and args.save_rollout_trace:
        try:
            from speculative_decode import pop_rollout_trace
            trace_getter = pop_rollout_trace
        except Exception as e:
            print(f"[opd-eval] trace import error: {e}")
    for i, (pi, out) in enumerate(zip(problem_idx_map, outputs)):
        text = out.outputs[0].text
        gen_token_ids = list(out.outputs[0].token_ids)
        request_id = str(getattr(out, "request_id", i))
        gen_token_ids, text, stop_info = _strip_internal_stop_token(
            gen_token_ids, text, request_id, stop_events, tok
        )
        finish = out.outputs[0].finish_reason
        raw_finish = finish
        if stop_info is not None and stop_info.get("forced_stop_token_removed"):
            finish = "trigger_stop"
        eos_reached = finish == "stop"
        gt = problems[pi]["gt"]
        correct = grade_math_answer(text, gt, bench)
        per_problem_scores[pi].append(int(correct))
        rec = {
            "req_idx": i, "problem_idx": pi, "request_id": request_id,
            "gt": str(gt) if gt is not None else None,
            "correct": correct, "gen_len": len(gen_token_ids),
            "eos_reached": eos_reached, "finish_reason": finish, "gen_text": text,
        }
        if stop_info is not None:
            rec["trigger_stop"] = stop_info
            rec["raw_finish_reason"] = raw_finish
        if args.save_token_ids:
            rec["gen_token_ids"] = gen_token_ids
        if request_id in side_trace_events:
            rec["rollout_trace"] = _rollout_trace_from_events(
                request_id=request_id,
                gen_len=len(gen_token_ids),
                events=side_trace_events.get(request_id, []),
                include_events=(trace_mode == "full"),
            )
        elif trace_getter is not None:
            trace = trace_getter(request_id)
            if trace is not None:
                rec["rollout_trace"] = _trim_rollout_trace(trace, len(gen_token_ids))
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fout.close()

    avg_at_k = sum(sum(s) / max(len(s), 1) for s in per_problem_scores) / max(len(problems), 1)
    pass_at_1 = sum(s[0] for s in per_problem_scores if s) / max(len(problems), 1)
    pass_at_k = sum(1 for s in per_problem_scores if any(s)) / max(len(problems), 1)
    # n_unique replicas per problem (diversity sanity check; ==1 means seed-collapsed)
    summary = {
        "tag": tag, "model": args.model, "bench": bench,
        "n_problems": len(problems), "n_samples": n_samples,
        "avg@k": avg_at_k, "pass@1": pass_at_1, "pass@k": pass_at_k, "wall_seconds": elapsed,
        "settings": {
            "max_new": args.max_new, "temperature": args.temperature,
            "top_p": args.top_p, "seed": args.seed,
            "disable_thinking": args.disable_thinking,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "total_problems_before_shard": total_problems_before_shard,
            "spec": (None if spec_config is None else {
                "draft_model": args.draft_model, "num_spec_tokens": args.num_spec_tokens,
                "draft_tp": args.draft_tp or args.tp,
                "rollout_mode": args.rollout_mode,
                "skd_topk": args.skd_topk,
                "trigger_topk": args.trigger_topk,
                "max_takeovers": args.max_takeovers,
                "paragraphs_per_takeover": args.paragraphs_per_takeover,
            }),
        },
        "opd_stats": opd_stats,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n=== {bench} summary ===")
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Model path; in speculative mode this is the target/teacher.")
    ap.add_argument("--bench", default=None, choices=list(BENCH_PATHS.keys()))
    ap.add_argument("--benches", default=None,
                    help="Comma-separated benches to eval in ONE model load "
                         "(e.g. aime24,aime25,math500). Overrides --bench. Each "
                         "bench writes <bench>.summary.json under --out_dir.")
    ap.add_argument("--bench_n_samples", default=None,
                    help="Comma-separated per-bench n_samples parallel to --benches "
                         "(e.g. 32,32,4). Defaults to --n_samples for all.")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=1, help="Samples per problem for avg@k")
    ap.add_argument("--max_new", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max_model_len", type=int, default=40960)
    ap.add_argument("--n_problems", type=int, default=-1, help="Limit unique problems (-1 = all)")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Split unique problems by idx %% num_shards for data-parallel eval.")
    ap.add_argument("--shard_id", type=int, default=0,
                    help="Shard id used with --num_shards.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None,
                    help="Output tag (single-bench). In --benches mode the tag is "
                         "the bench name.")
    ap.add_argument("--out_dir", default="eval_results")
    ap.add_argument("--disable_thinking", action="store_true",
                    help="Pass enable_thinking=False to Qwen3-style chat templates.")
    ap.add_argument("--save_token_ids", action="store_true",
                    help="Include generated token ids in jsonl records.")
    ap.add_argument("--save_rollout_trace", action="store_true",
                    help="Include token-source positions and diagnostics in JSONL records.")
    ap.add_argument("--trace_mode", default="auto",
                    choices=["auto", "none", "mask", "full"],
                    help="Trace payload for --save_rollout_trace. mask stores only teacher "
                         "positions/tokens; full also stores logp/top-k diagnostics.")
    ap.add_argument("--tokenizer_model", default=None,
                    help="Override tokenizer source (use a different model's template).")
    ap.add_argument("--enforce_eager", action="store_true",
                    help="Force eager mode (disable CUDA graphs). Default: CUDA graphs "
                         "ON for baseline eval (big speedup on long decodes like AIME), "
                         "eager for spec modes.")

    ap.add_argument("--moe_backend", default=os.environ.get("VLLM_MOE_BACKEND"),
                    help="vLLM MoE backend override, e.g. triton. Defaults to "
                         "VLLM_MOE_BACKEND when set.")

    # Patched speculative rollout args (optional).
    ap.add_argument("--draft_model", default=None,
                    help="Enable patched speculative rollout with this draft/student model.")
    ap.add_argument("--num_spec_tokens", type=int, default=4,
                    help="Number of speculative draft tokens per verification step.")
    ap.add_argument("--draft_tp", type=int, default=None,
                    help="Draft-model tensor parallelism; defaults to --tp.")
    ap.add_argument("--skd_topk", type=int, default=25,
                    help="Teacher top-k acceptance threshold used by SKD.")
    ap.add_argument("--trigger_topk", type=int, default=5,
                    help="Student-rank divergence threshold used by Relay-OPD/Trigger-Stop.")
    ap.add_argument("--max_takeovers", type=int, default=2,
                    help="Maximum Relay-OPD teacher takeovers per response.")
    ap.add_argument("--paragraphs_per_takeover", type=int, default=3,
                    help="Teacher paragraphs generated by each Relay-OPD takeover.")
    ap.add_argument("--rollout_mode", default="skd", choices=["skd", "trigger_stop", "relay"],
                    help="Patched speculative rollout policy.")

    args = ap.parse_args()

    # Resolve bench list + per-bench n_samples. --benches enables multi-bench
    # eval under ONE model load; else fall back to single --bench/--tag.
    if args.benches:
        bench_list = [b.strip() for b in args.benches.split(",") if b.strip()]
        if args.bench_n_samples:
            ns_list = [int(x) for x in args.bench_n_samples.split(",")]
            assert len(ns_list) == len(bench_list), "bench_n_samples must match benches"
        else:
            ns_list = [args.n_samples] * len(bench_list)
    else:
        assert args.bench is not None, "provide --bench or --benches"
        bench_list = [args.bench]
        ns_list = [args.n_samples]

    os.makedirs(args.out_dir, exist_ok=True)

    # Setup must happen before vLLM initialization.
    _prepare_rollout_trace_env(args)
    spec_config = _setup_speculative_rollout(args)
    if spec_config is not None:
        print(f"[opd-eval] target={args.model} draft={args.draft_model} "
              f"mode={args.rollout_mode} num_spec={args.num_spec_tokens}")

    # Tokenizer selection priority: --tokenizer_model > --draft_model > --model
    # In spec-decode mode, draft_model's tokenizer ensures correct template.
    tok_model = args.tokenizer_model or args.draft_model or args.model
    tok = AutoTokenizer.from_pretrained(tok_model)
    if tok_model != args.model:
        print(f"[tok] using tokenizer from {tok_model} (not {args.model})")
    print(f"[load] {args.model}")
    llm_kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        enforce_eager=(args.enforce_eager or spec_config is not None),
        seed=args.seed,
        disable_log_stats=True,
        disable_custom_all_reduce=True,
    )
    if spec_config is not None:
        llm_kwargs["speculative_config"] = spec_config
    if args.moe_backend:
        llm_kwargs["moe_backend"] = args.moe_backend
        print(f"[vllm] moe_backend={args.moe_backend}")
    llm = LLM(**llm_kwargs)

    # One model load, multiple benches.
    for bench, n_samples in zip(bench_list, ns_list):
        run_one_bench(args, llm, tok, spec_config, bench, n_samples)


if __name__ == "__main__":
    main()
