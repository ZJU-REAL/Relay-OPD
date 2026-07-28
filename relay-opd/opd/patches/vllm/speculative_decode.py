"""vLLM speculative-decoding patches for OPD rollouts.

The patch provides the SKD top-k baseline, Trigger-Stop, and Relay-OPD while
keeping the implementation independent of any training frontend.

Usage: add this directory to PYTHONPATH and set VERL_OPD_VLLM_PATCH=1.
Patches are applied via sitecustomize.py or explicit apply_patches() call.
"""

from __future__ import annotations

import logging
import json
import math
import os
import re
import socket
import struct
from typing import Any

import torch

logger = logging.getLogger(__name__)

PLACEHOLDER_TOKEN_ID = -1

_PATCHED = False
_OPD_STATS: dict[str, torch.Tensor] = {}
_TEACHER_MASK_BY_REQUEST_ID: dict[str, list[bool]] = {}
_TOKEN_EVENTS_BY_REQUEST_ID: dict[str, list[int]] = {}
_POSITION_EVENTS_BY_REQUEST_ID: dict[str, list[int]] = {}
_TRACE_EVENTS_BY_REQUEST_ID: dict[str, list[dict[str, Any]]] = {}
_TAKEOVER_TOKENS_REMAINING: dict[str, int] = {}
_TAKEOVER_PARAGRAPHS_REMAINING: dict[str, int] = {}
_COMPLETED_TAKEOVERS: dict[str, int] = {}
_PARAGRAPH_BOUNDARY_IDS_CACHE: dict[str, torch.Tensor] = {}
_INTERNAL_STOP_ID_CACHE: dict[str, int | None] = {}

_RANDOM_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$")


def pop_rollout_trace(request_id: str) -> dict[str, Any] | None:
    """Pop per-token trace for an in-process eval request."""
    if request_id is None:
        return None
    rid = _strip_vllm_suffix(str(request_id))
    mask = _TEACHER_MASK_BY_REQUEST_ID.pop(rid, None)
    tokens = _TOKEN_EVENTS_BY_REQUEST_ID.pop(rid, None)
    positions = _POSITION_EVENTS_BY_REQUEST_ID.pop(rid, None)
    events = _TRACE_EVENTS_BY_REQUEST_ID.pop(rid, None)
    if mask is None and tokens is None and positions is None and events is None:
        return None
    return {
        "request_id": rid,
        "teacher_mask": mask,
        "tokens": tokens,
        "positions": positions,
        "events": events or [],
    }


_MASK_IPC_SOCK: socket.socket | None = None


def _get_mask_ipc_sock() -> socket.socket | None:
    global _MASK_IPC_SOCK
    sock_path = os.environ.get("VERL_OPD_MASK_IPC_SOCKET")
    if not sock_path:
        return None
    if _MASK_IPC_SOCK is None:
        _MASK_IPC_SOCK = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        _MASK_IPC_SOCK.connect(sock_path)
    return _MASK_IPC_SOCK


def _send_mask_chunk_ipc(
    request_id: str,
    mask_chunk: list[bool],
    token_chunk: list[int],
    position_chunk: list[int] | None = None,
    action_token_chunk: list[int] | None = None,
    teacher_action_logp_chunk: list[float] | None = None,
) -> None:
    """Send emitted-token/mask events to the server process via Unix datagram."""
    global _MASK_IPC_SOCK
    if not request_id or not mask_chunk:
        return
    if (
        len(mask_chunk) != len(token_chunk) or len(mask_chunk) > 255
        or position_chunk is None
        or len(position_chunk) != len(mask_chunk)
        or (action_token_chunk is not None and len(action_token_chunk) != len(mask_chunk))
        or (teacher_action_logp_chunk is not None and len(teacher_action_logp_chunk) != len(mask_chunk))
    ):
        raise RuntimeError(
            "malformed OPD mask IPC chunk: "
            f"rid={request_id} masks={len(mask_chunk)} tokens={len(token_chunk)} "
            f"positions={None if position_chunk is None else len(position_chunk)} "
            f"actions={None if action_token_chunk is None else len(action_token_chunk)} "
            f"action_logps={None if teacher_action_logp_chunk is None else len(teacher_action_logp_chunk)}"
        )
    sock = _get_mask_ipc_sock()
    if sock is None:
        return
    n = len(mask_chunk)
    mask_bytes = bytes(1 if x else 0 for x in mask_chunk)
    token_bytes = struct.pack(f"!{n}i", *[int(x) for x in token_chunk])
    position_bytes = struct.pack(f"!{n}i", *[int(x) for x in position_chunk])
    if action_token_chunk is not None and teacher_action_logp_chunk is not None:
        action_bytes = struct.pack(f"!{n}i", *[int(x) for x in action_token_chunk])
        action_logp_bytes = struct.pack(f"!{n}f", *[float(x) for x in teacher_action_logp_chunk])
        body = b"OPD3" + bytes([n]) + mask_bytes + token_bytes + position_bytes + action_bytes + action_logp_bytes
    else:
        # OPD2 carries absolute response positions. This lets the server place
        # each mask bit directly instead of guessing an offset from token text.
        body = b"OPD2" + bytes([n]) + mask_bytes + token_bytes + position_bytes
    payload = request_id.encode("utf-8") + b"\0" + body
    try:
        sock.send(payload)
    except Exception:
        try:
            _MASK_IPC_SOCK.close()
        except Exception:
            pass
        _MASK_IPC_SOCK = None


def _send_stop_event_ipc(request_id: str, event: dict[str, Any]) -> None:
    """Send an internal trigger-stop event to the rollout server."""
    global _MASK_IPC_SOCK
    if not request_id or not event:
        return
    sock = _get_mask_ipc_sock()
    if sock is None:
        return
    try:
        body = b"OPS1" + json.dumps(event, ensure_ascii=False).encode("utf-8")
        payload = request_id.encode("utf-8") + b"\0" + body
        sock.send(payload)
    except Exception:
        try:
            _MASK_IPC_SOCK.close()
        except Exception:
            pass
        _MASK_IPC_SOCK = None


def _strip_vllm_suffix(internal_rid: str) -> str:
    return _RANDOM_SUFFIX_RE.sub("", internal_rid)


def _flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on", "sample"}


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _stats_enabled() -> bool:
    return _flag("VERL_OPD_SOURCE_METRICS", True)


def _trace_enabled() -> bool:
    return _flag("VERL_OPD_TRACE", False)


def _trace_mode() -> str:
    mode = os.environ.get("VERL_OPD_TRACE_MODE", "full").strip().lower()
    if mode in {"mask", "masks", "position", "positions", "minimal", "light"}:
        return "mask"
    return "full"


def _trace_file_writer_enabled() -> bool:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank()) == 0
    except Exception:
        return True
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank not in (None, ""):
        try:
            return int(local_rank) == 0
        except ValueError:
            return False
    return True


def _stats_add(key: str, value: torch.Tensor | int | float, device: torch.device | None = None) -> None:
    if not _stats_enabled():
        return
    with torch.no_grad():
        if isinstance(value, torch.Tensor):
            val = value.detach().float().sum()
        else:
            val = torch.tensor(float(value), dtype=torch.float32, device=device)
        if key in _OPD_STATS:
            _OPD_STATS[key] = _OPD_STATS[key].to(device=val.device) + val
        else:
            _OPD_STATS[key] = val


def pop_opd_stats() -> dict[str, float]:
    stats = {key: float(value.detach().float().cpu().item()) for key, value in _OPD_STATS.items()}
    _OPD_STATS.clear()
    return stats


def _get_internal_stop_token_id(vocab_size: int) -> int | None:
    """Resolve the internal stop token used only to make vLLM finish a request."""
    env_val = os.environ.get("VERL_OPD_INTERNAL_STOP_TOKEN_ID")
    tok_path = os.environ.get("VERL_OPD_TOKENIZER_PATH") or os.environ.get(
        "VERL_OPD_TARGET_MODEL"
    ) or os.environ.get("STUDENT_MODEL")
    cache_key = f"{env_val or ''}:{tok_path or ''}:{vocab_size}"
    if cache_key in _INTERNAL_STOP_ID_CACHE:
        return _INTERNAL_STOP_ID_CACHE[cache_key]

    stop_id: int | None = None
    if env_val not in (None, ""):
        try:
            stop_id = int(env_val)
        except ValueError:
            logger.warning("[opd-rollout] invalid VERL_OPD_INTERNAL_STOP_TOKEN_ID=%r", env_val)
            stop_id = None
    elif tok_path:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
            eos = getattr(tok, "eos_token_id", None)
            if isinstance(eos, (list, tuple)):
                eos = eos[0] if eos else None
            if eos is not None:
                stop_id = int(eos)
        except Exception:
            logger.exception("[opd-rollout] failed to resolve internal stop token from tokenizer")

    if stop_id is not None and not (0 <= stop_id < vocab_size):
        logger.warning(
            "[opd-rollout] internal stop token id out of range: stop_id=%s vocab_size=%s",
            stop_id,
            vocab_size,
        )
        stop_id = None
    logger.warning("[opd-rollout] internal stop token id=%s", stop_id)
    _INTERNAL_STOP_ID_CACHE[cache_key] = stop_id
    return stop_id


def _record_trigger_stop_events(
    sampling_metadata: Any,
    output: torch.Tensor,
    internal_stop_output: torch.Tensor,
    student_topk_threshold_by_output: torch.Tensor | None = None,
) -> None:
    req_id_strs = getattr(sampling_metadata, "req_ids", None)
    if req_id_strs is None or not internal_stop_output.any():
        return
    batch_size = int(output.shape[0])
    base_positions = _get_response_base_positions(sampling_metadata, batch_size)
    output_cpu = output.detach().cpu()
    trunc_cpu = internal_stop_output.detach().bool().cpu()
    student_topk_threshold_cpu = (
        student_topk_threshold_by_output.detach().cpu()
        if student_topk_threshold_by_output is not None
        else None
    )
    trace_path = os.environ.get("VERL_OPD_STOP_EVENTS_JSONL")
    file_records: list[dict[str, Any]] = []
    for bi in range(batch_size):
        if bi >= len(req_id_strs):
            continue
        rid_raw = req_id_strs[bi]
        if rid_raw is None:
            continue
        rid = _strip_vllm_suffix(str(rid_raw))
        response_pos = base_positions[bi] if base_positions is not None else 0
        for out_pos in torch.nonzero(trunc_cpu[bi], as_tuple=False).flatten().tolist():
            tok = int(output_cpu[bi, out_pos])
            rec: dict[str, Any] = {
                "request_id": rid,
                "response_pos": int(response_pos + out_pos),
                "spec_pos": int(out_pos),
                "internal_stop_token_id": tok,
            }
            if student_topk_threshold_cpu is not None:
                threshold_val = int(student_topk_threshold_cpu[bi, out_pos])
                if threshold_val > 0:
                    rec["student_rank_teacher_argmax_gt"] = threshold_val
            if os.environ.get("VERL_OPD_MASK_IPC_SOCKET"):
                _send_stop_event_ipc(rid, rec)
            file_records.append(rec)

    if trace_path and file_records and _trace_file_writer_enabled():
        try:
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            with open(trace_path, "a", encoding="utf-8") as f:
                for record in file_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("[opd-rollout] failed to write stop-event JSONL: %s", trace_path)


def _append_trace_events(
    sampling_metadata: Any,
    output: torch.Tensor,
    teacher_output: torch.Tensor,
    req_ids: torch.Tensor,
    pos_in_req: torch.Tensor,
    valid_pos: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor | None,
    trigger_mask: torch.Tensor | None = None,
    takeover_mask_by_output: torch.Tensor | None = None,
    standard_target_by_output: torch.Tensor | None = None,
) -> None:
    if not _trace_enabled():
        return
    req_id_strs = getattr(sampling_metadata, "req_ids", None)
    if req_id_strs is None:
        return
    batch_size = int(output.shape[0])
    base_positions = _get_response_base_positions(sampling_metadata, batch_size)

    with torch.no_grad():
        file_records: list[dict[str, Any]] = []
        emitted = output.ge(0) & output.ne(PLACEHOLDER_TOKEN_ID)
        teacher_cpu = teacher_output.detach().bool().cpu()
        emitted_cpu = emitted.detach().cpu()
        output_cpu = output.detach().cpu()
        req_cpu = req_ids.detach().cpu()
        pos_cpu = pos_in_req.detach().cpu()
        valid_cpu = valid_pos.detach().bool().cpu()

        if _trace_mode() == "mask":
            for bi in range(batch_size):
                rid_raw = req_id_strs[bi] if bi < len(req_id_strs) else None
                if rid_raw is None:
                    continue
                rid = _strip_vllm_suffix(str(rid_raw))
                response_pos = base_positions[bi] if base_positions is not None else 0
                chunk_positions: list[int] = []
                chunk_tokens: list[int] = []
                for out_pos in torch.nonzero(emitted_cpu[bi], as_tuple=False).flatten().tolist():
                    if bool(teacher_cpu[bi, out_pos]):
                        rec = {
                            "response_pos": int(response_pos),
                            "spec_pos": int(out_pos),
                            "emit_token_id": int(output_cpu[bi, out_pos]),
                            "has_logits": False,
                        }
                        _TRACE_EVENTS_BY_REQUEST_ID.setdefault(rid, []).append(rec)
                        chunk_positions.append(int(response_pos))
                        chunk_tokens.append(int(output_cpu[bi, out_pos]))
                    response_pos += 1
                if chunk_positions:
                    file_records.append({
                        "request_id": rid,
                        "positions": chunk_positions,
                        "tokens": chunk_tokens,
                    })
            trace_path = os.environ.get("VERL_OPD_TRACE_JSONL")
            if trace_path and file_records and _trace_file_writer_enabled():
                try:
                    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                    with open(trace_path, "a", encoding="utf-8") as f:
                        for record in file_records:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception:
                    logger.exception("[opd-rollout] failed to write SKD trace jsonl: %s", trace_path)
            return

        if draft_probs is None:
            return

        topk = min(max(_int("VERL_OPD_TRACE_TOPK", 128), 1), int(target_logits.shape[-1]))
        target_logprobs = target_logits.float().log_softmax(dim=-1)
        target_top_logp, target_top_ids = torch.topk(target_logprobs, k=topk, dim=-1)
        target_top_probs = target_top_logp.exp()
        target_mass = target_top_probs.sum(dim=-1).clamp_min(1e-30)
        target_top_q = target_top_probs / target_mass[:, None]
        target_top_logq = target_top_q.clamp_min(1e-30).log()
        student_top_probs = draft_probs.gather(1, target_top_ids).clamp_min(1e-30)
        student_top_mass = student_top_probs.sum(dim=-1).clamp_min(1e-30)
        fkl_teacher_renorm = (target_top_q * (target_top_logq - student_top_probs.log())).sum(dim=-1)

        teacher_argmax = target_logprobs.argmax(dim=-1)
        safe_teacher_argmax = teacher_argmax.clamp_min(0).clamp_max(target_logits.shape[-1] - 1)
        teacher_argmax_logp = target_logprobs.gather(1, safe_teacher_argmax[:, None]).squeeze(1)
        student_argmax_prob = draft_probs.gather(1, safe_teacher_argmax[:, None]).squeeze(1).clamp_min(1e-30)
        student_argmax_logp = student_argmax_prob.log()

        safe_draft = draft_token_ids.long().clamp_min(0).clamp_max(target_logits.shape[-1] - 1)
        teacher_draft_logp = target_logprobs.gather(1, safe_draft[:, None]).squeeze(1)
        student_draft_prob = draft_probs.gather(1, safe_draft[:, None]).squeeze(1).clamp_min(1e-30)
        student_draft_logp = student_draft_prob.log()

        trigger_cpu = trigger_mask.detach().bool().cpu() if trigger_mask is not None else None
        takeover_cpu = takeover_mask_by_output.detach().bool().cpu() if takeover_mask_by_output is not None else None
        standard_target_cpu = (
            standard_target_by_output.detach().bool().cpu() if standard_target_by_output is not None else None
        )

        flat_by_req_pos: dict[tuple[int, int], int] = {}
        for flat_idx in range(int(req_cpu.numel())):
            if not bool(valid_cpu[flat_idx]):
                continue
            flat_by_req_pos[(int(req_cpu[flat_idx]), int(pos_cpu[flat_idx]))] = flat_idx

        for bi in range(batch_size):
            rid_raw = req_id_strs[bi] if bi < len(req_id_strs) else None
            if rid_raw is None:
                continue
            rid = _strip_vllm_suffix(str(rid_raw))
            response_pos = base_positions[bi] if base_positions is not None else 0
            for out_pos in torch.nonzero(emitted_cpu[bi], as_tuple=False).flatten().tolist():
                is_teacher = bool(teacher_cpu[bi, out_pos])
                if not is_teacher:
                    response_pos += 1
                    continue
                flat_idx = flat_by_req_pos.get((bi, int(out_pos)))
                rec: dict[str, Any] = {
                    "response_pos": int(response_pos),
                    "spec_pos": int(out_pos),
                    "emit_token_id": int(output_cpu[bi, out_pos]),
                    "has_logits": flat_idx is not None,
                }
                if flat_idx is not None:
                    rec.update({
                        "draft_token_id": int(draft_token_ids.detach().cpu()[flat_idx]),
                        "teacher_argmax_token_id": int(teacher_argmax.detach().cpu()[flat_idx]),
                        "is_trigger_first_token": bool(trigger_cpu[flat_idx]) if trigger_cpu is not None else False,
                        "is_takeover_continuation": (
                            bool(takeover_cpu[bi, out_pos]) if takeover_cpu is not None else False
                        ),
                        "is_standard_target_emit": (
                            bool(standard_target_cpu[bi, out_pos]) if standard_target_cpu is not None else False
                        ),
                        "teacher_logp_teacher_argmax": float(teacher_argmax_logp.detach().cpu()[flat_idx]),
                        "student_logp_teacher_argmax": float(student_argmax_logp.detach().cpu()[flat_idx]),
                        "student_p_teacher_argmax": float(student_argmax_prob.detach().cpu()[flat_idx]),
                        "teacher_student_logp_gap_argmax": float(
                            (teacher_argmax_logp - student_argmax_logp).detach().cpu()[flat_idx]
                        ),
                        "teacher_logp_draft": float(teacher_draft_logp.detach().cpu()[flat_idx]),
                        "student_logp_draft": float(student_draft_logp.detach().cpu()[flat_idx]),
                        "teacher_topk_mass": float(target_mass.detach().cpu()[flat_idx]),
                        "student_mass_on_teacher_topk": float(student_top_mass.detach().cpu()[flat_idx]),
                        "fkl_topk_teacher_renorm": float(fkl_teacher_renorm.detach().cpu()[flat_idx]),
                    })
                _TRACE_EVENTS_BY_REQUEST_ID.setdefault(rid, []).append(rec)
                file_records.append({"request_id": rid, **rec})
                response_pos += 1
        trace_path = os.environ.get("VERL_OPD_TRACE_JSONL")
        if trace_path and file_records and _trace_file_writer_enabled():
            try:
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                with open(trace_path, "a", encoding="utf-8") as f:
                    for record in file_records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                logger.exception("[opd-rollout] failed to write SKD trace jsonl: %s", trace_path)


def _get_response_base_positions(sampling_metadata: Any, batch_size: int) -> list[int] | None:
    """Return response-relative base positions for the current vLLM sample call."""
    base_positions = getattr(sampling_metadata, "verl_opd_response_base_positions", None)
    if base_positions is None:
        return None
    try:
        if len(base_positions) < batch_size:
            return None
        return [int(x) for x in base_positions[:batch_size]]
    except Exception:
        return None


def _get_pending_mask_rows(sampling_metadata: Any, batch_size: int) -> list[dict[str, list[int] | list[bool] | list[float]]]:
    pending = getattr(sampling_metadata, "verl_opd_pending_mask_rows", None)
    if pending is None or len(pending) < batch_size:
        pending = [{"mask": [], "tokens": [], "actions": [], "teacher_action_logps": []} for _ in range(batch_size)]
        sampling_metadata.verl_opd_pending_mask_rows = pending
    return pending


def _record_token_source_events(
    sampling_metadata: Any,
    batch_size: int,
    device: torch.device,
    output: torch.Tensor,
    is_teacher_per_req: torch.Tensor | None = None,
    is_teacher_per_token: torch.Tensor | None = None,
    req_ids: torch.Tensor | None = None,
    pos_in_req: torch.Tensor | None = None,
    teacher_output_override: torch.Tensor | None = None,
    action_token_by_output: torch.Tensor | None = None,
    teacher_action_logp_by_output: torch.Tensor | None = None,
    valid_vocab_size: int | None = None,
) -> None:
    """Record every emitted token in a speculative chunk.

    A speculative step may emit several accepted draft tokens, so the event
    stream includes every non-placeholder output position.
    """
    req_id_strs = getattr(sampling_metadata, "req_ids", None)
    if req_id_strs is None:
        return

    emitted = output.ge(0) & output.ne(PLACEHOLDER_TOKEN_ID)
    if not emitted.any():
        return

    if teacher_output_override is not None:
        teacher_output = teacher_output_override.detach().bool() & emitted
    else:
        teacher_output = torch.zeros_like(output, dtype=torch.bool)
    if teacher_output_override is None and is_teacher_per_token is not None and req_ids is not None and pos_in_req is not None:
        valid_teacher = (
            is_teacher_per_token
            & (req_ids >= 0)
            & (req_ids < batch_size)
            & (pos_in_req >= 0)
            & (pos_in_req < output.shape[1])
        )
        if valid_teacher.any():
            teacher_output[req_ids[valid_teacher], pos_in_req[valid_teacher]] = True
    elif teacher_output_override is None and is_teacher_per_req is not None:
        teacher_req = is_teacher_per_req.detach().bool()
        teacher_output = emitted & teacher_req[:, None]

    output_cpu = output.detach().cpu()
    emitted_cpu = emitted.detach().cpu()
    teacher_cpu = teacher_output.detach().cpu()
    action_cpu = action_token_by_output.detach().cpu() if action_token_by_output is not None else None
    action_logp_cpu = (
        teacher_action_logp_by_output.detach().cpu() if teacher_action_logp_by_output is not None else None
    )
    base_positions = _get_response_base_positions(sampling_metadata, batch_size)
    use_ipc = bool(os.environ.get("VERL_OPD_MASK_IPC_SOCKET"))
    for i in range(batch_size):
        if i >= len(req_id_strs):
            continue
        rid = req_id_strs[i]
        if rid is None:
            continue
        positions = torch.nonzero(emitted_cpu[i], as_tuple=False).flatten().tolist()
        if not positions:
            continue
        mask_chunk: list[bool] = []
        token_chunk: list[int] = []
        action_chunk: list[int] | None = [] if action_cpu is not None else None
        action_logp_chunk: list[float] | None = [] if action_logp_cpu is not None else None
        position_chunk: list[int] | None = [] if base_positions is not None else None
        for pos in positions:
            tok = int(output_cpu[i, pos])
            if tok == PLACEHOLDER_TOKEN_ID or tok < 0 or (valid_vocab_size is not None and tok >= valid_vocab_size):
                continue
            response_pos = int(base_positions[i]) + len(token_chunk) if base_positions is not None else None
            token_chunk.append(tok)
            mask_chunk.append(bool(teacher_cpu[i, pos]))
            if action_chunk is not None:
                action_chunk.append(int(action_cpu[i, pos]))
            if action_logp_chunk is not None:
                action_logp_chunk.append(float(action_logp_cpu[i, pos]))
            if position_chunk is not None:
                position_chunk.append(int(response_pos))
        if not token_chunk:
            continue
        external_rid = _strip_vllm_suffix(rid)
        if use_ipc:
            pending_rows = _get_pending_mask_rows(sampling_metadata, batch_size)
            pending_rows[i]["mask"].extend(mask_chunk)
            pending_rows[i]["tokens"].extend(token_chunk)
            if action_chunk is not None:
                pending_rows[i]["actions"].extend(action_chunk)
            if action_logp_chunk is not None:
                pending_rows[i]["teacher_action_logps"].extend(action_logp_chunk)
        else:
            _TEACHER_MASK_BY_REQUEST_ID.setdefault(external_rid, []).extend(mask_chunk)
            _TOKEN_EVENTS_BY_REQUEST_ID.setdefault(external_rid, []).extend(token_chunk)
            if position_chunk is not None:
                _POSITION_EVENTS_BY_REQUEST_ID.setdefault(external_rid, []).extend(position_chunk)


def _sample_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    temperature = max(_float("VERL_OPD_DRAFT_TEMPERATURE", 1.0), 0.0)
    top_p = _float("VERL_OPD_DRAFT_TOP_P", 1.0)
    top_k = _int("VERL_OPD_DRAFT_TOP_K", -1)

    logits_f = logits.to(torch.float32)
    if temperature <= 1e-6:
        probs = torch.zeros_like(logits_f)
        token_ids = logits_f.argmax(dim=-1)
        probs.scatter_(1, token_ids[:, None], 1.0)
        return token_ids, probs

    logits_f = logits_f / temperature
    if top_k is not None and top_k > 0 and top_k < logits_f.shape[-1]:
        kth = logits_f.topk(top_k, dim=-1).values[:, -1, None]
        logits_f = logits_f.masked_fill(logits_f < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = logits_f.sort(dim=-1, descending=True)
        sorted_probs = sorted_logits.softmax(dim=-1, dtype=torch.float32)
        remove = sorted_probs.cumsum(dim=-1) > top_p
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(logits_f, float("-inf"))
        logits_f = filtered.scatter(1, sorted_idx, sorted_logits)

    probs = logits_f.softmax(dim=-1, dtype=torch.float32)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)
    row_sum = probs.sum(dim=-1, keepdim=True)
    degenerate = row_sum.squeeze(-1) <= 0
    probs = torch.where(
        degenerate[:, None],
        torch.full_like(probs, 1.0 / max(probs.shape[-1], 1)),
        probs / row_sum.clamp_min(1e-30),
    )

    race = torch.empty_like(probs)
    race.exponential_()
    token_ids = (probs / race).argmax(dim=-1)
    token_ids = torch.where(degenerate, logits.to(torch.float32).argmax(dim=-1), token_ids)
    return token_ids, probs


def _request_layout(
    num_draft_tokens: list[int],
    num_tokens: int,
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(num_draft_tokens)
    lengths = torch.tensor(num_draft_tokens, dtype=torch.long, device=device)
    req_idx_all = torch.arange(batch_size, device=device, dtype=torch.long)
    req_ids = torch.repeat_interleave(req_idx_all, lengths)
    starts = cu_num_draft_tokens.to(device=device, dtype=torch.long) - lengths
    pos_in_req = torch.arange(num_tokens, device=device, dtype=torch.long)
    pos_in_req = pos_in_req - torch.repeat_interleave(starts, lengths)
    valid_pos = (pos_in_req >= 0) & (pos_in_req < (max_spec_len + 1))
    return lengths, req_idx_all, req_ids, pos_in_req, valid_pos


# ---- Reflection-token resolution ----

_REFLECTION_TOKEN_IDS_CACHE: dict[str, torch.Tensor] = {}
_REFLECTION_TOKENS_LOGGED = False


def _get_reflection_token_ids(device: torch.device) -> torch.Tensor:
    global _REFLECTION_TOKENS_LOGGED
    env_val = os.environ.get(
        "RELAY_OPD_REFLECTION_TOKENS",
        "Wait, Wait,wait, wait,But, But,but, but,"
        "Hmm, Hmm, hmm,Actually, Actually,actually, actually,"
        "Hold, Hold,hold, hold,However, However,however, however,"
        "Yet, Yet,yet, yet,Oh, Oh,oh, oh,"
        "Alternatively, Alternatively,No, No,no, no,"
        "Ah, Ah,ah, ah,Oops, Oops,Well, Well",
    )
    cache_key = f"{device}:{env_val}"
    cached = _REFLECTION_TOKEN_IDS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    from transformers import AutoTokenizer
    tok_path = os.environ.get("VERL_OPD_TOKENIZER_PATH") or os.environ.get(
        "VERL_OPD_TARGET_MODEL"
    ) or os.environ.get("STUDENT_MODEL")
    if not tok_path:
        logger.warning("[opd-rollout] no tokenizer path resolvable from env")
        tensor = torch.zeros(0, dtype=torch.long, device=device)
        _REFLECTION_TOKEN_IDS_CACHE[cache_key] = tensor
        return tensor
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    markers = [s for s in env_val.split(",") if s]
    ids: list[int] = []
    skipped: list[str] = []
    for m in markers:
        enc = tok.encode(m, add_special_tokens=False)
        if len(enc) == 1:
            tid = int(enc[0])
            if tid not in ids:
                ids.append(tid)
        else:
            skipped.append(m)
    if not _REFLECTION_TOKENS_LOGGED:
        logger.warning(
            "[opd-rollout] trigger token IDs (n=%d): %s; skipped multi-token: %s",
            len(ids), ids, skipped,
        )
        _REFLECTION_TOKENS_LOGGED = True
    tensor = torch.tensor(ids, dtype=torch.long, device=device)
    _REFLECTION_TOKEN_IDS_CACHE[cache_key] = tensor
    return tensor


# ---- Rejection sample modes ----

def _skd_topk_rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Google-style SKD top-k speculative replacement.

    Draft tokens are accepted while they are in the teacher top-k set. At the
    first rejection for each request, emit a teacher top-k sample and mark only
    that emitted token as teacher-owned for loss routing.
    """
    del kwargs, draft_probs, bonus_token_ids
    device = target_logits.device
    batch_size = len(num_draft_tokens)
    num_tokens, vocab_size = target_logits.shape
    output = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )
    if num_tokens == 0:
        return output

    k = min(max(_int("SKD_ACCEPTANCE_TOPK", 25), 1), vocab_size)
    valid_draft = (draft_token_ids >= 0) & (draft_token_ids < vocab_size)
    if k == 1:
        teacher_argmax = target_logits.argmax(dim=-1)
        accept_mask = teacher_argmax.eq(draft_token_ids.long()) & valid_draft
    else:
        topk_ids = target_logits.topk(k, dim=-1).indices
        teacher_argmax = topk_ids[:, 0]
        accept_mask = topk_ids.eq(draft_token_ids.long()[:, None]).any(dim=-1) & valid_draft

    _, _, req_ids, pos_in_req, valid_pos = _request_layout(
        num_draft_tokens, num_tokens, max_spec_len, cu_num_draft_tokens, device
    )
    first_rej_pos = torch.full((batch_size,), max_spec_len, dtype=torch.long, device=device)
    rejected = torch.nonzero((~accept_mask) & valid_pos, as_tuple=False).flatten()
    if rejected.numel() > 0:
        first_rej_pos.scatter_reduce_(
            dim=0,
            index=req_ids[rejected],
            src=pos_in_req[rejected],
            reduce="amin",
            include_self=True,
        )

    first_for_token = first_rej_pos[req_ids]
    emit_accept = accept_mask & (pos_in_req < first_for_token) & valid_pos
    emit_reject = (~accept_mask) & (pos_in_req == first_for_token) & valid_pos

    if emit_accept.any():
        output[req_ids[emit_accept], pos_in_req[emit_accept]] = draft_token_ids[emit_accept].to(torch.int32)

    reject_flat = torch.nonzero(emit_reject, as_tuple=False).flatten()
    if reject_flat.numel() > 0:
        reject_logits = target_logits.index_select(0, reject_flat)
        reject_probs = reject_logits.softmax(dim=-1, dtype=torch.float32)
        reject_probs = torch.nan_to_num(reject_probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)

        topk_idx = reject_logits.topk(k, dim=-1).indices
        topk_mask = torch.zeros_like(reject_probs)
        topk_mask.scatter_(1, topk_idx, 1.0)
        reject_probs = reject_probs * topk_mask

        reject_draft = draft_token_ids.index_select(0, reject_flat).long()
        valid_reject_draft = (reject_draft >= 0) & (reject_draft < vocab_size)
        if valid_reject_draft.any():
            reject_probs[valid_reject_draft].scatter_(1, reject_draft[valid_reject_draft, None], 0.0)

        row_sum = reject_probs.sum(dim=-1, keepdim=True)
        degenerate = row_sum.squeeze(-1) <= 0
        reject_probs = torch.where(
            degenerate[:, None],
            torch.full_like(reject_probs, 1.0 / max(vocab_size, 1)),
            reject_probs / row_sum.clamp_min(1e-30),
        )
        sampled = torch.multinomial(reject_probs, num_samples=1).squeeze(-1)
        recovered = torch.where(degenerate, teacher_argmax.index_select(0, reject_flat), sampled)
        output[req_ids[reject_flat], pos_in_req[reject_flat]] = recovered.to(torch.int32)

    teacher_output = torch.zeros_like(output, dtype=torch.bool)
    if reject_flat.numel() > 0:
        teacher_output[req_ids[reject_flat], pos_in_req[reject_flat]] = True

    attempted = valid_pos & valid_draft
    _stats_add("calls", 1, device=device)
    _stats_add("requests", batch_size, device=device)
    _stats_add("drafted_tokens", attempted)
    _stats_add("emitted_tokens", emit_accept.sum() + reject_flat.numel(), device=device)
    _stats_add("student_tokens", emit_accept)
    _stats_add("teacher_tokens", reject_flat.numel(), device=device)
    _stats_add("teacher_requests", reject_flat.numel(), device=device)
    _stats_add("accepted_draft_tokens", emit_accept)
    _stats_add("rejected_draft_tokens", reject_flat.numel(), device=device)

    _record_token_source_events(
        sampling_metadata,
        batch_size,
        device,
        output,
        teacher_output_override=teacher_output,
        valid_vocab_size=vocab_size,
    )
    _append_trace_events(
        sampling_metadata,
        output,
        teacher_output,
        req_ids,
        pos_in_req,
        valid_pos,
        draft_token_ids,
        target_logits,
        None,
        trigger_mask=emit_reject,
        takeover_mask_by_output=teacher_output,
        standard_target_by_output=teacher_output,
    )
    return output


def _get_paragraph_boundary_token_ids(device: torch.device) -> torch.Tensor:
    cache_key = f"{device}"
    cached = _PARAGRAPH_BOUNDARY_IDS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    from transformers import AutoTokenizer
    tok_path = os.environ.get("VERL_OPD_TOKENIZER_PATH") or os.environ.get(
        "VERL_OPD_TARGET_MODEL") or os.environ.get("STUDENT_MODEL")
    ids: list[int] = []
    if tok_path:
        try:
            tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
            vocab_size = getattr(tok, "vocab_size", None) or len(tok)
            for tid in range(vocab_size):
                try:
                    s = tok.decode([tid], skip_special_tokens=False)
                except Exception:
                    continue
                if "\n\n" in s:
                    ids.append(tid)
        except Exception:
            logger.exception("[opd-rollout] para_break: failed to scan vocab")
    logger.warning("[opd-rollout] para_break token IDs (n=%d): %s", len(ids), ids[:30])
    tensor = torch.tensor(ids, dtype=torch.long, device=device)
    _PARAGRAPH_BOUNDARY_IDS_CACHE[cache_key] = tensor
    return tensor


def _standard_rejection_sample_output(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
    **kwargs: Any,
) -> torch.Tensor:
    try:
        import vllm.v1.sample.rejection_sampler as rs_mod
    except Exception:
        logger.exception("[opd-rollout] failed to import original vLLM rejection sampler")
        raise
    original = getattr(rs_mod, "_verl_opd_original_rejection_sample", None)
    if original is None:
        raise RuntimeError("[opd-rollout] original vLLM rejection_sample is not available")
    return original(
        draft_token_ids,
        num_draft_tokens,
        max_spec_len,
        cu_num_draft_tokens,
        draft_probs,
        target_logits,
        bonus_token_ids,
        sampling_metadata,
        **kwargs,
    )


def _infer_standard_target_emits(
    output: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Best-effort mask for recovered/bonus target tokens in standard spec."""
    num_tokens = draft_token_ids.shape[0]
    emitted = output.ge(0) & output.ne(PLACEHOLDER_TOKEN_ID)
    if num_tokens == 0:
        return emitted
    _, _, req_ids, pos_in_req, valid_pos = _request_layout(
        num_draft_tokens, num_tokens, max_spec_len, cu_num_draft_tokens, device
    )
    draft_by_pos = torch.full_like(output, PLACEHOLDER_TOKEN_ID)
    valid = valid_pos & (draft_token_ids >= 0)
    if valid.any():
        draft_by_pos[req_ids[valid], pos_in_req[valid]] = draft_token_ids[valid].to(torch.int32)
    lengths = torch.tensor(num_draft_tokens, dtype=torch.long, device=device)
    positions = torch.arange(output.shape[1], dtype=torch.long, device=device)[None, :]
    draft_positions = positions < lengths[:, None]
    bonus_positions = positions == lengths[:, None]
    return emitted & ((draft_positions & output.ne(draft_by_pos)) | bonus_positions)


def _relay_opd_rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
    **kwargs: Any,
) -> torch.Tensor:
    device = target_logits.device
    batch_size = len(num_draft_tokens)
    num_tokens, vocab_size = target_logits.shape

    output = torch.full(
        (batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=device,
    )
    if num_tokens == 0:
        return output

    rollout_mode = os.environ.get("VERL_OPD_ROLLOUT_MODE", "relay").strip().lower()
    if rollout_mode not in {"relay", "trigger_stop"}:
        raise RuntimeError(
            f"Relay sampler requires VERL_OPD_ROLLOUT_MODE=relay or trigger_stop, got {rollout_mode!r}"
        )
    takeover_enabled = rollout_mode == "relay"
    max_takeover_tokens = int(os.environ.get("RELAY_OPD_MAX_TAKEOVER_TOKENS", "256"))
    paragraphs_per_takeover = int(os.environ.get("RELAY_OPD_PARAGRAPHS_PER_TAKEOVER", "3"))
    max_takeovers = _int("RELAY_OPD_MAX_TAKEOVERS", 2 if takeover_enabled else 0)
    if takeover_enabled and max_takeovers <= 0:
        raise RuntimeError(
            "RELAY_OPD_MAX_TAKEOVERS must be positive when takeover is enabled"
        )
    if max_takeover_tokens <= 0:
        raise RuntimeError("RELAY_OPD_MAX_TAKEOVER_TOKENS must be positive")
    if paragraphs_per_takeover < 0:
        raise RuntimeError("RELAY_OPD_PARAGRAPHS_PER_TAKEOVER must be non-negative")

    reflection_token_ids = _get_reflection_token_ids(device)
    _, _, req_ids, pos_in_req, valid_pos = _request_layout(
        num_draft_tokens, num_tokens, max_spec_len, cu_num_draft_tokens, device
    )
    valid_draft = (draft_token_ids >= 0) & (draft_token_ids < vocab_size)
    if reflection_token_ids.numel() == 0:
        emit = valid_pos & valid_draft
        if emit.any():
            output[req_ids[emit], pos_in_req[emit]] = draft_token_ids[emit].to(torch.int32)
        _record_token_source_events(
            sampling_metadata, batch_size, device, output,
            is_teacher_per_token=torch.zeros_like(emit), req_ids=req_ids, pos_in_req=pos_in_req,
            valid_vocab_size=vocab_size,
        )
        return output

    req_id_strs = getattr(sampling_metadata, "req_ids", None)
    in_takeover_per_req = torch.zeros(batch_size, dtype=torch.bool, device=device)
    ext_rids: list[str | None] = [None] * batch_size
    if req_id_strs is not None:
        for bi in range(min(batch_size, len(req_id_strs))):
            rid = req_id_strs[bi]
            if rid is None:
                continue
            ext = _strip_vllm_suffix(rid)
            ext_rids[bi] = ext
            if takeover_enabled and _TAKEOVER_TOKENS_REMAINING.get(ext, 0) > 0:
                in_takeover_per_req[bi] = True

    teacher_output = torch.zeros_like(output, dtype=torch.bool)
    standard_target_output = torch.zeros_like(output, dtype=torch.bool)
    emitted_takeover = torch.zeros_like(output, dtype=torch.bool)

    if in_takeover_per_req.any():
        standard_output = _standard_rejection_sample_output(
            draft_token_ids,
            num_draft_tokens,
            max_spec_len,
            cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            **kwargs,
        )
        # A bonus token is generated beyond the verified speculative prefix.
        # Relay-OPD drops it so every takeover token has an unambiguous source
        # label and the next decoding step resumes from the verified prefix.
        lengths = torch.tensor(num_draft_tokens, dtype=torch.long, device=device)
        valid_bonus_pos = in_takeover_per_req & (lengths >= 0) & (lengths < standard_output.shape[1])
        if valid_bonus_pos.any():
            bonus_rows = torch.nonzero(valid_bonus_pos, as_tuple=False).flatten()
            bonus_pos = lengths[bonus_rows]
            bonus_present = (
                standard_output[bonus_rows, bonus_pos].ge(0)
                & standard_output[bonus_rows, bonus_pos].ne(PLACEHOLDER_TOKEN_ID)
            )
            if bonus_present.any():
                standard_output[bonus_rows[bonus_present], bonus_pos[bonus_present]] = PLACEHOLDER_TOKEN_ID
                _stats_add("relay_takeover_bonus_dropped", int(bonus_present.sum().item()), device=device)
        standard_emitted = standard_output.ge(0) & standard_output.ne(PLACEHOLDER_TOKEN_ID)
        standard_target_output = _infer_standard_target_emits(
            standard_output, draft_token_ids, num_draft_tokens, max_spec_len, cu_num_draft_tokens, device
        )
        output[in_takeover_per_req] = standard_output[in_takeover_per_req]
        emitted_takeover = standard_emitted & in_takeover_per_req[:, None]
        # The Relay-OPD loss treats the whole takeover segment as teacher
        # controlled. Standard speculative decoding is only the accelerator.
        teacher_output |= emitted_takeover

    in_takeover_per_token = in_takeover_per_req[req_ids]
    teacher_argmax = target_logits.argmax(dim=-1)
    teacher_is_reflection = torch.isin(teacher_argmax, reflection_token_ids)
    raw_trigger_candidate = (
        teacher_is_reflection & teacher_argmax.ne(draft_token_ids.long())
        & valid_pos & valid_draft & ~in_takeover_per_token
    )
    trigger_topk = _int("RELAY_OPD_TRIGGER_TOPK", -1)
    if trigger_topk <= 0:
        raise RuntimeError("RELAY_OPD_TRIGGER_TOPK must be positive")
    if draft_probs is None:
        # vLLM's profile-run invokes the rejection sampler with synthetic
        # metadata (no req_ids) and deliberately omits draft probabilities.
        # Real patched requests always carry req_ids and must not take this path.
        if req_id_strs is None:
            return _standard_rejection_sample_output(
                draft_token_ids,
                num_draft_tokens,
                max_spec_len,
                cu_num_draft_tokens,
                draft_probs,
                target_logits,
                bonus_token_ids,
                sampling_metadata,
                **kwargs,
            )
        raise RuntimeError("Relay-OPD trigger detection requires draft_probs")

    gather_ids = teacher_argmax.clamp_min(0).clamp_max(vocab_size - 1)[:, None]
    draft_probs_aligned = draft_probs[:gather_ids.shape[0]]
    k = min(trigger_topk, draft_probs_aligned.shape[-1])
    student_topk_ids = draft_probs_aligned.topk(k=k, dim=-1).indices
    teacher_in_student_topk = student_topk_ids.eq(gather_ids).any(dim=-1)
    stop_candidate = raw_trigger_candidate & ~teacher_in_student_topk
    trigger_topk_threshold = torch.where(
        stop_candidate,
        torch.full((num_tokens,), int(trigger_topk), dtype=torch.int32, device=device),
        torch.zeros((num_tokens,), dtype=torch.int32, device=device),
    )
    _stats_add("relay_reflection_candidates", raw_trigger_candidate)
    _stats_add("relay_divergence_triggers", stop_candidate)
    _stats_add("relay_teacher_argmax_in_student_topk", raw_trigger_candidate & teacher_in_student_topk)

    relay_trigger_candidate = torch.zeros_like(stop_candidate)
    if takeover_enabled and stop_candidate.any():
        relay_trigger_candidate = stop_candidate
        stop_candidate = torch.zeros_like(stop_candidate)
        _stats_add("relay_takeover_triggers", relay_trigger_candidate)

    trigger_candidate = (
        relay_trigger_candidate
        if takeover_enabled
        else torch.zeros_like(raw_trigger_candidate)
    )

    allowed_trigger = trigger_candidate
    allowed_stop = stop_candidate
    INF_POS = max_spec_len + 2
    first_event_pos = torch.full((batch_size,), INF_POS, dtype=torch.long, device=device)
    event_candidate = trigger_candidate | allowed_stop
    if event_candidate.any():
        masked_pos = torch.where(event_candidate, pos_in_req, torch.full_like(pos_in_req, INF_POS))
        first_event_pos.scatter_reduce_(0, req_ids, masked_pos, reduce="amin", include_self=True)

    first_event_for_token = first_event_pos[req_ids]
    emit_student = valid_pos & valid_draft & ~in_takeover_per_token & (pos_in_req < first_event_for_token)
    emit_trigger = allowed_trigger & (pos_in_req == first_event_for_token)
    emit_stop = allowed_stop & (pos_in_req == first_event_for_token)

    if emit_student.any():
        output[req_ids[emit_student], pos_in_req[emit_student]] = draft_token_ids[emit_student].to(torch.int32)
    if emit_trigger.any():
        output[req_ids[emit_trigger], pos_in_req[emit_trigger]] = teacher_argmax[emit_trigger].to(torch.int32)
        teacher_output[req_ids[emit_trigger], pos_in_req[emit_trigger]] = True
    internal_stop_output = torch.zeros_like(output, dtype=torch.bool)
    stop_topk_threshold_by_output = None
    if emit_stop.any():
        stop_id = _get_internal_stop_token_id(vocab_size)
        if stop_id is None:
            _stats_add("trigger_stop_missing_internal_stop_id", emit_stop)
            emit_stop = torch.zeros_like(emit_stop)
        else:
            stop_indices = torch.nonzero(emit_stop, as_tuple=False).flatten()
            event_req_ids = req_ids[stop_indices]
            stopped_token_pos = pos_in_req[stop_indices]
            # Keep the divergent student action in the training response, then
            # emit one private stop token that rollout strips before loss.
            output[event_req_ids, stopped_token_pos] = draft_token_ids[stop_indices].to(torch.int32)
            internal_stop_pos = stopped_token_pos + 1
            can_place_stop = internal_stop_pos < output.shape[1]
            if can_place_stop.any():
                stop_req_ids = event_req_ids[can_place_stop]
                stop_pos = internal_stop_pos[can_place_stop]
                output[stop_req_ids, stop_pos] = int(stop_id)
                internal_stop_output[stop_req_ids, stop_pos] = True
            if (~can_place_stop).any():
                # This should not happen because output has max_spec_len + 1
                # slots, but fall back to the old behavior to force a stop.
                bad_req_ids = event_req_ids[~can_place_stop]
                bad_pos = stopped_token_pos[~can_place_stop]
                output[bad_req_ids, bad_pos] = int(stop_id)
                internal_stop_output[bad_req_ids, bad_pos] = True
                _stats_add("trigger_stop_no_extra_slot", int((~can_place_stop).sum().item()), device=device)

    if internal_stop_output.any():
        stop_topk_threshold_by_output = torch.zeros(output.shape, dtype=torch.int32, device=device)
        stop_indices = torch.nonzero(emit_stop, as_tuple=False).flatten()
        stop_req_ids = req_ids[stop_indices]
        internal_stop_pos = pos_in_req[stop_indices] + 1
        can_place_stop = internal_stop_pos < output.shape[1]
        if can_place_stop.any():
            stop_topk_threshold_by_output[
                stop_req_ids[can_place_stop], internal_stop_pos[can_place_stop]
            ] = trigger_topk_threshold[stop_indices][can_place_stop]
        if (~can_place_stop).any():
            bad_pos = pos_in_req[stop_indices][~can_place_stop]
            bad_req_ids = stop_req_ids[~can_place_stop]
            stop_topk_threshold_by_output[bad_req_ids, bad_pos] = (
                trigger_topk_threshold[stop_indices][~can_place_stop]
            )

    paragraph_boundary_ids = _get_paragraph_boundary_token_ids(device)
    new_trigger_per_req = torch.zeros(batch_size, dtype=torch.bool, device=device)
    trigger_is_boundary_per_req = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if emit_trigger.any():
        new_trigger_per_req[req_ids[emit_trigger]] = True
        if paragraph_boundary_ids.numel() > 0:
            trigger_hit_para = emit_trigger & torch.isin(teacher_argmax, paragraph_boundary_ids)
            if trigger_hit_para.any():
                trigger_is_boundary_per_req[req_ids[trigger_hit_para]] = True

    takeover_terminated = torch.zeros(batch_size, dtype=torch.bool, device=device)
    stopped_after_takeover = torch.zeros(batch_size, dtype=torch.bool, device=device)
    internal_stop_id = (
        _get_internal_stop_token_id(vocab_size)
        if max_takeovers > 0
        else None
    )
    if paragraphs_per_takeover <= 0 and emit_trigger.any() and max_takeovers > 0:
        # Trigger-only ablation: keep the teacher trigger token in the loss, but
        # do not continue with a teacher takeover segment. Count the trigger as
        # a completed takeover so stop1/stop2/... semantics remain comparable.
        for idx in torch.nonzero(emit_trigger, as_tuple=False).flatten().detach().cpu().tolist():
            bi = int(req_ids[idx].detach().cpu().item())
            ext = ext_rids[bi] if 0 <= bi < len(ext_rids) else None
            if ext is None:
                continue
            end_count = _COMPLETED_TAKEOVERS.get(ext, 0) + 1
            if end_count >= max_takeovers:
                _COMPLETED_TAKEOVERS.pop(ext, None)
                if internal_stop_id is None:
                    _stats_add("relay_stop_missing_internal_stop_id", 1, device=device)
                    continue
                stop_pos = int(pos_in_req[idx].detach().cpu().item()) + 1
                if stop_pos >= output.shape[1]:
                    stop_pos = int(pos_in_req[idx].detach().cpu().item())
                    _stats_add("relay_stop_no_extra_slot", 1, device=device)
                output[bi, stop_pos] = int(internal_stop_id)
                teacher_output[bi, stop_pos] = False
                emitted_takeover[bi, stop_pos] = False
                standard_target_output[bi, stop_pos] = False
                internal_stop_output[bi, stop_pos] = True
                stopped_after_takeover[bi] = True
            else:
                _COMPLETED_TAKEOVERS[ext] = end_count

    emitted = output.ge(0) & output.ne(PLACEHOLDER_TOKEN_ID)
    if paragraph_boundary_ids.numel() > 0:
        takeover_boundary_hits = emitted_takeover & torch.isin(output.long(), paragraph_boundary_ids)
    else:
        takeover_boundary_hits = torch.zeros_like(emitted_takeover)
    if emitted_takeover.any():
        for bi in torch.nonzero(in_takeover_per_req, as_tuple=False).flatten().detach().cpu().tolist():
            ext = ext_rids[bi]
            if ext is None:
                continue
            pos = torch.nonzero(emitted_takeover[bi], as_tuple=False).flatten()
            if pos.numel() == 0:
                continue
            cap_cur = _TAKEOVER_TOKENS_REMAINING.get(ext, 0)
            paragraphs_cur = _TAKEOVER_PARAGRAPHS_REMAINING.get(ext, 0)
            if cap_cur <= 0 or paragraphs_cur <= 0:
                continue

            keep_until_pos: int | None = None
            para_seen = 0
            pos_cpu = pos.detach().cpu().tolist()
            para_cpu = takeover_boundary_hits[bi, pos].detach().cpu().tolist()
            for emitted_idx, (out_pos, is_para) in enumerate(zip(pos_cpu, para_cpu), start=1):
                if bool(is_para):
                    para_seen += 1
                if emitted_idx >= cap_cur or paragraphs_cur - para_seen <= 0:
                    keep_until_pos = int(out_pos)
                    break

            if keep_until_pos is None:
                continue
            drop_pos = pos[pos > keep_until_pos]
            if drop_pos.numel() > 0:
                output[bi, drop_pos] = PLACEHOLDER_TOKEN_ID
                teacher_output[bi, drop_pos] = False
                emitted_takeover[bi, drop_pos] = False
                takeover_boundary_hits[bi, drop_pos] = False
                standard_target_output[bi, drop_pos] = False
            takeover_terminated[bi] = True
            if max_takeovers > 0:
                end_count = _COMPLETED_TAKEOVERS.get(ext, 0) + 1
                if end_count >= max_takeovers:
                    _COMPLETED_TAKEOVERS.pop(ext, None)
                    if internal_stop_id is None:
                        _stats_add("relay_stop_missing_internal_stop_id", 1, device=device)
                    else:
                        stop_pos = int(keep_until_pos) + 1
                        if stop_pos >= output.shape[1]:
                            # The fast path normally has an extra non-loss slot.
                            # If a future vLLM shape lacks it, force termination
                            # by sacrificing the final emitted takeover token.
                            stop_pos = int(keep_until_pos)
                            _stats_add("relay_stop_no_extra_slot", 1, device=device)
                        output[bi, stop_pos] = int(internal_stop_id)
                        teacher_output[bi, stop_pos] = False
                        emitted_takeover[bi, stop_pos] = False
                        takeover_boundary_hits[bi, stop_pos] = False
                        standard_target_output[bi, stop_pos] = False
                        internal_stop_output[bi, stop_pos] = True
                        stopped_after_takeover[bi] = True
                else:
                    _COMPLETED_TAKEOVERS[ext] = end_count

    new_trigger_list = new_trigger_per_req.detach().cpu().tolist()
    trigger_is_boundary_list = trigger_is_boundary_per_req.detach().cpu().tolist()
    takeover_list = in_takeover_per_req.detach().cpu().tolist()
    takeover_terminated_list = takeover_terminated.detach().cpu().tolist()
    takeover_emit_counts = emitted_takeover.sum(dim=1).detach().cpu().tolist()
    takeover_boundary_counts = takeover_boundary_hits.sum(dim=1).detach().cpu().tolist()
    for bi in range(batch_size):
        ext = ext_rids[bi]
        if ext is None:
            continue
        if new_trigger_list[bi]:
            paragraphs_left = paragraphs_per_takeover - (1 if trigger_is_boundary_list[bi] else 0)
            if paragraphs_left <= 0:
                _TAKEOVER_TOKENS_REMAINING.pop(ext, None)
                _TAKEOVER_PARAGRAPHS_REMAINING.pop(ext, None)
            else:
                _TAKEOVER_TOKENS_REMAINING[ext] = max_takeover_tokens
                _TAKEOVER_PARAGRAPHS_REMAINING[ext] = paragraphs_left
        elif takeover_list[bi] and takeover_emit_counts[bi] > 0:
            cap_cur = _TAKEOVER_TOKENS_REMAINING.get(ext, 0)
            paragraphs_cur = _TAKEOVER_PARAGRAPHS_REMAINING.get(ext, 0)
            paragraphs_cur -= int(takeover_boundary_counts[bi])
            cap_cur -= int(takeover_emit_counts[bi])
            if takeover_terminated_list[bi] or paragraphs_cur <= 0 or cap_cur <= 0:
                _TAKEOVER_TOKENS_REMAINING.pop(ext, None)
                _TAKEOVER_PARAGRAPHS_REMAINING.pop(ext, None)
            else:
                _TAKEOVER_TOKENS_REMAINING[ext] = cap_cur
                _TAKEOVER_PARAGRAPHS_REMAINING[ext] = paragraphs_cur

    _stats_add("calls", 1, device=device)
    _stats_add("requests", batch_size, device=device)
    _stats_add("drafted_tokens", valid_pos & (draft_token_ids >= 0))
    final_emitted = emitted & ~internal_stop_output
    _stats_add("emitted_tokens", final_emitted)
    _stats_add("student_tokens", final_emitted & ~teacher_output)
    _stats_add("teacher_tokens", teacher_output)
    _stats_add("relay_new_triggers", emit_trigger)
    _stats_add("trigger_stop_events", emit_stop)
    _stats_add("relay_takeover_tokens", emitted_takeover)
    _stats_add("relay_standard_target_tokens", standard_target_output & in_takeover_per_req[:, None])
    _stats_add("relay_paragraph_boundaries", takeover_boundary_hits)
    _stats_add("relay_takeovers_completed", takeover_terminated)
    _stats_add("relay_stopped_after_max_takeovers", stopped_after_takeover)
    action_token_by_output = None
    teacher_action_logp_by_output = None
    if _flag("RELAY_OPD_EXPORT_STUDENT_ACTION", False):
        action_token_by_output = output.clone()
        teacher_action_logp_by_output = torch.full(output.shape, float("nan"), dtype=torch.float32, device=device)
        valid_action = (
            valid_pos
            & valid_draft
            & (req_ids >= 0)
            & (req_ids < batch_size)
            & (pos_in_req >= 0)
            & (pos_in_req < output.shape[1])
        )
        if valid_action.any():
            flat_req = req_ids[valid_action]
            flat_pos = pos_in_req[valid_action]
            teacher_flat = teacher_output[flat_req, flat_pos]
            if teacher_flat.any():
                action_indices = torch.nonzero(valid_action, as_tuple=False).flatten()[teacher_flat]
                action_req = req_ids[action_indices]
                action_pos = pos_in_req[action_indices]
                safe_draft = draft_token_ids[action_indices].long().clamp_min(0).clamp_max(vocab_size - 1)
                teacher_draft_logp = target_logits[action_indices].float().log_softmax(dim=-1).gather(
                    1, safe_draft[:, None]
                ).squeeze(1)
                action_token_by_output[action_req, action_pos] = draft_token_ids[action_indices].to(output.dtype)
                teacher_action_logp_by_output[action_req, action_pos] = teacher_draft_logp.to(torch.float32)
    _record_token_source_events(
        sampling_metadata, batch_size, device, output,
        teacher_output_override=teacher_output,
        action_token_by_output=action_token_by_output,
        teacher_action_logp_by_output=teacher_action_logp_by_output,
        valid_vocab_size=vocab_size,
    )
    _record_trigger_stop_events(
        sampling_metadata,
        output,
        internal_stop_output,
        student_topk_threshold_by_output=stop_topk_threshold_by_output,
    )
    _append_trace_events(
        sampling_metadata,
        output,
        teacher_output,
        req_ids,
        pos_in_req,
        valid_pos,
        draft_token_ids,
        target_logits,
        draft_probs,
        trigger_mask=emit_trigger,
        takeover_mask_by_output=emitted_takeover,
        standard_target_by_output=standard_target_output,
    )
    return output


def _patch_vllm_vocab_check() -> None:
    try:
        from vllm.config.speculative import SpeculativeConfig
    except Exception:
        logger.exception("[opd-rollout] failed to import SpeculativeConfig for vocab patch")
        return
    if getattr(SpeculativeConfig, "_verl_opd_vocab_patch", False):
        return

    def _warn_only(self) -> None:
        target = self.target_model_config.get_vocab_size() if self.target_model_config else None
        draft = self.draft_model_config.get_vocab_size() if self.draft_model_config else None
        if target != draft:
            logger.warning(
                "[opd-rollout] vocab_size mismatch tolerated: target=%s draft=%s",
                target, draft,
            )

    SpeculativeConfig.verify_equal_vocab_size_if_draft_model = _warn_only
    SpeculativeConfig._verl_opd_vocab_patch = True


def _patch_vllm_draft_sampling() -> None:
    try:
        from vllm.v1.spec_decode import llm_base_proposer as proposer_mod
        from vllm.v1.worker import gpu_model_runner as runner_mod
    except Exception:
        logger.exception("[opd-rollout] failed to import vLLM proposer/runner for draft patch")
        return

    cls = proposer_mod.SpecDecodeBaseProposer
    if not getattr(cls, "_verl_opd_draft_patch", False):
        original_greedy_sample = cls._greedy_sample
        original_propose = cls.propose

        def _patched_greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
            if not _flag("VERL_OPD_DRAFT_SAMPLING", True):
                return original_greedy_sample(self, hidden_states)
            try:
                token_ids, probs = _sample_from_logits(self.model.compute_logits(hidden_states))
                if getattr(self, "_verl_opd_collect_draft_probs", False):
                    self._verl_opd_draft_prob_chunks.append(probs.detach())
                return token_ids
            except Exception:
                logger.exception("[opd-rollout] draft sampling failed; falling back to greedy")
                return original_greedy_sample(self, hidden_states)

        def _patched_propose(self, *args, **kwargs):
            collect = _flag("VERL_OPD_COLLECT_DRAFT_PROBS", True)
            if collect:
                self._verl_opd_draft_prob_chunks = []
                self._verl_opd_collect_draft_probs = True
            try:
                draft_token_ids = original_propose(self, *args, **kwargs)
                if collect and getattr(self, "_verl_opd_draft_prob_chunks", None):
                    chunks = self._verl_opd_draft_prob_chunks
                    probs = torch.stack(chunks, dim=1).reshape(-1, chunks[0].shape[-1])
                    self._verl_opd_draft_probs = probs
                else:
                    self._verl_opd_draft_probs = None
                return draft_token_ids
            finally:
                if collect:
                    self._verl_opd_collect_draft_probs = False

        cls._greedy_sample = _patched_greedy_sample
        cls.propose = _patched_propose
        cls._verl_opd_draft_patch = True

    runner_cls = runner_mod.GPUModelRunner
    if not getattr(runner_cls, "_verl_opd_sample_patch", False):
        original_sample = runner_cls._sample

        def _patched_sample(self, logits, spec_decode_metadata):
            if spec_decode_metadata is None:
                # vLLM emits the first response token before speculative
                # decoding starts. Defer its mask event to bookkeeping, where
                # the accepted output tokens and absolute positions are known.
                self._verl_opd_non_spec_sample = True
                self._verl_opd_pending_mask_rows = None
                return original_sample(self, logits, spec_decode_metadata)

            self._verl_opd_non_spec_sample = False
            sampling_metadata = self.input_batch.sampling_metadata
            sampling_metadata.req_ids = list(self.input_batch.req_ids)
            try:
                num_reqs = len(sampling_metadata.req_ids)
                num_tokens_no_spec = self.input_batch.num_tokens_no_spec
                num_prompt_tokens = self.input_batch.num_prompt_tokens
                sampling_metadata.verl_opd_response_base_positions = [
                    max(0, int(num_tokens_no_spec[i]) - int(num_prompt_tokens[i]))
                    for i in range(num_reqs)
                ]
            except Exception:
                raise RuntimeError("failed to compute absolute response positions for OPD mask IPC")
            self.input_batch.update_async_output_token_ids()
            if self.use_async_scheduling and self._draft_token_req_ids is not None:
                draft_token_ids_cpu, _ = self._get_draft_token_ids_cpu()
                self.input_batch.update_async_spec_token_ids(draft_token_ids_cpu)

            draft_probs = None
            if _flag("VERL_OPD_COLLECT_DRAFT_PROBS", True) and hasattr(self, "drafter"):
                draft_probs = getattr(self.drafter, "_verl_opd_draft_probs", None)
            sampler_output = self.rejection_sampler(spec_decode_metadata, draft_probs, logits, sampling_metadata)
            self._verl_opd_pending_mask_rows = getattr(
                sampling_metadata, "verl_opd_pending_mask_rows", None
            )
            sampling_metadata.verl_opd_pending_mask_rows = None
            return sampler_output

        runner_cls._sample = _patched_sample
        runner_cls._verl_opd_sample_patch = True

    if not getattr(runner_cls, "_verl_opd_bookkeeping_patch", False):
        original_bookkeeping_sync = runner_cls._bookkeeping_sync

        def _patched_bookkeeping_sync(self, *args, **kwargs):
            result = original_bookkeeping_sync(self, *args, **kwargs)
            pending_rows = getattr(self, "_verl_opd_pending_mask_rows", None)
            non_spec_sample = getattr(self, "_verl_opd_non_spec_sample", False)
            self._verl_opd_pending_mask_rows = None
            self._verl_opd_non_spec_sample = False
            if not os.environ.get("VERL_OPD_MASK_IPC_SOCKET"):
                return result
            try:
                valid_sampled_token_ids = result[2]
                req_ids_output_copy = result[4]
                for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
                    if req_idx >= len(req_ids_output_copy):
                        continue
                    if not sampled_ids:
                        continue
                    rid = req_ids_output_copy[req_idx]
                    if rid is None:
                        continue
                    sampled = [int(x) for x in sampled_ids]
                    n = len(sampled)
                    if non_spec_sample:
                        row_tokens = sampled
                        row_mask = [False] * n
                        row_actions: list[int] = []
                        row_action_logps: list[float] = []
                    else:
                        if pending_rows is None or req_idx >= len(pending_rows):
                            raise RuntimeError(
                                "[opd-rollout] speculative output has no pending mask row: "
                                f"rid={rid} req_idx={req_idx} sampled_len={n}"
                            )
                        row = pending_rows[req_idx]
                        row_tokens = [int(x) for x in row.get("tokens", [])]
                        row_mask = [bool(x) for x in row.get("mask", [])]
                        row_actions = [int(x) for x in row.get("actions", [])]
                        row_action_logps = [float(x) for x in row.get("teacher_action_logps", [])]
                        if row_tokens[:n] != sampled:
                            raise RuntimeError(
                                "[opd-rollout] pending mask tokens do not match vLLM bookkeeping output: "
                                f"rid={rid} row_head={row_tokens[:min(n, 8)]} "
                                f"sampled_head={sampled[:min(n, 8)]} row_len={len(row_tokens)} sampled_len={n}"
                            )
                    req_state = self.requests.get(rid)
                    if req_state is None:
                        continue
                    end_pos = len(req_state.output_token_ids)
                    start_pos = end_pos - n
                    positions = list(range(start_pos, end_pos))
                    actions = row_actions[:n] if len(row_actions) >= n else None
                    action_logps = row_action_logps[:n] if len(row_action_logps) >= n else None
                    _send_mask_chunk_ipc(
                        _strip_vllm_suffix(rid), row_mask[:n], sampled, positions, actions, action_logps
                    )
            except Exception:
                logger.exception("[opd-rollout] failed to flush deferred mask IPC after vLLM bookkeeping")
                raise
            return result

        runner_cls._bookkeeping_sync = _patched_bookkeeping_sync
        runner_cls._verl_opd_bookkeeping_patch = True


def _patch_vllm_rejection_sample() -> None:
    try:
        import vllm.v1.sample.rejection_sampler as rs_mod
    except Exception:
        logger.exception("[opd-rollout] failed to import vLLM rejection sampler")
        return

    if getattr(rs_mod, "_verl_opd_rejection_patch", False):
        return

    mode_raw = os.environ.get("VERL_OPD_ROLLOUT_MODE", "relay")
    mode = mode_raw.strip().lower()
    fn_map = {
        "skd": _skd_topk_rejection_sample,
        "trigger_stop": _relay_opd_rejection_sample,
        "relay": _relay_opd_rejection_sample,
    }
    if mode not in fn_map:
        raise RuntimeError(
            f"Unknown VERL_OPD_ROLLOUT_MODE={mode_raw!r} (normalized: {mode!r}); "
            f"supported modes: {sorted(fn_map)}"
        )
    rs_mod._verl_opd_original_rejection_sample = rs_mod.rejection_sample
    rs_mod.rejection_sample = fn_map[mode]
    rs_mod._verl_opd_rejection_patch = True
    logger.warning("[opd-rollout] patched vLLM rejection_sample mode=%s", mode)


def apply_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _patch_vllm_vocab_check()
    _patch_vllm_draft_sampling()
    _patch_vllm_rejection_sample()
    _PATCHED = True
    logger.warning("[opd-rollout] vLLM speculative-decoding patches active")
