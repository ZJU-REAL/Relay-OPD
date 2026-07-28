"""Chained mathematical-answer grader as a verl custom reward function.

Use with ``reward.reward_manager.name=remote`` so compute_score runs in a
RewardComputeWorker process, where signal.SIGALRM can be installed safely.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _candidate_grader_paths():
    env_path = os.environ.get("MATH_GRADER_PATH") or os.environ.get("GRADER_PATH")
    if env_path:
        yield env_path
    yield os.path.join(_THIS_DIR, "grader")


def _prepare_grader_path():
    for path in _candidate_grader_paths():
        if path and os.path.exists(os.path.join(path, "openmathinst_utils.py")):
            if path not in sys.path:
                sys.path.insert(0, path)
            return path
    return None


_ACTIVE_GRADER_PATH = _prepare_grader_path()


def _ensure_grader_path():
    global _ACTIVE_GRADER_PATH
    if _ACTIVE_GRADER_PATH and _ACTIVE_GRADER_PATH not in sys.path:
        sys.path.insert(0, _ACTIVE_GRADER_PATH)
        return _ACTIVE_GRADER_PATH
    if _ACTIVE_GRADER_PATH is None:
        _ACTIVE_GRADER_PATH = _prepare_grader_path()
    return _ACTIVE_GRADER_PATH


def _normalize_ground_truth(ground_truth):
    if isinstance(ground_truth, dict):
        for key in ("answer", "ground_truth", "target", "solution"):
            if key in ground_truth:
                return "" if ground_truth[key] is None else str(ground_truth[key])
    if isinstance(ground_truth, (list, tuple)) and ground_truth:
        return "" if ground_truth[0] is None else str(ground_truth[0])
    return "" if ground_truth is None else str(ground_truth)


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    try:
        if _ensure_grader_path() is None:
            raise RuntimeError(
                "Cannot find openmathinst_utils.py. Set MATH_GRADER_PATH to the directory containing it."
            )
        from openmathinst_utils import process_results
        from math_verify import parse, verify

        resp = solution_str or ""
        gt = _normalize_ground_truth(ground_truth)

        ok = (
            process_results(resp, gt, response_extract_from_boxed=True)
            or process_results(
                resp,
                gt,
                response_extract_from_boxed=False,
                response_extract_regex=r"The answer is: (.+)$",
            )
            or verify(parse(f"\\boxed{{{gt}}}"), parse(resp))
        )
        return float(bool(ok))
    except Exception:
        return 0.0
