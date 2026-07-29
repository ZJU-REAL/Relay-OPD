#!/usr/bin/env python3
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

import cv2
import ray
import torch
import transformers
import vllm


LOCKED_EXPECTED = {
    "math-verify": "0.9.0",
    "numpy": "1.26.4",
    "opencv-python-headless": "4.13.0.92",
    "ray": "2.56.1",
    "torch": "2.11.0",
    "transformers": "5.14.1",
    "vllm": "0.21.0",
}

if importlib.metadata.version("vllm") != "0.21.0":
    raise RuntimeError(
        "Relay-OPD requires vllm==0.21.0 because it patches version-specific "
        "speculative-decoding interfaces"
    )

if os.environ.get("RELAY_OPD_STRICT_VERSIONS", "0") == "1":
    for distribution, expected in LOCKED_EXPECTED.items():
        actual = importlib.metadata.version(distribution)
        if actual != expected:
            raise RuntimeError(f"{distribution}: expected {expected}, found {actual}")

if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"Python 3.12 is required, found {sys.version.split()[0]}")

pip_check = subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    check=False,
    capture_output=True,
    text=True,
)
pip_check_lines = {
    line.strip()
    for line in (pip_check.stdout + pip_check.stderr).splitlines()
    if line.strip() and line.strip() != "No broken requirements found."
}
def is_known_opencv_numpy_conflict(line: str) -> bool:
    return (
        line.startswith("opencv-python-headless ")
        and ' has requirement numpy>=2; python_version >= "3.9", '
        "but you have numpy 1." in line
    )


unexpected_conflicts = {
    line
    for line in pip_check_lines
    if not is_known_opencv_numpy_conflict(line)
}
if unexpected_conflicts:
    raise RuntimeError(
        "Unexpected dependency conflicts:\n" + "\n".join(sorted(unexpected_conflicts))
    )

repo_root = Path(__file__).resolve().parents[1]
patch_dir = repo_root / "opd" / "patches" / "vllm"
sys.path.insert(0, str(patch_dir))
os.environ["VERL_OPD_ROLLOUT_MODE"] = "relay"
os.environ["VERL_OPD_DRAFT_SAMPLING"] = "1"
os.environ["VERL_OPD_COLLECT_DRAFT_PROBS"] = "1"

from speculative_decode import apply_patches

apply_patches()

from vllm.config.speculative import SpeculativeConfig
from vllm.v1.sample import rejection_sampler
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if not all(
    (
        SpeculativeConfig._verl_opd_vocab_patch,
        SpecDecodeBaseProposer._verl_opd_draft_patch,
        GPUModelRunner._verl_opd_sample_patch,
        GPUModelRunner._verl_opd_bookkeeping_patch,
        rejection_sampler._verl_opd_rejection_patch,
    )
):
    raise RuntimeError("Relay-OPD did not patch all required vLLM 0.21.0 interfaces")

from opd.reward.math_reward import compute_score

if compute_score(
    solution_str=r"The final answer is \boxed{2}.",
    ground_truth="2",
) != 1.0:
    raise RuntimeError("Relay-OPD math grader smoke test failed")

require_cuda = os.environ.get("RELAY_OPD_REQUIRE_CUDA", "1") == "1"
if require_cuda:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    x = torch.randn((256, 256), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    if not torch.isfinite(y).all().item():
        raise RuntimeError("CUDA matmul smoke test produced non-finite values")

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
print(f"vLLM: {vllm.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Ray: {ray.__version__}")
print(f"OpenCV: {cv2.__version__}")
print("Relay-OPD vLLM patch: OK")
print("Math grader: OK")
print("CUDA: OK" if require_cuda else "CUDA check: skipped")
