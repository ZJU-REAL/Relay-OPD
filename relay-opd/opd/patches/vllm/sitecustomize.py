import os

# Putting opd/patches/vllm on PYTHONPATH and setting VERL_OPD_VLLM_PATCH=1
# patches vLLM at interpreter startup. Set VERL_OPD_VLLM_PATCH_DEFER=1 to let
# the rollout server apply it after Ray actors are alive.
if os.environ.get("VERL_OPD_VLLM_PATCH") == "1" and os.environ.get("VERL_OPD_VLLM_PATCH_DEFER") != "1":
    from speculative_decode import apply_patches
    apply_patches()
