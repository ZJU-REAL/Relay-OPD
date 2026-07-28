# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
import verl.utils.torch_functional as verl_F
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    response_mask_bool = _to_padded_bool(response_mask)
    distillation_losses_response = distillation_losses[response_mask_bool]
    if distillation_losses_response.numel() == 0:
        zero = distillation_losses.new_zeros(())
        return {
            "distillation/loss_min": Metric(AggregationType.MIN, zero),
            "distillation/loss_max": Metric(AggregationType.MAX, zero),
        }
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def _to_padded_bool(mask: torch.Tensor) -> torch.Tensor:
    if mask.is_nested:
        return mask.bool().to_padded_tensor(False)
    return mask.bool()


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    match config.strategy:
        # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
        case "fsdp" | "veomni":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            match distillation_config.distillation_loss.loss_mode:
                case "forward_kl_topk":
                    distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
                case "forward_kl_topk_teacher_renorm" | "relay_opd_fkl":
                    distillation_loss_fn = fsdp_losses.compute_forward_kl_topk_teacher_renorm
                case loss_mode:
                    raise NotImplementedError(f"Unsupported top-k distillation loss mode for FSDP: {loss_mode}")
        case "megatron":
            if distillation_config.distillation_loss.loss_mode != "forward_kl_topk":
                raise NotImplementedError(
                    "Megatron top-k renormalized KL is not implemented yet; "
                    f"got loss_mode={distillation_config.distillation_loss.loss_mode!r}."
                )
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss

    if distillation_loss_config.loss_mode in {"relay_opd", "relay_opd_fkl"}:
        # Set global_batch_info that ppo_loss() normally sets. Relay-OPD
        # combines its token-level branches before the shared reduction.
        config.global_batch_info["dp_size"] = data["dp_size"]
        config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
        config.global_batch_info["global_batch_size"] = data["global_batch_size"]
        config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor
        distill_loss, distill_metrics = relay_opd_loss(
            config=config, distillation_config=distillation_config,
            model_output=model_output, data=data, dp_group=dp_group,
        )
        distill_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)
        return distill_loss, distill_metrics

    # distillation_loss() uses agg_loss(), which needs the global normalization
    # metadata before ppo_loss() has a chance to populate it.
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
    if not distillation_loss_config.use_task_rewards:
        policy_loss = 0.0

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    return policy_loss, policy_metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    distillation_metrics.update(_compute_relay_teacher_mask_metrics(data=data, response_mask=response_mask))
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        response_mask = _to_padded_bool(response_mask)
        rollout_is_weights = data.get("rollout_is_weights", None)
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        response_mask = _to_padded_bool(response_mask)
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    return distillation_loss, distillation_metrics


def _compute_relay_teacher_mask_metrics(data: TensorDict, response_mask: torch.Tensor) -> dict[str, Metric]:
    teacher_mask = data.get("relay_teacher_mask", None)
    if teacher_mask is None:
        return {}
    if teacher_mask.is_nested:
        teacher_mask = teacher_mask.to_padded_tensor(False)
    if response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(False)
    teacher_mask = teacher_mask.bool() & response_mask.bool()
    valid = response_mask.float().sum().clamp_min(1.0)
    token_ratio = teacher_mask.float().sum() / valid
    seq_valid = response_mask.float().sum(dim=-1).clamp_min(1.0)
    seq_ratio = (teacher_mask.float().sum(dim=-1) / seq_valid).mean()
    seq_has_teacher = teacher_mask.any(dim=-1).float().mean()
    return {
        "distillation/relay_teacher_mask_ratio": Metric(AggregationType.MEAN, token_ratio.detach()),
        "distillation/relay_teacher_mask_seq_ratio": Metric(AggregationType.MEAN, seq_ratio.detach()),
        "distillation/relay_teacher_mask_seq_any": Metric(AggregationType.MEAN, seq_has_teacher.detach()),
    }


@register_distillation_loss(
    DistillationLossSettings(
        names=[
            "forward_kl_topk",
            "forward_kl_topk_teacher_renorm",
        ],
        use_topk=True,
    )
)  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    overlap_count = model_output.get("overlap_count")
    overlap_token_advantage = model_output.get("overlap_token_advantage")
    if overlap_count is not None and overlap_token_advantage is not None:
        overlap_count = no_padding_2_padding(overlap_count, data)
        overlap_token_advantage = no_padding_2_padding(overlap_token_advantage, data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    overlap_metrics = {}
    if overlap_count is not None and overlap_token_advantage is not None:
        assert overlap_count.shape == overlap_token_advantage.shape == response_mask_bool.shape
        valid_overlap_count = overlap_count[response_mask_bool]
        k = distillation_config.distillation_loss.topk
        assert k is not None
        # Diagnostics for tracking teacher/student top-k overlap in OPD, following
        # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016):
        # overlap ratio and average teacher-token KL contribution on overlapped tokens.
        overlap_metrics["distillation/overlap_ratio"] = (valid_overlap_count.float().mean() / k).item()
        overlap_position_mask = response_mask_bool & (overlap_count > 0)
        if overlap_position_mask.any():
            overlap_metrics["distillation/overlap_token_advantage"] = (
                overlap_token_advantage[overlap_position_mask].mean().item()
            )
        else:
            overlap_metrics["distillation/overlap_token_advantage"] = 0.0

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
        **overlap_metrics,
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    response_mask_bool = _to_padded_bool(data["response_mask"])
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss.
    valid_abs_losses = distillation_losses[response_mask_bool].abs()
    abs_loss = valid_abs_losses.mean() if valid_abs_losses.numel() > 0 else distillation_losses.new_zeros(())
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, abs_loss),
    }
    return distillation_losses, metrics


def _ppo_pg_loss_mat(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    config,
    rollout_is_weights=None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Token-level PPO clipped PG loss (no reduction)."""
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)

    neg_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(neg_kl)
    ppo_kl = verl_F.masked_mean(-neg_kl, response_mask)

    pg1 = -advantages * ratio
    pg2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip1 = torch.maximum(pg1, pg2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg2, pg1).float(), response_mask)

    pg3 = -advantages * clip_ratio_c
    clip2 = torch.min(pg3, clip1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip1, pg3).float() * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip2, clip1)
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_losses, metrics


@register_distillation_loss(DistillationLossSettings(names="relay_opd", use_estimator=True))
def compute_relay_opd(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Registry hook; the combined Relay-OPD loss is reduced separately."""
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    return torch.zeros_like(student_log_probs), {}


@register_distillation_loss(DistillationLossSettings(names="relay_opd_fkl", use_topk=True))
def compute_relay_opd_fkl(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Registry hook for the teacher top-k FKL ablation."""
    return compute_relay_opd(config, distillation_config, model_output, data)


def relay_opd_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
    dp_group=None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Combine student-token PG with the configured Relay-OPD teacher branch."""
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    teacher_loss_mode = "fkl" if loss_config.loss_mode == "relay_opd_fkl" else "rkl"
    action_token = loss_config.relay_action_token.lower()
    if action_token not in {"emitted", "student_draft"}:
        raise ValueError(
            f"Unsupported relay_action_token={action_token!r}; "
            "expected 'emitted' or 'student_draft'."
        )

    response_mask = data["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(False)
    response_mask = response_mask.bool()

    teacher_mask = data.get(loss_config.relay_teacher_mask_key, None)
    if teacher_mask is None:
        if loss_config.relay_fallback_all_k1:
            teacher_mask = torch.zeros_like(response_mask, dtype=torch.bool)
        else:
            raise RuntimeError(
                f"{loss_config.relay_teacher_mask_key} not found in data. "
                "Set relay_fallback_all_k1=True to fall back to pure k1 PG."
            )
    if teacher_mask.is_nested:
        teacher_mask = teacher_mask.to_padded_tensor(False)
    teacher_mask = teacher_mask.bool() & response_mask
    student_mask = response_mask & ~teacher_mask

    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)

    # k1 PG branch: use actual-token teacher logprob (fail-fast, no fallback)
    teacher_token_lp = data.get("teacher_token_logprobs", None)
    if teacher_token_lp is None and teacher_loss_mode == "rkl":
        # relay_opd is an estimator loss, so the regular K=1 teacher output is
        # already the log-probability of the emitted trajectory token.
        teacher_token_lp = data.get("teacher_logprobs", None)
    if teacher_token_lp is None:
        raise RuntimeError(
            "Relay-OPD requires the teacher log-probability of each emitted token. "
            "For relay_opd_fkl, set VERL_DISTILLATION_DUAL_LOGPROBS=1 so the "
            "actual-token score is returned alongside top-k probabilities."
        )
    teacher_token_lp = no_padding_2_padding(teacher_token_lp, data)
    if teacher_token_lp.dim() > student_log_probs.dim():
        teacher_token_lp = teacher_token_lp.squeeze(-1)
    if teacher_token_lp.shape != student_log_probs.shape:
        raise RuntimeError(
            f"teacher_token_logprobs shape mismatch: "
            f"{teacher_token_lp.shape=} {student_log_probs.shape=}"
        )

    teacher_action_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    teacher_action_ids = data.get("relay_student_action_ids", None)
    if action_token == "student_draft":
        teacher_action_lp = data.get("relay_student_action_teacher_logprobs", None)
        teacher_action_mask_raw = data.get("relay_student_action_mask", None)
        if teacher_mask.any() and (teacher_action_lp is None or teacher_action_mask_raw is None):
            raise RuntimeError(
                "relay_action_token=student_draft requires relay_student_action_teacher_logprobs "
                "and relay_student_action_mask from rollout IPC."
            )
        if teacher_action_lp is not None and teacher_action_mask_raw is not None:
            if teacher_action_lp.is_nested:
                teacher_action_lp = teacher_action_lp.to_padded_tensor(0.0)
            if teacher_action_mask_raw.is_nested:
                teacher_action_mask_raw = teacher_action_mask_raw.to_padded_tensor(False)
            teacher_action_lp = teacher_action_lp.to(device=student_log_probs.device, dtype=teacher_token_lp.dtype)
            teacher_action_mask = teacher_action_mask_raw.to(device=response_mask.device).bool() & teacher_mask
            if teacher_action_lp.shape != teacher_token_lp.shape or teacher_action_mask.shape != response_mask.shape:
                raise RuntimeError(
                    "Relay-OPD action shape mismatch: "
                    f"{teacher_action_lp.shape=} {teacher_action_mask.shape=} {response_mask.shape=}"
                )
            if teacher_mask.any() and not teacher_action_mask[teacher_mask].any():
                raise RuntimeError("relay_action_token=student_draft saw a teacher mask but no valid action rows.")
            missing_teacher_action = teacher_mask & ~teacher_action_mask
            if missing_teacher_action.any():
                missing = int(missing_teacher_action.sum().detach().cpu().item())
                total = int(teacher_mask.sum().detach().cpu().item())
                raise RuntimeError(
                    "relay_action_token=student_draft requires every teacher-mask token to have "
                    f"a draft action; missing={missing} total_teacher={total}. "
                    "Disable takeover bonus tokens or mask these positions explicitly."
                )
            teacher_token_lp = torch.where(teacher_action_mask, teacher_action_lp, teacher_token_lp)

    k1_losses = kl_penalty(logprob=student_log_probs, ref_logprob=teacher_token_lp, kl_penalty="k1")
    if loss_config.loss_max_clamp is not None:
        k1_losses = k1_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    old_log_prob = data["old_log_probs"]
    if old_log_prob.is_nested:
        old_log_prob = old_log_prob.to_padded_tensor(0.0)

    for k, v in config.global_batch_info.items():
        loss_config.global_batch_info[k] = v

    pg_response_mask = response_mask if teacher_loss_mode == "rkl" else student_mask
    pg_loss_tok, pg_metrics = _ppo_pg_loss_mat(
        old_log_prob=old_log_prob,
        log_prob=student_log_probs,
        advantages=(-k1_losses).detach(),
        response_mask=pg_response_mask,
        config=loss_config,
        rollout_is_weights=data.get("rollout_is_weights", None),
    )

    # Direct top-k FKL branch: computed by logits processor during student forward
    if teacher_loss_mode == "fkl" and "distillation_losses" not in model_output:
        raise RuntimeError(
            "relay_opd_fkl requires model_output['distillation_losses']; "
            "loss must be registered with use_topk=True."
        )
    if "distillation_losses" in model_output:
        fkl_loss_tok = no_padding_2_padding(model_output["distillation_losses"], data)
        fkl_loss_tok = fkl_loss_tok.clamp_min(0.0)
        if loss_config.loss_max_clamp is not None:
            fkl_loss_tok = fkl_loss_tok.clamp_max(loss_config.loss_max_clamp)
    else:
        fkl_loss_tok = torch.zeros_like(pg_loss_tok)

    if teacher_loss_mode == "rkl":
        loss_tok = response_mask.float() * pg_loss_tok
    else:
        loss_tok = (
            student_mask.float() * pg_loss_tok
            + teacher_mask.float() * loss_config.relay_teacher_fkl_coef * fkl_loss_tok
        )

    total_loss = agg_loss(
        loss_mat=loss_tok,
        loss_mask=response_mask,
        loss_agg_mode=config.loss_agg_mode,
        **config.global_batch_info,
    )
    teacher_ratio = teacher_mask.float().sum() / response_mask.float().sum().clamp_min(1)
    student_ratio = student_mask.float().sum() / response_mask.float().sum().clamp_min(1)
    teacher_action_coverage = teacher_action_mask.float().sum() / teacher_mask.float().sum().clamp_min(1)
    teacher_action_mismatch = torch.zeros((), dtype=torch.float32, device=student_log_probs.device)
    if teacher_action_ids is not None:
        if teacher_action_ids.is_nested:
            teacher_action_ids = teacher_action_ids.to_padded_tensor(0)
        teacher_action_ids = teacher_action_ids.to(device=student_log_probs.device)
        responses = data.get("responses", None)
        if responses is not None:
            if responses.is_nested:
                responses = responses.to_padded_tensor(0)
            responses = responses.to(device=student_log_probs.device)
            if responses.shape == teacher_action_ids.shape and teacher_action_mask.any():
                teacher_action_mismatch = (teacher_action_ids != responses).float()[teacher_action_mask].mean()
    pg_loss_mean = verl_F.masked_mean(pg_loss_tok, pg_response_mask) if pg_response_mask.any() else pg_loss_tok.new_zeros(())
    teacher_pg_loss_mean = (
        verl_F.masked_mean(pg_loss_tok, teacher_mask) if teacher_mask.any() else pg_loss_tok.new_zeros(())
    )
    fkl_loss_mean = verl_F.masked_mean(fkl_loss_tok, teacher_mask) if teacher_mask.any() else fkl_loss_tok.new_zeros(())
    k1_adv = (-k1_losses).detach()
    k1_adv_on_student = k1_adv[student_mask] if student_mask.any() else k1_adv.new_zeros(2)

    metrics = {
        "distillation/relay_teacher_mask_ratio": Metric(AggregationType.MEAN, teacher_ratio.detach()),
        "distillation/relay_student_mask_ratio": Metric(AggregationType.MEAN, student_ratio.detach()),
        "distillation/relay_pg_loss": Metric(AggregationType.MEAN, pg_loss_mean.detach()),
        "distillation/relay_teacher_pg_loss": Metric(AggregationType.MEAN, teacher_pg_loss_mean.detach()),
        "distillation/relay_teacher_loss_rkl": Metric(
            AggregationType.MEAN,
            torch.as_tensor(1.0 if teacher_loss_mode == "rkl" else 0.0, device=student_log_probs.device),
        ),
        "distillation/relay_action_token_student_draft": Metric(
            AggregationType.MEAN,
            torch.as_tensor(1.0 if action_token == "student_draft" else 0.0, device=student_log_probs.device),
        ),
        "distillation/relay_teacher_action_coverage": Metric(
            AggregationType.MEAN, teacher_action_coverage.detach()
        ),
        "distillation/relay_teacher_action_mismatch_ratio": Metric(
            AggregationType.MEAN, teacher_action_mismatch.detach()
        ),
        "distillation/abs_loss": Metric(AggregationType.MEAN, k1_losses[response_mask].abs().mean().detach()),
        "distillation/k1_adv_mean_student": Metric(AggregationType.MEAN, k1_adv_on_student.mean().detach()),
        "distillation/k1_adv_std_student": Metric(AggregationType.MEAN, k1_adv_on_student.std(unbiased=False).detach()),
        **{f"distillation/{k[len('actor/'):]}" if k.startswith("actor/") else f"distillation/{k}": v
           for k, v in pg_metrics.items()},
    }
    if teacher_loss_mode == "fkl":
        metrics.update(
            {
                "distillation/relay_teacher_fkl_loss": Metric(
                    AggregationType.MEAN, fkl_loss_mean.detach()
                ),
                "distillation/relay_teacher_fkl_loss_max": Metric(
                    AggregationType.MEAN,
                    fkl_loss_tok[teacher_mask].max().detach()
                    if teacher_mask.any()
                    else fkl_loss_tok.new_zeros(()),
                ),
            }
        )
    return total_loss, metrics
