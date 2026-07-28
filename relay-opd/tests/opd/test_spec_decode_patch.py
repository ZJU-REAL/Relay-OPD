from types import SimpleNamespace

import pytest
import torch

from opd.patches.vllm import speculative_decode


def test_trigger_stop_profile_without_draft_probs_uses_original_sampler(monkeypatch):
    monkeypatch.setenv("VERL_OPD_ROLLOUT_MODE", "trigger_stop")
    monkeypatch.setenv("RELAY_OPD_TRIGGER_TOPK", "5")
    monkeypatch.setattr(
        speculative_decode,
        "_get_reflection_token_ids",
        lambda device: torch.tensor([1], dtype=torch.long, device=device),
    )

    expected = torch.tensor([[7, -1]], dtype=torch.int32)
    calls = []

    def fake_original(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(speculative_decode, "_standard_rejection_sample_output", fake_original)

    output = speculative_decode._relay_opd_rejection_sample(
        draft_token_ids=torch.tensor([0], dtype=torch.int32),
        num_draft_tokens=[1],
        max_spec_len=1,
        cu_num_draft_tokens=torch.tensor([0, 1], dtype=torch.int32),
        draft_probs=None,
        target_logits=torch.zeros((1, 8), dtype=torch.float32),
        bonus_token_ids=torch.tensor([0], dtype=torch.int32),
        sampling_metadata=SimpleNamespace(),
    )

    assert output is expected
    assert len(calls) == 1


def test_trigger_stop_real_request_without_draft_probs_fails(monkeypatch):
    monkeypatch.setenv("VERL_OPD_ROLLOUT_MODE", "trigger_stop")
    monkeypatch.setenv("RELAY_OPD_TRIGGER_TOPK", "5")
    monkeypatch.setattr(
        speculative_decode,
        "_get_reflection_token_ids",
        lambda device: torch.tensor([1], dtype=torch.long, device=device),
    )

    with pytest.raises(RuntimeError, match="Relay-OPD trigger detection requires draft_probs"):
        speculative_decode._relay_opd_rejection_sample(
            draft_token_ids=torch.tensor([0], dtype=torch.int32),
            num_draft_tokens=[1],
            max_spec_len=1,
            cu_num_draft_tokens=torch.tensor([0, 1], dtype=torch.int32),
            draft_probs=None,
            target_logits=torch.zeros((1, 8), dtype=torch.float32),
            bonus_token_ids=torch.tensor([0], dtype=torch.int32),
            sampling_metadata=SimpleNamespace(req_ids=["request-00000000"]),
        )
