from verl.trainer.distillation.losses import get_distillation_loss_settings


def test_main_relay_loss_uses_k1_estimator_without_topk():
    settings = get_distillation_loss_settings("relay_opd")

    assert settings.use_estimator is True
    assert settings.use_topk is False


def test_relay_fkl_ablation_requests_topk_distribution():
    settings = get_distillation_loss_settings("relay_opd_fkl")

    assert settings.use_estimator is False
    assert settings.use_topk is True
