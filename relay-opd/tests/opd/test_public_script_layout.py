import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "opd" / "scripts"

PUBLIC_ENTRYPOINTS = [
    "data/teacher_trajectories.sh",
    "data/trd_trajectories.sh",
    "baselines/sft.sh",
    "baselines/seqkd.sh",
    "baselines/grpo.sh",
    "baselines/opd.sh",
    "baselines/fastopd/1024.sh",
    "baselines/fastopd/2048.sh",
    "baselines/fastopd/4096.sh",
    "baselines/fastopd/8192.sh",
    "baselines/trd.sh",
    "baselines/skd.sh",
    "relay_opd/train.sh",
    "ablations/trigger_topk/k1.sh",
    "ablations/trigger_topk/k10.sh",
    "ablations/takeover_count/m1.sh",
    "ablations/takeover_count/m3.sh",
    "ablations/takeover_count/m4.sh",
    "ablations/takeover_length/l0.sh",
    "ablations/takeover_length/l1.sh",
    "ablations/takeover_length/l2.sh",
    "ablations/takeover_length/l4.sh",
    "ablations/takeover_length/l5.sh",
    "ablations/takeover_length/l6.sh",
    "ablations/loss/teacher_token_rkl.sh",
    "ablations/loss/student_action_rkl.sh",
    "ablations/loss/teacher_fkl.sh",
    "evaluation/math.sh",
]


def test_public_entrypoints_exist_and_are_executable():
    for relative_path in PUBLIC_ENTRYPOINTS:
        script = SCRIPTS / relative_path
        assert script.is_file(), relative_path
        assert os.access(script, os.X_OK), relative_path


def test_legacy_flat_entrypoints_are_absent():
    assert not list(SCRIPTS.glob("train_*.sh"))
    assert not list(SCRIPTS.glob("prepare_*.sh"))
    assert not (SCRIPTS / "evaluate_math.sh").exists()


def test_ablation_scripts_delegate_to_main_relay_entrypoint():
    for script in (SCRIPTS / "ablations").glob("*/*.sh"):
        source = script.read_text(encoding="utf-8")
        assert "../../relay_opd/train.sh" in source, script
