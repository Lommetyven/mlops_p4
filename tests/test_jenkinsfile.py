from pathlib import Path


def test_ai_lab_inference_accepts_an_empty_wandb_run_name():
    jenkinsfile = Path("Jenkinsfile").read_text(encoding="utf-8")

    assert (
        'WANDB_RUN_NAME_B64="$(printf \'%s\' "${WANDB_RUN_NAME:-}" '
        "| base64 | tr -d '\\n')\""
    ) in jenkinsfile
    assert (
        'WANDB_RUN_NAME="$(printf \'%s\' "$WANDB_RUN_NAME_B64" | base64 -d)"'
        in jenkinsfile
    )


def test_pruning_experiment_uses_a_single_gpu_and_archives_results():
    jenkinsfile = Path("Jenkinsfile").read_text(encoding="utf-8")
    pruning_stage = jenkinsfile.split(
        "stage('Run Pruning Experiment on AI Lab')",
        maxsplit=1,
    )[1].split("stage('Prepare Inference Input')", maxsplit=1)[0]

    assert jenkinsfile.count("name: 'RUN_PRUNING_EXPERIMENT'") == 2
    assert "--nodes=1" in pruning_stage
    assert "--gres=gpu:1" in pruning_stage
    assert "scripts/pruning_mode.sh" in pruning_stage
    assert "reports/pruning_experiment.*" in jenkinsfile
    assert "reports/pruning_metrics.png" in jenkinsfile


def test_drift_monitoring_uses_a_single_gpu_and_archives_evidence():
    jenkinsfile = Path("Jenkinsfile").read_text(encoding="utf-8")
    monitoring_stage = jenkinsfile.split(
        "stage('Run Drift Monitoring on AI Lab')",
        maxsplit=1,
    )[1].split("stage('Prepare Inference Input')", maxsplit=1)[0]

    assert jenkinsfile.count("name: 'RUN_DRIFT_MONITORING'") == 2
    assert "--nodes=1" in monitoring_stage
    assert "--gres=gpu:1" in monitoring_stage
    assert "scripts/evidently_monitoring.py" in monitoring_stage
    assert "scripts/monitoring_mode.sh" in monitoring_stage
    assert "reports/evidently/**" in jenkinsfile
    assert "reports/monitoring-slurm-*.out" in jenkinsfile
