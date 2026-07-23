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
