pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '20'))
    }

    parameters {
        booleanParam(
            name: 'RUN_DVC_REPRO',
            defaultValue: true,
            description: 'Rebuild processed data and local archives with DVC.'
        )
        booleanParam(
            name: 'RUN_TRAINING',
            defaultValue: true,
            description: 'Run full GRU training and log metrics, hardware, and weights to W&B.'
        )
        booleanParam(
            name: 'PUSH_DVC',
            defaultValue: true,
            description: 'Push DVC cache updates to the configured MinIO remote.'
        )
        booleanParam(
            name: 'UPLOAD_READABLE_ARTIFACTS',
            defaultValue: true,
            description: 'Upload raw, processed, and model files under readable_artifacts/.'
        )
    }

    environment {
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
        PYTHONUNBUFFERED = '1'
        DVC_REMOTE = 'minio'
        AWS_ENDPOINT_URL = 'http://172.24.198.42:9000'
        WANDB_ENTITY = 'tobiasr-aalborg-universitet'
        WANDB_PROJECT = 'MLOps'
    }

    stages {
        stage('Prepare Python') {
            steps {
                sh '''
                    set -eu

                    if command -v python3.13 >/dev/null 2>&1; then
                        PYTHON_BIN=python3.13
                    elif command -v python3 >/dev/null 2>&1; then
                        PYTHON_BIN=python3
                    else
                        PYTHON_BIN=python
                    fi

                    "$PYTHON_BIN" --version
                    "$PYTHON_BIN" -m venv .venv
                    .venv/bin/python -m pip install --upgrade pip
                    .venv/bin/python -m pip install -r requirements.txt
                '''
            }
        }

        stage('Pre-commit') {
            steps {
                sh '''
                    set -eu
                    .venv/bin/python -m pre_commit run --all-files
                '''
            }
        }

        stage('Unit Tests') {
            steps {
                sh '''
                    set -eu
                    mkdir -p reports
                    .venv/bin/python -m pytest -q --junitxml=reports/pytest.xml
                '''
            }
        }

        stage('Configure MinIO') {
            when {
                anyOf {
                    expression { return params.RUN_DVC_REPRO }
                    expression { return params.RUN_TRAINING }
                    expression { return params.PUSH_DVC }
                    expression { return params.UPLOAD_READABLE_ARTIFACTS }
                }
            }
            steps {
                withCredentials([
                    usernamePassword(
                        credentialsId: 'minio-energyconsumption',
                        usernameVariable: 'MINIO_ACCESS_KEY',
                        passwordVariable: 'MINIO_SECRET_KEY'
                    )
                ]) {
                    sh '''
                        set -eu
                        .venv/bin/python -m dvc remote modify --local "$DVC_REMOTE" access_key_id "$MINIO_ACCESS_KEY"
                        .venv/bin/python -m dvc remote modify --local "$DVC_REMOTE" secret_access_key "$MINIO_SECRET_KEY"
                        .venv/bin/python -m dvc remote modify --local "$DVC_REMOTE" endpointurl "$AWS_ENDPOINT_URL"
                        .venv/bin/python -m dvc remote list
                    '''
                }
            }
        }

        stage('Restore DVC Data') {
            when {
                anyOf {
                    expression { return params.RUN_DVC_REPRO }
                    expression { return params.RUN_TRAINING }
                    expression { return params.PUSH_DVC }
                    expression { return params.UPLOAD_READABLE_ARTIFACTS }
                }
            }
            steps {
                sh '''
                    set -eu
                    .venv/bin/python -m dvc pull -r "$DVC_REMOTE"

                    if [ -f data/dvc_archives/raw.tar.gz ]; then
                        .venv/bin/python scripts/archive_paths.py unpack --archive data/dvc_archives/raw.tar.gz --output data
                    fi
                '''
            }
        }

        stage('DVC Repro') {
            when {
                expression { return params.RUN_DVC_REPRO }
            }
            steps {
                sh '''
                    set -eu
                    .venv/bin/python -m dvc repro
                '''
            }
        }

        stage('Train') {
            when {
                expression { return params.RUN_TRAINING }
            }
            steps {
                withCredentials([
                    string(credentialsId: 'wandb-api-key', variable: 'WANDB_API_KEY')
                ]) {
                    sh '''
                        set -eu
                        mkdir -p reports

                        if ! command -v sbatch >/dev/null 2>&1; then
                            echo "sbatch is required on this Jenkins agent to submit scripts/train_mode.sh" >&2
                            exit 1
                        fi

                        sbatch --wait --export=ALL scripts/train_mode.sh
                    '''
                }
            }
        }

        stage('Update Model Archive') {
            when {
                expression { return params.RUN_TRAINING }
            }
            steps {
                sh '''
                    set -eu
                    .venv/bin/python -m dvc repro archive_models
                '''
            }
        }

        stage('DVC Push') {
            when {
                expression { return params.PUSH_DVC }
            }
            steps {
                sh '''
                    set -eu
                    .venv/bin/python -m dvc push -r "$DVC_REMOTE"
                '''
            }
        }

        stage('Readable Artifacts') {
            when {
                expression { return params.UPLOAD_READABLE_ARTIFACTS }
            }
            steps {
                sh '''
                    set -eu
                    .venv/bin/python scripts/upload_readable_artifacts.py --remote-name "$DVC_REMOTE"
                '''
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'reports/pytest.xml'
            archiveArtifacts(
                artifacts: 'dvc.lock,data/dvc_archives/*.tar.gz,data/dvc_archives/readable_artifacts_manifest.json,models/*.pt,reports/slurm-*.out,reports/slurm-*.err',
                allowEmptyArchive: true,
                fingerprint: true
            )
        }
    }
}
