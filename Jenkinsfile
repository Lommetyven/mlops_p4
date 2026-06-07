pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '20'))
    }

    triggers {
        pollSCM('H/5 * * * *')
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
        DVC_REMOTE_URL = 's3://energyconsumption/dvc'
        AWS_ENDPOINT_URL = 'http://172.24.198.42:9000'
        READABLE_ARTIFACTS_BUCKET = 'energyconsumption'
        READABLE_ARTIFACTS_PREFIX = 'readable_artifacts'
        AI_LAB_HOST = 'ailab-fe01.srv.aau.dk'
        AI_LAB_REPO_PATH = '/ceph/home/student.aau.dk/sl38ze/MLOps/mlops_p4'
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
                        credentialsId: 'energyconsumption_minio',
                        usernameVariable: 'MINIO_ACCESS_KEY',
                        passwordVariable: 'MINIO_SECRET_KEY'
                    )
                ]) {
                    sh '''
                        set -eu
                        .venv/bin/python -m dvc remote modify --local "$DVC_REMOTE" url "$DVC_REMOTE_URL"
                        .venv/bin/python -m dvc remote modify --local "$DVC_REMOTE" access_key_id "$MINIO_ACCESS_KEY"
                        .venv/bin/python -m dvc remote modify --local "$DVC_REMOTE" secret_access_key "$MINIO_SECRET_KEY"
                        .venv/bin/python -m dvc remote modify --local "$DVC_REMOTE" endpointurl "$AWS_ENDPOINT_URL"

                        REMOTE_URL="$(.venv/bin/python -m dvc config "remote.$DVC_REMOTE.url")"
                        REMOTE_ENDPOINT="$(.venv/bin/python -m dvc config "remote.$DVC_REMOTE.endpointurl")"
                        if [ "$REMOTE_URL" != "$DVC_REMOTE_URL" ]; then
                            echo "Refusing to continue: DVC remote URL is $REMOTE_URL, expected $DVC_REMOTE_URL" >&2
                            exit 1
                        fi
                        if [ "$REMOTE_ENDPOINT" != "$AWS_ENDPOINT_URL" ]; then
                            echo "Refusing to continue: DVC endpoint is $REMOTE_ENDPOINT, expected $AWS_ENDPOINT_URL" >&2
                            exit 1
                        fi

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
                    export PATH="$PWD/.venv/bin:$PATH"
                    .venv/bin/python -m dvc repro archive_processed
                '''
            }
        }

        stage('Train') {
            when {
                expression { return params.RUN_TRAINING }
            }
            steps {
                withCredentials([
                    sshUserPrivateKey(
                        credentialsId: 'energyconsumption_ai-lab',
                        keyFileVariable: 'AI_LAB_SSH_KEY',
                        usernameVariable: 'AI_LAB_SSH_USER'
                    ),
                    usernamePassword(
                        credentialsId: 'energyconsumption_minio',
                        usernameVariable: 'MINIO_ACCESS_KEY',
                        passwordVariable: 'MINIO_SECRET_KEY'
                    ),
                    string(credentialsId: 'energyconsumption_key', variable: 'WANDB_API_KEY')
                ]) {
                    sh '''
                        set +x
                        set -eu
                        mkdir -p reports

                        MINIO_ACCESS_KEY_B64="$(printf '%s' "$MINIO_ACCESS_KEY" | base64 | tr -d '\n')"
                        MINIO_SECRET_KEY_B64="$(printf '%s' "$MINIO_SECRET_KEY" | base64 | tr -d '\n')"
                        WANDB_API_KEY_B64="$(printf '%s' "$WANDB_API_KEY" | base64 | tr -d '\n')"

                        SSH_OPTS="-i $AI_LAB_SSH_KEY -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
                        REMOTE_ENV="MINIO_ACCESS_KEY_B64='$MINIO_ACCESS_KEY_B64' MINIO_SECRET_KEY_B64='$MINIO_SECRET_KEY_B64' WANDB_API_KEY_B64='$WANDB_API_KEY_B64' DVC_REMOTE='$DVC_REMOTE' DVC_REMOTE_URL='$DVC_REMOTE_URL' AWS_ENDPOINT_URL='$AWS_ENDPOINT_URL' WANDB_ENTITY='$WANDB_ENTITY' WANDB_PROJECT='$WANDB_PROJECT' AI_LAB_REPO_PATH='$AI_LAB_REPO_PATH' PUSH_DVC_PARAM='$PUSH_DVC' UPLOAD_READABLE_ARTIFACTS_PARAM='$UPLOAD_READABLE_ARTIFACTS' READABLE_ARTIFACTS_BUCKET='$READABLE_ARTIFACTS_BUCKET' READABLE_ARTIFACTS_PREFIX='$READABLE_ARTIFACTS_PREFIX'"

                        ssh $SSH_OPTS -l "$AI_LAB_SSH_USER" "$AI_LAB_HOST" "$REMOTE_ENV bash -s" <<'REMOTE_SCRIPT'
set -eu

MINIO_ACCESS_KEY="$(printf '%s' "$MINIO_ACCESS_KEY_B64" | base64 -d)"
MINIO_SECRET_KEY="$(printf '%s' "$MINIO_SECRET_KEY_B64" | base64 -d)"
WANDB_API_KEY="$(printf '%s' "$WANDB_API_KEY_B64" | base64 -d)"
export MINIO_ACCESS_KEY MINIO_SECRET_KEY WANDB_API_KEY
export DVC_REMOTE DVC_REMOTE_URL AWS_ENDPOINT_URL WANDB_ENTITY WANDB_PROJECT

cd "$AI_LAB_REPO_PATH"
git fetch origin main
git checkout main
git pull --ff-only origin main

if command -v python3.13 >/dev/null 2>&1; then
    PYTHON_BIN=python3.13
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    PYTHON_BIN=python
fi

"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python -m dvc remote modify --local "$DVC_REMOTE" url "$DVC_REMOTE_URL"
.venv/bin/python -m dvc remote modify --local "$DVC_REMOTE" access_key_id "$MINIO_ACCESS_KEY"
.venv/bin/python -m dvc remote modify --local "$DVC_REMOTE" secret_access_key "$MINIO_SECRET_KEY"
.venv/bin/python -m dvc remote modify --local "$DVC_REMOTE" endpointurl "$AWS_ENDPOINT_URL"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is not available on $HOSTNAME; Jenkins must SSH to an AAU AI Lab login node." >&2
    exit 1
fi

mkdir -p reports
sbatch --wait --export=ALL scripts/train_mode.sh
.venv/bin/python -m dvc repro archive_models

if [ "$PUSH_DVC_PARAM" = "true" ]; then
    .venv/bin/python -m dvc push -r "$DVC_REMOTE"
fi

if [ "$UPLOAD_READABLE_ARTIFACTS_PARAM" = "true" ]; then
    .venv/bin/python scripts/upload_readable_artifacts.py \
        --remote-name "$DVC_REMOTE" \
        --bucket "$READABLE_ARTIFACTS_BUCKET" \
        --prefix "$READABLE_ARTIFACTS_PREFIX"
fi
REMOTE_SCRIPT

                        scp $SSH_OPTS -o User="$AI_LAB_SSH_USER" "$AI_LAB_HOST:$AI_LAB_REPO_PATH/reports/slurm-*.out" reports/ 2>/dev/null || true
                        scp $SSH_OPTS -o User="$AI_LAB_SSH_USER" "$AI_LAB_HOST:$AI_LAB_REPO_PATH/reports/slurm-*.err" reports/ 2>/dev/null || true
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
                    echo "Model archive is created on AAU AI Lab during the Train stage."
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
                    if [ "$READABLE_ARTIFACTS_BUCKET" != "energyconsumption" ]; then
                        echo "Refusing to upload readable artifacts outside bucket energyconsumption." >&2
                        exit 1
                    fi
                    if [ "$READABLE_ARTIFACTS_PREFIX" != "readable_artifacts" ]; then
                        echo "Refusing to upload readable artifacts outside prefix readable_artifacts." >&2
                        exit 1
                    fi

                    .venv/bin/python scripts/upload_readable_artifacts.py \
                        --remote-name "$DVC_REMOTE" \
                        --bucket "$READABLE_ARTIFACTS_BUCKET" \
                        --prefix "$READABLE_ARTIFACTS_PREFIX"
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
