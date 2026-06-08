def normalizeChoiceValues(values) {
    def choices = ['']
    if (values != null) {
        values.each { value ->
            def choice = value == null ? '' : value.toString().trim()
            if (choice) {
                choices.add(choice)
            }
        }
    }
    return choices.unique()
}

def jobParameterDefinitions(datasetChoices = [''], modelVersionChoices = ['']) {
    def datasets = normalizeChoiceValues(datasetChoices)
    def modelVersions = normalizeChoiceValues(modelVersionChoices)

    return [
        booleanParam(name: 'REFRESH_MINIO_CHOICES', defaultValue: true, description: 'Refresh dataset and model-version dropdown choices from readable MinIO artifacts. Updated choices appear on the next build.'),
        booleanParam(name: 'RUN_DVC_REPRO', defaultValue: true, description: 'Rebuild processed data and local archives with DVC.'),
        booleanParam(name: 'RUN_TRAINING', defaultValue: false, description: 'Run GRU training.'),
        booleanParam(name: 'RUN_INFERENCE', defaultValue: false, description: 'Run Rust inference after producing or restoring the TorchScript model.'),
        booleanParam(name: 'PUSH_DVC', defaultValue: true, description: 'Push DVC cache updates to the configured MinIO remote.'),
        booleanParam(name: 'UPLOAD_READABLE_ARTIFACTS', defaultValue: true, description: 'Upload readable raw, processed, and model files under readable_artifacts/.'),

        booleanParam(name: 'DO_TRAIN', defaultValue: true, description: 'Run the training loop.'),
        booleanParam(name: 'DO_VALIDATE', defaultValue: true, description: 'Run validation during training.'),
        booleanParam(name: 'DO_TEST', defaultValue: true, description: 'Run final test evaluation.'),

        choice(name: 'MODEL_VERSION', choices: modelVersions, description: 'Optional W&B/model version from readable MinIO model artifacts. Blank uses configs/train_config.yaml.'),
        string(name: 'MODEL_HIDDEN_SIZE', defaultValue: '', description: 'Optional GRU hidden size override. Blank uses config.'),
        string(name: 'MODEL_NUM_LAYERS', defaultValue: '', description: 'Optional GRU layer count override. Blank uses config.'),
        string(name: 'EPOCHS', defaultValue: '', description: 'Optional epoch override. Blank uses config.'),
        string(name: 'BATCH_SIZE', defaultValue: '', description: 'Optional batch size override. Blank uses config.'),
        string(name: 'SEQUENCE_LENGTH', defaultValue: '', description: 'Optional sequence length override. Blank uses config.'),
        string(name: 'LEARNING_RATE', defaultValue: '', description: 'Optional learning rate override. Blank uses config.'),
        string(name: 'WEIGHT_DECAY', defaultValue: '', description: 'Optional weight decay override. Blank uses config.'),
        choice(name: 'PRECISION_MODE', choices: ['float32', 'amp_float16', 'amp_bfloat16'], description: 'Training precision mode.'),

        choice(name: 'DATASET_PATH', choices: datasets, description: 'Optional processed dataset from readable MinIO artifacts. Blank uses config.'),
        string(name: 'VALIDATION_SPLIT', defaultValue: '', description: 'Optional validation split, e.g. 0.2. Blank uses config.'),
        string(name: 'TEST_SPLIT', defaultValue: '', description: 'Optional test split, e.g. 0.1. Blank uses config.'),
        string(name: 'RANDOM_SEED', defaultValue: '', description: 'Optional random seed override. Blank uses config.'),

        choice(name: 'TRAIN_RUNNER', choices: ['AI_LAB', 'DAKI_WORKER'], description: 'Where training runs.'),
        choice(name: 'AI_LAB_NODES', choices: ['1', '2'], description: 'AI Lab Slurm node count. Use 1 unless multi-node resources are available.'),
        choice(name: 'AI_LAB_GPUS', choices: ['4', '3', '2'], description: 'AI Lab Slurm GPU count.'),
        choice(name: 'AI_LAB_CPUS', choices: ['8', '1', '2', '3', '4', '5', '6', '7', '9', '10', '11', '12', '13', '14', '15'], description: 'AI Lab Slurm CPUs per task.'),
        choice(name: 'AI_LAB_TIME_LIMIT', choices: ['04:00:00', '00:30:00', '01:00:00', '01:30:00', '02:00:00', '02:30:00', '03:00:00', '03:30:00'], description: 'AI Lab Slurm max wall time.'),

        string(name: 'WANDB_RUN_NAME', defaultValue: '', description: 'Optional W&B run name. Blank lets W&B choose.'),
        booleanParam(name: 'CARBON_TRACKING', defaultValue: true, description: 'Enable CarbonTracker.'),
        booleanParam(name: 'HARDWARE_TRACKING', defaultValue: true, description: 'Log GPU utilization, GPU memory, GPU temperature, GPU power, CPU, RAM, runtime, and energy/carbon estimates where available.')
    ]
}

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
        booleanParam(name: 'REFRESH_MINIO_CHOICES', defaultValue: true, description: 'Refresh dataset and model-version dropdown choices from readable MinIO artifacts. Updated choices appear on the next build.')
        booleanParam(name: 'RUN_DVC_REPRO', defaultValue: true, description: 'Rebuild processed data and local archives with DVC.')
        booleanParam(name: 'RUN_TRAINING', defaultValue: false, description: 'Run GRU training.')
        booleanParam(name: 'RUN_INFERENCE', defaultValue: false, description: 'Run Rust inference after producing or restoring the TorchScript model.')
        booleanParam(name: 'PUSH_DVC', defaultValue: true, description: 'Push DVC cache updates to the configured MinIO remote.')
        booleanParam(name: 'UPLOAD_READABLE_ARTIFACTS', defaultValue: true, description: 'Upload readable raw, processed, and model files under readable_artifacts/.')

        booleanParam(name: 'DO_TRAIN', defaultValue: true, description: 'Run the training loop.')
        booleanParam(name: 'DO_VALIDATE', defaultValue: true, description: 'Run validation during training.')
        booleanParam(name: 'DO_TEST', defaultValue: true, description: 'Run final test evaluation.')

        choice(name: 'MODEL_VERSION', choices: [''], description: 'Optional W&B/model version from readable MinIO model artifacts. Blank uses configs/train_config.yaml.')
        string(name: 'MODEL_HIDDEN_SIZE', defaultValue: '', description: 'Optional GRU hidden size override. Blank uses config.')
        string(name: 'MODEL_NUM_LAYERS', defaultValue: '', description: 'Optional GRU layer count override. Blank uses config.')
        string(name: 'EPOCHS', defaultValue: '', description: 'Optional epoch override. Blank uses config.')
        string(name: 'BATCH_SIZE', defaultValue: '', description: 'Optional batch size override. Blank uses config.')
        string(name: 'SEQUENCE_LENGTH', defaultValue: '', description: 'Optional sequence length override. Blank uses config.')
        string(name: 'LEARNING_RATE', defaultValue: '', description: 'Optional learning rate override. Blank uses config.')
        string(name: 'WEIGHT_DECAY', defaultValue: '', description: 'Optional weight decay override. Blank uses config.')
        choice(name: 'PRECISION_MODE', choices: ['float32', 'amp_float16', 'amp_bfloat16'], description: 'Training precision mode.')

        choice(name: 'DATASET_PATH', choices: [''], description: 'Optional processed dataset from readable MinIO artifacts. Blank uses config.')
        string(name: 'VALIDATION_SPLIT', defaultValue: '', description: 'Optional validation split, e.g. 0.2. Blank uses config.')
        string(name: 'TEST_SPLIT', defaultValue: '', description: 'Optional test split, e.g. 0.1. Blank uses config.')
        string(name: 'RANDOM_SEED', defaultValue: '', description: 'Optional random seed override. Blank uses config.')

        choice(name: 'TRAIN_RUNNER', choices: ['AI_LAB', 'DAKI_WORKER'], description: 'Where training runs.')
        choice(name: 'AI_LAB_NODES', choices: ['1', '2'], description: 'AI Lab Slurm node count. Use 1 unless multi-node resources are available.')
        choice(name: 'AI_LAB_GPUS', choices: ['4', '3', '2'], description: 'AI Lab Slurm GPU count.')
        choice(name: 'AI_LAB_CPUS', choices: ['8', '1', '2', '3', '4', '5', '6', '7', '9', '10', '11', '12', '13', '14', '15'], description: 'AI Lab Slurm CPUs per task.')
        choice(
            name: 'AI_LAB_TIME_LIMIT',
            choices: ['04:00:00', '00:30:00', '01:00:00', '01:30:00', '02:00:00', '02:30:00', '03:00:00', '03:30:00'],
            description: 'AI Lab Slurm max wall time.'
        )

        string(name: 'WANDB_RUN_NAME', defaultValue: '', description: 'Optional W&B run name. Blank lets W&B choose.')
        booleanParam(name: 'CARBON_TRACKING', defaultValue: true, description: 'Enable CarbonTracker.')
        booleanParam(name: 'HARDWARE_TRACKING', defaultValue: true, description: 'Log GPU utilization, GPU memory, GPU temperature, GPU power, CPU, RAM, runtime, and energy/carbon estimates where available.')
    }

    environment {
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
        PYTHONUNBUFFERED = '1'
        DVC_REMOTE = 'minio'
        DVC_REMOTE_URL = 's3://energyconsumption/dvc'
        AWS_ENDPOINT_URL = 'http://172.24.198.42:9000'
        READABLE_ARTIFACTS_BUCKET = 'energyconsumption'
        READABLE_ARTIFACTS_PREFIX = 'readable_artifacts'
        DAKI_DOCKER_REGISTRY = '172.24.198.42:5000'
        RUST_TORCH_DOCKER_IMAGE = '172.24.198.42:5000/mlops-p4/rust-inference:latest'
        AI_LAB_HOST = 'ailab-fe01.srv.aau.dk'
        AI_LAB_REPO_PATH = '/ceph/home/student.aau.dk/sl38ze/MLOps/mlops_p4'
        WANDB_ENTITY = 'tobiasr-aalborg-universitet'
        WANDB_PROJECT = 'MLOps'
        TRAIN_CONFIG_PATH = 'reports/runtime_train_config.yaml'
        MONITORING_CONFIG_PATH = 'reports/runtime_monitoring_config.yaml'
        TRAIN_DISTRIBUTED = 'true'
    }

    stages {
        stage('Apply Parameter Defaults') {
            steps {
                script {
                    env.REFRESH_MINIO_CHOICES = "${params.REFRESH_MINIO_CHOICES == null ? true : params.REFRESH_MINIO_CHOICES}"
                    env.RUN_DVC_REPRO = "${params.RUN_DVC_REPRO == null ? true : params.RUN_DVC_REPRO}"
                    env.RUN_TRAINING = "${params.RUN_TRAINING == null ? false : params.RUN_TRAINING}"
                    env.RUN_INFERENCE = "${params.RUN_INFERENCE == null ? false : params.RUN_INFERENCE}"
                    env.PUSH_DVC = "${params.PUSH_DVC == null ? true : params.PUSH_DVC}"
                    env.UPLOAD_READABLE_ARTIFACTS = "${params.UPLOAD_READABLE_ARTIFACTS == null ? true : params.UPLOAD_READABLE_ARTIFACTS}"

                    env.DO_TRAIN = "${params.DO_TRAIN == null ? true : params.DO_TRAIN}"
                    env.DO_VALIDATE = "${params.DO_VALIDATE == null ? true : params.DO_VALIDATE}"
                    env.DO_TEST = "${params.DO_TEST == null ? true : params.DO_TEST}"

                    env.MODEL_VERSION = params.MODEL_VERSION ?: ''
                    env.MODEL_HIDDEN_SIZE = params.MODEL_HIDDEN_SIZE ?: ''
                    env.MODEL_NUM_LAYERS = params.MODEL_NUM_LAYERS ?: ''
                    env.EPOCHS = params.EPOCHS ?: ''
                    env.BATCH_SIZE = params.BATCH_SIZE ?: ''
                    env.SEQUENCE_LENGTH = params.SEQUENCE_LENGTH ?: ''
                    env.LEARNING_RATE = params.LEARNING_RATE ?: ''
                    env.WEIGHT_DECAY = params.WEIGHT_DECAY ?: ''
                    env.PRECISION_MODE = params.PRECISION_MODE ?: 'float32'

                    env.DATASET_PATH = params.DATASET_PATH ?: ''
                    env.VALIDATION_SPLIT = params.VALIDATION_SPLIT ?: ''
                    env.TEST_SPLIT = params.TEST_SPLIT ?: ''
                    env.RANDOM_SEED = params.RANDOM_SEED ?: ''

                    env.TRAIN_RUNNER = params.TRAIN_RUNNER ?: 'AI_LAB'
                    env.TRAIN_DISTRIBUTED = env.TRAIN_RUNNER == 'AI_LAB' ? 'true' : 'false'
                    env.AI_LAB_NODES = params.AI_LAB_NODES ?: '1'
                    env.AI_LAB_GPUS = params.AI_LAB_GPUS ?: '4'
                    env.AI_LAB_CPUS = params.AI_LAB_CPUS ?: '8'
                    env.AI_LAB_TIME_LIMIT = params.AI_LAB_TIME_LIMIT ?: '04:00:00'

                    env.WANDB_RUN_NAME = params.WANDB_RUN_NAME ?: ''
                    env.CARBON_TRACKING = "${params.CARBON_TRACKING == null ? true : params.CARBON_TRACKING}"
                    env.HARDWARE_TRACKING = "${params.HARDWARE_TRACKING == null ? true : params.HARDWARE_TRACKING}"
                }
            }
        }

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
                    expression { return env.REFRESH_MINIO_CHOICES == 'true' }
                    expression { return env.RUN_DVC_REPRO == 'true' }
                    expression { return env.RUN_TRAINING == 'true' }
                    expression { return env.RUN_INFERENCE == 'true' }
                    expression { return env.PUSH_DVC == 'true' }
                    expression { return env.UPLOAD_READABLE_ARTIFACTS == 'true' }
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
                        export AWS_ACCESS_KEY_ID="$MINIO_ACCESS_KEY"
                        export AWS_SECRET_ACCESS_KEY="$MINIO_SECRET_KEY"

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

        stage('Refresh MinIO Parameter Choices') {
            when {
                expression { return env.REFRESH_MINIO_CHOICES == 'true' }
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
                        mkdir -p reports
                        export AWS_ACCESS_KEY_ID="$MINIO_ACCESS_KEY"
                        export AWS_SECRET_ACCESS_KEY="$MINIO_SECRET_KEY"
                        .venv/bin/python scripts/list_minio_parameter_choices.py \
                            --remote-name "$DVC_REMOTE" \
                            --bucket "$READABLE_ARTIFACTS_BUCKET" \
                            --prefix "$READABLE_ARTIFACTS_PREFIX" \
                            --output reports/minio_parameter_choices.json \
                            --datasets-output reports/minio_dataset_choices.txt \
                            --model-versions-output reports/minio_model_version_choices.txt
                    '''
                    script {
                        def datasets = readFile('reports/minio_dataset_choices.txt').split('\n')
                        def modelVersions = readFile('reports/minio_model_version_choices.txt').split('\n')
                        properties([
                            buildDiscarder(logRotator(numToKeepStr: '20')),
                            pipelineTriggers([pollSCM('H/5 * * * *')]),
                            parameters(jobParameterDefinitions(
                                datasets,
                                modelVersions
                            ))
                        ])
                        echo "Refreshed Jenkins dataset choices: ${normalizeChoiceValues(datasets).size() - 1}"
                        echo "Refreshed Jenkins model-version choices: ${normalizeChoiceValues(modelVersions).size() - 1}"
                    }
                }
            }
        }

        stage('Show MinIO Context') {
            when {
                anyOf {
                    expression { return env.RUN_TRAINING == 'true' }
                    expression { return env.RUN_INFERENCE == 'true' }
                    expression { return env.UPLOAD_READABLE_ARTIFACTS == 'true' }
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
                        export AWS_ACCESS_KEY_ID="$MINIO_ACCESS_KEY"
                        export AWS_SECRET_ACCESS_KEY="$MINIO_SECRET_KEY"
                        .venv/bin/python - <<'PY'
import os
import s3fs

bucket = os.environ["READABLE_ARTIFACTS_BUCKET"]
readable_prefix = os.environ["READABLE_ARTIFACTS_PREFIX"].strip("/")
endpoint_url = os.environ["AWS_ENDPOINT_URL"]
fs = s3fs.S3FileSystem(
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    client_kwargs={"endpoint_url": endpoint_url},
)
for label, prefix in {
    "DVC cache": f"{bucket}/dvc",
    "Readable datasets": f"{bucket}/{readable_prefix}/processed",
    "Readable models": f"{bucket}/{readable_prefix}/models",
}.items():
    print(f"\\n{label}: s3://{prefix}")
    try:
        for path in fs.ls(prefix, detail=False)[:20]:
            print(f"  {path}")
    except FileNotFoundError:
        print("  not found yet")
PY
                    '''
                }
            }
        }

        stage('Restore DVC Data') {
            when {
                anyOf {
                    expression { return env.RUN_DVC_REPRO == 'true' }
                    expression { return env.RUN_TRAINING == 'true' }
                    expression { return env.RUN_INFERENCE == 'true' }
                    expression { return env.PUSH_DVC == 'true' }
                    expression { return env.UPLOAD_READABLE_ARTIFACTS == 'true' }
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
                expression { return env.RUN_DVC_REPRO == 'true' }
            }
            steps {
                sh '''
                    set -eu
                    export PATH="$PWD/.venv/bin:$PATH"
                    .venv/bin/python -m dvc repro archive_processed
                '''
            }
        }

        stage('Generate Runtime Config') {
            when {
                anyOf {
                    expression { return env.RUN_TRAINING == 'true' }
                    expression { return env.RUN_INFERENCE == 'true' }
                }
            }
            steps {
                sh '''
                    set -eu
                    mkdir -p reports
                    .venv/bin/python scripts/write_runtime_config.py \
                        --output "$TRAIN_CONFIG_PATH" \
                        --monitoring-output "$MONITORING_CONFIG_PATH"
                    echo "Effective runtime training config:"
                    sed -n '1,180p' "$TRAIN_CONFIG_PATH"
                    echo "Effective runtime monitoring config:"
                    sed -n '1,120p' "$MONITORING_CONFIG_PATH"
                '''
            }
        }

        stage('Train on DAKI Worker') {
            when {
                allOf {
                    expression { return env.RUN_TRAINING == 'true' }
                    expression { return env.TRAIN_RUNNER == 'DAKI_WORKER' }
                }
            }
            steps {
                withCredentials([
                    string(credentialsId: 'energyconsumption_key', variable: 'WANDB_API_KEY')
                ]) {
                    sh '''
                        set +x
                        set -eu
                        export WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT
                        .venv/bin/python main.py --config "$TRAIN_CONFIG_PATH"
                    '''
                }
            }
        }

        stage('Train on AI Lab') {
            when {
                allOf {
                    expression { return env.RUN_TRAINING == 'true' }
                    expression { return env.TRAIN_RUNNER == 'AI_LAB' }
                }
            }
            steps {
                withCredentials([
                    sshUserPrivateKey(
                        credentialsId: 'energyconsumption_ai-lab',
                        keyFileVariable: 'AI_LAB_SSH_KEY',
                        usernameVariable: 'AI_LAB_SSH_USER'
                    ),
                    string(credentialsId: 'energyconsumption_key', variable: 'WANDB_API_KEY')
                ]) {
                    sh '''
                        set +x
                        set -eu
                        mkdir -p reports

                        WANDB_API_KEY_B64="$(printf '%s' "$WANDB_API_KEY" | base64 | tr -d '\n')"

                        SSH_OPTS="-i $AI_LAB_SSH_KEY -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
                        REMOTE_ENV="WANDB_API_KEY_B64='$WANDB_API_KEY_B64' WANDB_ENTITY='$WANDB_ENTITY' WANDB_PROJECT='$WANDB_PROJECT' AI_LAB_REPO_PATH='$AI_LAB_REPO_PATH' TRAIN_CONFIG_PATH='$TRAIN_CONFIG_PATH' TRAIN_DISTRIBUTED='$TRAIN_DISTRIBUTED' AI_LAB_NODES='$AI_LAB_NODES' AI_LAB_GPUS='$AI_LAB_GPUS' AI_LAB_CPUS='$AI_LAB_CPUS' AI_LAB_TIME_LIMIT='$AI_LAB_TIME_LIMIT'"

                        tar -czf reports/ai_lab_code.tar.gz \
                            configs \
                            data_insight \
                            monitoring \
                            scripts \
                            train \
                            reports/runtime_train_config.yaml \
                            reports/runtime_monitoring_config.yaml \
                            dvc.lock \
                            dvc.yaml \
                            main.py \
                            pyproject.toml \
                            requirements.txt
                        ssh $SSH_OPTS -l "$AI_LAB_SSH_USER" "$AI_LAB_HOST" "mkdir -p '$AI_LAB_REPO_PATH/reports'"
                        scp $SSH_OPTS -o User="$AI_LAB_SSH_USER" reports/ai_lab_code.tar.gz "$AI_LAB_HOST:$AI_LAB_REPO_PATH/reports/"

                        ssh $SSH_OPTS -l "$AI_LAB_SSH_USER" "$AI_LAB_HOST" "$REMOTE_ENV bash -s" <<'REMOTE_SCRIPT'
set -eu

WANDB_API_KEY="$(printf '%s' "$WANDB_API_KEY_B64" | base64 -d)"
export WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT TRAIN_CONFIG_PATH TRAIN_DISTRIBUTED

cd "$AI_LAB_REPO_PATH"
tar -xzf reports/ai_lab_code.tar.gz
rm -f reports/ai_lab_code.tar.gz

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

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is not available on $HOSTNAME; Jenkins must SSH to an AAU AI Lab login node." >&2
    exit 1
fi

mkdir -p data models reports
REMOTE_SCRIPT

                        tar -czf reports/ai_lab_data.tar.gz data
                        scp $SSH_OPTS -o User="$AI_LAB_SSH_USER" reports/ai_lab_data.tar.gz "$AI_LAB_HOST:$AI_LAB_REPO_PATH/reports/"

                        ssh $SSH_OPTS -l "$AI_LAB_SSH_USER" "$AI_LAB_HOST" "$REMOTE_ENV bash -s" <<'REMOTE_SCRIPT'
set -eu

WANDB_API_KEY="$(printf '%s' "$WANDB_API_KEY_B64" | base64 -d)"
export WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT TRAIN_CONFIG_PATH TRAIN_DISTRIBUTED

cd "$AI_LAB_REPO_PATH"
tar -xzf reports/ai_lab_data.tar.gz
rm -f reports/ai_lab_data.tar.gz
mkdir -p models reports
sbatch \
    --wait \
    --nodes="${AI_LAB_NODES}" \
    --gres="gpu:${AI_LAB_GPUS}" \
    --cpus-per-task="${AI_LAB_CPUS}" \
    --time="${AI_LAB_TIME_LIMIT}" \
    --export=ALL,TRAIN_CONFIG_PATH="$TRAIN_CONFIG_PATH",TRAIN_DISTRIBUTED="$TRAIN_DISTRIBUTED",TORCHRUN_NNODES="$AI_LAB_NODES",TORCHRUN_NPROC_PER_NODE="$AI_LAB_GPUS" \
    scripts/train_mode.sh
tar --exclude=reports/ai_lab_results.tar.gz -czf reports/ai_lab_results.tar.gz models reports
REMOTE_SCRIPT

                        scp $SSH_OPTS -o User="$AI_LAB_SSH_USER" "$AI_LAB_HOST:$AI_LAB_REPO_PATH/reports/ai_lab_results.tar.gz" reports/
                        tar -xzf reports/ai_lab_results.tar.gz
                    '''
                }
            }
        }

        stage('Prepare Inference Input') {
            when {
                expression { return env.RUN_INFERENCE == 'true' }
            }
            steps {
                sh '''
                    set -eu
                    mkdir -p reports
                    .venv/bin/python scripts/extract_inference_window.py \
                        --config "$TRAIN_CONFIG_PATH" \
                        --output reports/inference_window.csv
                    echo "Prepared inference input:"
                    sed -n '1,8p' reports/inference_window.csv
                '''
            }
        }

        stage('Update Model Archive') {
            when {
                anyOf {
                    expression { return env.RUN_TRAINING == 'true' }
                    expression { return env.RUN_INFERENCE == 'true' }
                }
            }
            steps {
                sh '''
                    set -eu
                    export PATH="$PWD/.venv/bin:$PATH"
                    if [ ! -f models/gru_model_torchscript.pt ] && [ -f models/gru_model.pt ]; then
                        .venv/bin/python scripts/export_torchscript.py \
                            --config "$TRAIN_CONFIG_PATH" \
                            --checkpoint models/gru_model.pt \
                            --output models/gru_model_torchscript.pt
                    fi
                    .venv/bin/python -m dvc repro archive_models
                '''
            }
        }

        stage('Run Inference') {
            when {
                expression { return env.RUN_INFERENCE == 'true' }
            }
            steps {
                sh '''
                    set -eu
                    mkdir -p reports

                    if [ ! -f models/gru_model_torchscript.pt ]; then
                        echo "TorchScript model not found at models/gru_model_torchscript.pt" >&2
                        exit 1
                    fi
                    if [ ! -f reports/inference_window.csv ]; then
                        echo "Inference input not found at reports/inference_window.csv" >&2
                        exit 1
                    fi

                    if command -v cargo >/dev/null 2>&1; then
                        (
                            cd rust_inference
                            cargo run --release -- \
                                --model ../models/gru_model_torchscript.pt \
                                --input ../reports/inference_window.csv
                        ) | tee reports/rust_inference_output.txt
                    elif command -v docker >/dev/null 2>&1; then
                        MODEL=models/gru_model_torchscript.pt \
                        INPUT=reports/inference_window.csv \
                        RUST_TORCH_DOCKER_IMAGE="$RUST_TORCH_DOCKER_IMAGE" \
                        bash scripts/rust_inference_docker.sh run \
                            | tee reports/rust_inference_output.txt
                    elif [ -f containers/build/rust_torch.sif ]; then
                        MODEL="$PWD/models/gru_model_torchscript.pt" \
                        INPUT="$PWD/reports/inference_window.csv" \
                        bash scripts/rust_inference_container.sh run \
                            | tee reports/rust_inference_output.txt
                    else
                        echo "No Rust runtime available. Install cargo, enable Docker, or build the Singularity image first." >&2
                        exit 1
                    fi
                '''
            }
        }

        stage('DVC Push') {
            when {
                expression { return env.PUSH_DVC == 'true' }
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
                expression { return env.UPLOAD_READABLE_ARTIFACTS == 'true' }
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
                artifacts: 'reports/runtime_*.yaml,reports/model_card.md,reports/minio_parameter_choices.json,reports/minio_*_choices.txt,reports/inference_window.csv,reports/rust_inference_output.txt,reports/docker_rust_inference_*.txt,dvc.lock,data/dvc_archives/*.tar.gz,data/dvc_archives/readable_artifacts_manifest.json,models/*.pt,models/*torchscript*.pt,reports/slurm-*.out,reports/slurm-*.err',
                allowEmptyArchive: true,
                fingerprint: true
            )
        }
    }
}
