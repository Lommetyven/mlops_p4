# Evidently Drift Monitoring

This implementation provides the evidence for the monitoring requirements:

- **D6.3:** detect drift and define how the pipeline reacts to a performance
  decrease.
- **D6.4:** show the monitoring system and the metrics it logs.

## What is monitored

The experiment compares a deterministic reference sample from the training
split with five production-like windows from the test split:

1. normal baseline;
2. normal control;
3. mild artificial drift;
4. strong artificial drift; and
5. recovery.

The same current samples are reused in every window. This makes the experiment
controlled: differences between windows are caused by the configured feature
shift and not by random sampling.

Evidently applies a two-sample Kolmogorov-Smirnov test to all 16 numerical model
features. A feature is drifted when its p-value is at or below `0.05`. Dataset
drift is raised when at least 25% of the features drift. The experiment also
logs prediction drift and, when ground truth is available, RMSE, MAE, and R2.
The current threshold marks performance as degraded when RMSE is more than 5%
above the first normal window.

The demonstration shifts four physically related measurements by a configurable
number of reference standard deviations:

- `Global_active_power`
- `Global_reactive_power`
- `Voltage`
- `Global_intensity`

The shift is applied to every time step in each GRU input sequence. The normal
and recovery windows are not shifted.

## Run locally

Install the project requirements, then run:

```bash
python scripts/evidently_monitoring.py \
  --checkpoint models/gru_model.pt \
  --config configs/train_config.yaml \
  --drift-config configs/drift_monitoring_config.yaml \
  --monitoring-config configs/monitering_config.yaml \
  --output-dir reports/evidently
```

Add `--wandb` to log the same measurements and evidence files to a dedicated
W&B run. If online mode is configured, `WANDB_API_KEY` must be available.

Start the Evidently UI after the experiment:

```bash
evidently ui --workspace reports/evidently/workspace --port 8000
```

Open `http://localhost:8000`. The workspace dashboard shows drifted-feature
share, prediction drift, RMSE, MAE, and R2 across the five windows.

## Run from Jenkins on AI Lab

Use:

- `RUN_DRIFT_MONITORING=true`
- `RUN_TRAINING=false` to monitor a saved model, or `true` to monitor the model
  produced by the build
- `RUN_DVC_REPRO=false` when the processed data is already versioned
- `TRAIN_RUNNER=AI_LAB`
- `MODEL_VERSION=<saved model version>` when training is disabled

The monitoring stage deliberately requests one node and one GPU because this
experiment evaluates one deployed-model replica. Jenkins transfers the
checkpoint, processed data, runtime configuration, and monitoring code to AI
Lab, submits the Slurm job, retrieves the evidence, and archives it.

## Evidence files

The output directory contains:

- `monitoring_dashboard.png`: a report-ready figure for D6.4;
- `monitoring_summary.md`: method, thresholds, results, and reaction;
- `monitoring_summary.csv`: one flat row per monitoring window;
- `monitoring_summary.json`: complete configuration and metrics;
- `reports/*.html`: interactive Evidently reports;
- `reports/*.json`: machine-readable Evidently snapshots;
- `workspace/`: the local Evidently project and dashboard; and
- `window_samples/*.csv`: the exact samples sent to the monitor.

Jenkins also archives the Slurm standard output and error logs.

## Reaction and mitigation

Feature drift is an investigation signal, not automatic proof of degraded model
quality. The operational response is:

1. Validate schema, ranges, missing values, sensor behavior, and upstream
   preprocessing for the drifted features.
2. Check whether drift persists over several windows and whether prediction
   drift or labelled regression quality also changes.
3. Continue monitoring when drift is absent and quality remains within the
   accepted tolerance.
4. Investigate the input pipeline when data drift is present but labelled model
   quality has not degraded.
5. When RMSE remains more than 5% above the baseline, create a new validated
   DVC data version and retrain through Jenkins.
6. Compare the challenger against the deployed model on the same validation and
   test criteria. Deploy only when it passes; otherwise retain or restore the
   last validated MinIO model version.

## Interpretation limits

This is a controlled implementation proof based on artificial covariate shift.
It verifies that the monitor detects a known change and connects that change to
model quality. It does not prove that every real production shift will follow
the same pattern.

The KS test is sensitive to sample size, and testing many features increases the
chance of false alarms. Thresholds must therefore be calibrated on multiple
normal production windows. Regression-quality metrics also require delayed
ground truth; until labels arrive, data and prediction drift are proxy signals
rather than direct measurements of performance.
