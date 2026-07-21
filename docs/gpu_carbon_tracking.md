# GPU Carbon Tracking

Training uses CarbonTracker for GPU-only energy and carbon measurements. CPU
and DRAM tracking are disabled because AI Lab compute nodes do not expose the
required Intel RAPL counters to jobs.

CarbonTracker 2.4.5 logs multi-GPU power as an average across devices. The
project summary corrects that value using the GPU device count recorded in the
same log. W&B and the generated model card include these metrics:

- `carbontracker/gpu_device_count`: allocated GPUs included in the measurement.
- `carbontracker/gpu_avg_power_per_device_watts`: mean power for one GPU.
- `carbontracker/gpu_avg_power_watts`: total mean power across allocated GPUs.
- `carbontracker/gpu_energy_kwh`: direct measured GPU electricity consumption.
- `carbontracker/gpu_co2eq_g`: direct GPU emissions at the detected grid carbon
  intensity.
- `carbontracker/actual_energy_kwh` and `actual_co2eq_g`: GPU-only totals after
  CarbonTracker's PUE adjustment.

The original CarbonTracker values are retained under `raw_actual_*` and
`raw_predicted_*` when a multi-GPU correction is applied.
