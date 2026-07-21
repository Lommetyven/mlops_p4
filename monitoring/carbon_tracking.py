import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

DEFAULT_CARBON_TRACKING_CONFIG = {
    "enabled": False,
    "log_dir": "reports/carbontracker",
    "log_file_prefix": "training",
    "epochs_before_pred": 1,
    "monitor_epochs": -1,
    "update_interval": 1,
    "interpretable": True,
    "stop_and_confirm": False,
    "ignore_errors": True,
    "components": "gpu",
    "devices_by_pid": False,
    "verbose": 1,
    "decimal_precision": 12,
}


def start_carbon_tracker_if_enabled(config: Mapping[str, Any]):
    carbon_config = merged_carbon_tracking_config(config)
    if not carbon_config["enabled"]:
        return None

    try:
        from carbontracker.tracker import CarbonTracker
    except ImportError as exc:
        print(f"CarbonTracker disabled: {exc}")
        return None

    log_dir = Path(carbon_config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        tracker = CarbonTracker(
            epochs=int(config["training"]["epochs"]),
            epochs_before_pred=int(carbon_config["epochs_before_pred"]),
            monitor_epochs=int(carbon_config["monitor_epochs"]),
            update_interval=float(carbon_config["update_interval"]),
            interpretable=bool(carbon_config["interpretable"]),
            stop_and_confirm=bool(carbon_config["stop_and_confirm"]),
            ignore_errors=bool(carbon_config["ignore_errors"]),
            components=carbon_config["components"],
            devices_by_pid=bool(carbon_config["devices_by_pid"]),
            log_dir=str(log_dir),
            log_file_prefix=carbon_config["log_file_prefix"],
            verbose=int(carbon_config["verbose"]),
            decimal_precision=int(carbon_config["decimal_precision"]),
        )
    except Exception as exc:
        print(f"CarbonTracker disabled: {exc}")
        return None

    return tracker


def finish_carbon_tracker(tracker, config: Mapping[str, Any]):
    if tracker is None:
        return {}

    try:
        tracker.stop()
    except Exception as exc:
        print(f"CarbonTracker stop failed: {exc}")

    carbon_config = merged_carbon_tracking_config(config)
    return collect_carbontracker_summary(carbon_config["log_dir"])


def collect_carbontracker_summary(log_dir):
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return {}

    try:
        from carbontracker.parser import parse_all_logs
    except ImportError:
        return {}

    try:
        logs = parse_all_logs(str(log_dir))
    except Exception as exc:
        print(f"CarbonTracker log parsing failed: {exc}")
        return {}

    if not logs:
        return {}

    latest_log = max(
        logs,
        key=lambda entry: Path(entry["output_filename"]).stat().st_mtime,
    )
    components = latest_log.get("components", {})
    summary = {
        "carbontracker/output_log": latest_log["output_filename"],
        "carbontracker/standard_log": latest_log["standard_filename"],
        "carbontracker/early_stop": int(bool(latest_log.get("early_stop"))),
    }
    fallback_consumptions = _fallback_consumptions_from_output(
        latest_log["output_filename"]
    )
    actual_consumption = _prefer_fallback_epochs(
        latest_log.get("actual"),
        fallback_consumptions.get("actual"),
    )
    predicted_consumption = _prefer_fallback_epochs(
        latest_log.get("pred"),
        fallback_consumptions.get("predicted"),
    )
    actual_consumption, predicted_consumption, gpu_metrics = (
        _correct_gpu_only_consumptions(
            actual_consumption,
            predicted_consumption,
            components,
        )
    )

    summary.update(_flatten_consumption("actual", actual_consumption))
    summary.update(_flatten_consumption("predicted", predicted_consumption))
    summary.update(_flatten_component_metrics(components))
    summary.update(gpu_metrics)

    return summary


def carbontracker_log_files(log_dir):
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return []

    return sorted(
        path
        for path in log_dir.glob("*carbontracker*.log")
        if path.is_file() and path.stat().st_size > 0
    )


def merged_carbon_tracking_config(config: Mapping[str, Any]):
    merged = dict(DEFAULT_CARBON_TRACKING_CONFIG)
    merged.update(config.get("carbon_tracking", {}))
    return merged


def _flatten_consumption(label, consumption):
    if not consumption:
        return {}

    return {
        f"carbontracker/{label}_epochs": consumption.get("epochs"),
        f"carbontracker/{label}_duration_seconds": consumption.get("duration (s)"),
        f"carbontracker/{label}_energy_kwh": consumption.get("energy (kWh)"),
        f"carbontracker/{label}_co2eq_g": consumption.get("co2eq (g)"),
    }


def _prefer_fallback_epochs(parsed, fallback):
    if not parsed:
        return fallback
    if not fallback:
        return parsed

    parsed_epochs = int(parsed.get("epochs") or 0)
    fallback_epochs = int(fallback.get("epochs") or 0)
    if fallback_epochs > parsed_epochs:
        merged = dict(parsed)
        merged["epochs"] = fallback_epochs
        return merged

    return parsed


def _fallback_consumptions_from_output(output_filename):
    output_path = Path(output_filename)
    if not output_path.exists():
        return {}

    output_text = output_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.finditer(
        r"(?P<label>Actual consumption(?:\s+for\s+(?P<actual_epochs>\d+)"
        r"\s+epoch(?:s|\(s\)))?|Predicted consumption\s+for\s+"
        r"(?P<predicted_epochs>\d+)\s+epoch\(s\)):"
        r"[\s\S]*?Time:\s*(?P<duration>[^\n]+)"
        r"[\s\S]*?Energy:\s*(?P<energy>[-+0-9.eE]+)\s*kWh"
        r"[\s\S]*?CO2eq:\s*(?P<co2eq>[-+0-9.eE]+)\s*g",
        output_text,
        flags=re.IGNORECASE,
    )

    consumptions = {}
    for match in matches:
        label = match.group("label").lower()
        key = "predicted" if label.startswith("predicted") else "actual"
        epochs = (
            match.group("predicted_epochs")
            if key == "predicted"
            else match.group("actual_epochs")
        )
        consumptions[key] = {
            "epochs": int(epochs or 1),
            "duration (s)": _parse_duration_seconds(match.group("duration")),
            "energy (kWh)": float(match.group("energy")),
            "co2eq (g)": float(match.group("co2eq")),
        }

    return consumptions


def _parse_duration_seconds(duration):
    match = re.search(r"(\d+):(\d{2}):(\d\d?(?:\.\d+)?)", duration)
    if not match:
        return None

    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _flatten_component_metrics(components):
    metrics = {}
    for component_name, component_metrics in components.items():
        prefix = f"carbontracker/{component_name}"
        devices = _component_devices(component_metrics)
        device_count = len(devices)

        power_usages = component_metrics.get("avg_power_usages (W)")
        if power_usages is not None:
            average_power = _nanmean(power_usages)
            if component_name == "gpu" and device_count:
                metrics[f"{prefix}_avg_power_per_device_watts"] = average_power
                average_power *= device_count
            metrics[f"{prefix}_avg_power_watts"] = average_power

        energy_usages = component_metrics.get("avg_energy_usages (J)")
        if energy_usages is not None:
            energy_joules = _nansum(energy_usages)
            if component_name == "gpu" and device_count:
                energy_joules *= _missing_device_dimension_scale(
                    energy_usages,
                    device_count,
                )
            metrics[f"{prefix}_energy_joules"] = energy_joules

        durations = component_metrics.get("epoch_durations (s)")
        if durations is not None:
            metrics[f"{prefix}_duration_seconds"] = _nansum(durations)

        if devices:
            metrics[f"{prefix}_devices"] = ", ".join(devices)
            metrics[f"{prefix}_device_count"] = device_count

    return metrics


def _correct_gpu_only_consumptions(actual, predicted, components):
    if set(components) != {"gpu"} or not actual:
        return actual, predicted, {}

    gpu_component = components["gpu"]
    devices = _component_devices(gpu_component)
    energy_usages = gpu_component.get("avg_energy_usages (J)")
    if not devices or energy_usages is None:
        return actual, predicted, {}

    device_correction_factor = _missing_device_dimension_scale(
        energy_usages,
        len(devices),
    )
    parsed_gpu_energy_joules = _nansum(energy_usages) * device_correction_factor

    try:
        from carbontracker.constants import PUE_2023
    except ImportError:
        PUE_2023 = 1.0

    raw_energy_kwh = _positive_float(actual.get("energy (kWh)"))
    raw_co2eq_g = _positive_float(actual.get("co2eq (g)"))
    if raw_energy_kwh is not None:
        correction_factor = device_correction_factor
        corrected_energy_kwh = raw_energy_kwh * correction_factor
        gpu_energy_kwh = corrected_energy_kwh / float(PUE_2023)
    else:
        correction_factor = 1.0
        gpu_energy_kwh = parsed_gpu_energy_joules / 3_600_000
        corrected_energy_kwh = gpu_energy_kwh * float(PUE_2023)

    carbon_intensity = (
        raw_co2eq_g / raw_energy_kwh
        if raw_energy_kwh is not None and raw_co2eq_g is not None
        else None
    )

    corrected_actual = dict(actual)
    corrected_actual["energy (kWh)"] = corrected_energy_kwh
    if carbon_intensity is not None:
        corrected_actual["co2eq (g)"] = corrected_energy_kwh * carbon_intensity

    corrected_predicted = _scale_consumption(predicted, correction_factor)
    metrics = {
        "carbontracker/gpu_energy_joules": gpu_energy_kwh * 3_600_000,
        "carbontracker/gpu_energy_kwh": gpu_energy_kwh,
        "carbontracker/gpu_pue": float(PUE_2023),
        "carbontracker/gpu_consumption_correction_factor": correction_factor,
    }
    if carbon_intensity is not None:
        metrics["carbontracker/gpu_carbon_intensity_g_per_kwh"] = carbon_intensity
        metrics["carbontracker/gpu_co2eq_g"] = gpu_energy_kwh * carbon_intensity

    if correction_factor != 1.0:
        metrics.update(_flatten_consumption("raw_actual", actual))
        metrics.update(_flatten_consumption("raw_predicted", predicted))

    return corrected_actual, corrected_predicted, metrics


def _scale_consumption(consumption, factor):
    if not consumption:
        return consumption

    scaled = dict(consumption)
    for key in ("energy (kWh)", "co2eq (g)"):
        if scaled.get(key) is not None:
            scaled[key] = float(scaled[key]) * factor
    return scaled


def _component_devices(component_metrics):
    return [
        str(device).strip()
        for device in component_metrics.get("devices", [])
        if str(device).strip()
    ]


def _missing_device_dimension_scale(values, device_count):
    array = np.asarray(values, dtype=float)
    measured_device_count = array.shape[-1] if array.ndim >= 2 else 1
    return device_count / max(1, measured_device_count)


def _positive_float(value):
    if value is None:
        return None
    number = float(value)
    return number if number > 0 else None


def _nanmean(values):
    return float(np.nanmean(np.asarray(values, dtype=float)))


def _nansum(values):
    return float(np.nansum(np.asarray(values, dtype=float)))
