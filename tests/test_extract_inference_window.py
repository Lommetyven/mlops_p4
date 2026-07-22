import pandas as pd

from scripts.extract_inference_window import extract_inference_window


def test_extract_inference_window_writes_feature_only_csv(tmp_path):
    processed_path = tmp_path / "processed.csv"
    output_path = tmp_path / "window.csv"
    targets_output_path = tmp_path / "targets.csv"
    config_path = tmp_path / "config.yaml"

    dataframe = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [4.0, 5.0, 6.0],
            "target_next_hour": [7.0, 8.0, 9.0],
        }
    )
    dataframe.to_csv(processed_path, index=False)
    config_path.write_text(
        f"""
data:
  processed_path: "{processed_path}"
  feature_columns:
    - "feature_a"
    - "feature_b"
  target_column: "target_next_hour"
training:
  sequence_length: 2
""",
        encoding="utf-8",
    )

    result = extract_inference_window(
        config_path,
        output_path,
        targets_output_path=targets_output_path,
    )
    window = pd.read_csv(result)
    targets = pd.read_csv(targets_output_path)

    assert result == output_path
    assert list(window.columns) == ["feature_a", "feature_b"]
    assert len(window) == 3
    assert window.iloc[0].tolist() == [1.0, 4.0]
    assert targets["target_next_hour"].tolist() == [8.0, 9.0]


def test_extract_inference_window_uses_explicit_sequence_length_for_targets(tmp_path):
    processed_path = tmp_path / "processed.csv"
    output_path = tmp_path / "window.csv"
    targets_output_path = tmp_path / "targets.csv"
    config_path = tmp_path / "config.yaml"

    pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "target_next_hour": [10.0, 20.0, 30.0, 40.0],
        }
    ).to_csv(processed_path, index=False)
    config_path.write_text(
        f"""
data:
  processed_path: "{processed_path}"
  feature_columns:
    - "feature_a"
  target_column: "target_next_hour"
training:
  sequence_length: 2
""",
        encoding="utf-8",
    )

    extract_inference_window(
        config_path,
        output_path,
        targets_output_path=targets_output_path,
        sequence_length=3,
    )

    assert pd.read_csv(targets_output_path)["target_next_hour"].tolist() == [30.0, 40.0]


def test_extract_inference_window_allows_explicit_row_limit(tmp_path):
    processed_path = tmp_path / "processed.csv"
    output_path = tmp_path / "window.csv"
    config_path = tmp_path / "config.yaml"

    pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [4.0, 5.0, 6.0],
        }
    ).to_csv(processed_path, index=False)
    config_path.write_text(
        f"""
data:
  processed_path: "{processed_path}"
  feature_columns:
    - "feature_a"
    - "feature_b"
training:
  sequence_length: 2
""",
        encoding="utf-8",
    )

    result = extract_inference_window(config_path, output_path, rows=2)

    assert len(pd.read_csv(result)) == 2


def test_extract_inference_window_rejects_short_processed_data(tmp_path):
    processed_path = tmp_path / "processed.csv"
    output_path = tmp_path / "window.csv"
    config_path = tmp_path / "config.yaml"

    pd.DataFrame(
        {
            "feature_a": [1.0],
            "feature_b": [2.0],
            "target_next_hour": [3.0],
        }
    ).to_csv(processed_path, index=False)
    config_path.write_text(
        f"""
data:
  processed_path: "{processed_path}"
  feature_columns:
    - "feature_a"
    - "feature_b"
training:
  sequence_length: 2
""",
        encoding="utf-8",
    )

    try:
        extract_inference_window(config_path, output_path)
    except ValueError as error:
        assert "needs at least 2" in str(error)
    else:
        raise AssertionError("Expected extract_inference_window to fail.")
