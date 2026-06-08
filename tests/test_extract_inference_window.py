import pandas as pd

from scripts.extract_inference_window import extract_inference_window


def test_extract_inference_window_writes_feature_only_csv(tmp_path):
    processed_path = tmp_path / "processed.csv"
    output_path = tmp_path / "window.csv"
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
training:
  sequence_length: 2
""",
        encoding="utf-8",
    )

    result = extract_inference_window(config_path, output_path)
    window = pd.read_csv(result)

    assert result == output_path
    assert list(window.columns) == ["feature_a", "feature_b"]
    assert len(window) == 2
    assert window.iloc[0].tolist() == [1.0, 4.0]


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
