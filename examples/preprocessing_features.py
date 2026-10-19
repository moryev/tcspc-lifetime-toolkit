"""Physical features and a separate normalized representation of a synthetic CSV."""

from pathlib import Path

from tcspc_toolkit import (
    CountNormalization, FeatureConfig, extract_features, load_tcspc_measurement_csv,
    normalize_counts, validate_histogram,
)


def run_example():
    path = Path(__file__).resolve().parents[1] / "data/examples/experimental_style_decay.csv"
    measurement = load_tcspc_measurement_csv(
        path, time_unit="ps", time_column="time_ps", data_kind="raw_counts",
    )
    raw = measurement.require_raw_counts()
    validate_histogram(measurement.time_ns, raw)
    config = FeatureConfig(tail_start_ns=5.0, early_stop_ns=3.0, late_start_ns=7.0)
    features = extract_features(measurement.time_ns, raw, config)
    normalized = normalize_counts(raw, mode=CountNormalization.TOTAL)
    print("Synthetic CSV; features describe the observed histogram, including background and IRF.")
    print(features[["total_counts", "mean_arrival_time_ns", "t50_ns",
                    "tail_log_slope_per_ns"]].to_string(index=False))
    print(f"Raw total: {raw.sum()}; separate normalized sum: {normalized.sum():.6f}")
    print("Feature windows are explicit. No background subtraction or temporal alignment is applied.")
    print("Normalized values are a derived representation; retain raw counts for Poisson inference.")
    return measurement, features, normalized


if __name__ == "__main__":
    run_example()
