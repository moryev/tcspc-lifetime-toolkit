"""Small held-out synthetic prediction example, independent of frozen A-F studies."""

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from tcspc_toolkit.ml.models import make_normalized_histogram_ridge_pipeline
from tcspc_toolkit.simulation.histograms import simulate_irf_convolved_histogram


def run_example():
    rng = np.random.default_rng(1403)
    time_ns = np.linspace(0.0, 20.0, 256)
    lifetimes_ns = rng.uniform(1.0, 4.0, 96)
    histograms = np.vstack([
        simulate_irf_convolved_histogram(
            time=time_ns, lifetime_ns=lifetime, signal_photon_count=20_000,
            background_per_bin=2.0, irf_centre_ns=1.0, irf_fwhm_ns=0.3,
            irf_shift_ns=0.0, rng=rng,
        )[0]
        for lifetime in lifetimes_ns
    ])
    # Each row has independently drawn physical parameters: no repeated groups.
    train, test = train_test_split(np.arange(len(histograms)), test_size=0.25, random_state=42)
    model = make_normalized_histogram_ridge_pipeline()
    # TOTAL normalization is per histogram; the scaler learns training statistics only.
    model.fit(histograms[train], lifetimes_ns[train])
    predictions = model.predict(histograms[test])
    baseline = np.full(len(test), lifetimes_ns[train].mean())
    mae = mean_absolute_error(lifetimes_ns[test], predictions)
    baseline_mae = mean_absolute_error(lifetimes_ns[test], baseline)
    print(f"Synthetic observations: {len(train)} training, {len(test)} held out")
    print(f"Normalized-histogram Ridge MAE: {mae:.4f} ns")
    print(f"Training-mean baseline MAE: {baseline_mae:.4f} ns")
    print("Held-out interpolation at fixed acquisition settings; not OOD or experimental validation.")
    print("No A-F observations, tuning, or uncertainty calibration are used.")
    return model, train, test, predictions, mae, baseline_mae


if __name__ == "__main__":
    run_example()
