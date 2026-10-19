"""Synthetic Poisson reconvolution; run with python examples/quickstart.py."""

# README imports begin
import numpy as np

from tcspc_toolkit import (
    build_expected_counts_from_irf, fit_monoexponential_reconvolution,
    generate_gaussian_irf_profile, monoexponential_decay, prepare_irf,
    sample_photon_counts,
)
from tcspc_toolkit.estimation.classical import estimate_reconvolution_initial_guess
from tcspc_toolkit.evaluation.diagnostics import calculate_poisson_deviance_residuals
# README imports end


def run_example():
    """Return observations and the computed fit for optional plotting."""
    # README workflow begin
    time_ns = np.linspace(0.0, 20.0, 512)
    lifetime_true_ns = 2.5
    decay = monoexponential_decay(time_ns, amplitude=1.0, lifetime=lifetime_true_ns)
    source = generate_gaussian_irf_profile(
        time_ns, gaussian_centre_ns=1.0, gaussian_fwhm_ns=0.3,
    )
    prepared = prepare_irf(source, time_ns)  # Explicit same-grid, unit-area kernel.
    expected = build_expected_counts_from_irf(
        decay, prepared, signal_photon_count=100_000, background_per_bin=2.0,
    )  # Convolve, scale the finite-window signal sum, then add background.
    counts = sample_photon_counts(expected, np.random.default_rng(42))
    guess = estimate_reconvolution_initial_guess(
        time=time_ns, counts=counts, irf=prepared.kernel,
    )
    fit = fit_monoexponential_reconvolution(
        time_ns, counts, prepared.kernel,
        initial_guess=(guess.amplitude, guess.lifetime_ns,
                       guess.background, guess.temporal_shift_ns),
        temporal_shift_bounds=(-0.5, 0.5), objective="poisson",
    )
    if not fit.success:
        raise RuntimeError(f"Fit did not pass numerical checks: {fit.optimizer_message}")
    residuals = calculate_poisson_deviance_residuals(counts, fit.fitted_curve)
    print(f"Synthetic lifetime: {lifetime_true_ns:.4f} ns; estimate: {fit.lifetime:.4f} ns")
    print(f"Background: {fit.background:.3f} counts/bin; shift: {fit.temporal_shift:.5f} ns")
    print(f"Numerical validation: {fit.numerical_validation_passed}")
    print(f"Poisson deviance: {np.sum(residuals**2):.3f} across {counts.size} bins")
    # README workflow end
    return time_ns, counts, expected, fit


def save_figure(path, time_ns, counts, expected, fit):
    """Plot the same observation/fit, without performing a second fit."""
    from pathlib import Path
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    residuals = calculate_poisson_deviance_residuals(counts, fit.fitted_curve)
    fig, (curve_ax, residual_ax) = plt.subplots(
        2, 1, figsize=(9, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}, layout="constrained",
    )
    curve_ax.scatter(time_ns, counts, s=9, color="#507c9b", alpha=0.65,
                     label="Observed Poisson counts", rasterized=True)
    curve_ax.plot(time_ns, expected, color="#222222", linestyle="--",
                  linewidth=1.4, label="Generating expected counts")
    curve_ax.plot(time_ns, fit.fitted_curve, color="#ce642e", linewidth=1.5,
                  label=f"Poisson reconvolution fit: {fit.lifetime:.4f} ns")
    curve_ax.set(ylabel="Counts per bin", title="Synthetic TCSPC: 2.5 ns lifetime, Gaussian IRF")
    curve_ax.legend(frameon=False, fontsize=9)
    residual_ax.plot(time_ns, residuals, color="#507c9b", linewidth=0.8)
    residual_ax.axhline(0, color="#555555", linewidth=0.7)
    residual_ax.set(xlabel="Time (ns)", ylabel="Deviance\nresidual", xlim=(0, 20))
    fig.savefig(target, dpi=160)
    plt.close(fig)
    print(f"Figure saved: {target}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", help="Optional output image path; no plot is written by default")
    args = parser.parse_args()
    result = run_example()
    if args.figure:
        save_figure(args.figure, *result)
