"""Experimental K=3 research workflow with explicit TRF settings and synthetic data."""

from collections import Counter
from pathlib import Path

import numpy as np

from tcspc_toolkit.estimation.triexponential import (
    TriexponentialFitSettings,
    TriexponentialParameters,
)
from tcspc_toolkit.irf.preparation import prepare_irf
from tcspc_toolkit.irf.profiles import generate_gaussian_irf_profile
from tcspc_toolkit.measurement.triexponential import (
    fit_experimental_triexponential_reconvolution,
)
from tcspc_toolkit.measurement.records import MeasurementDataKind, TCSPCMeasurement
from tcspc_toolkit.physics.forward_model import (
    multiexponential_reconvolution_component_counts,
    multiexponential_reconvolution_expected_counts,
)
from tcspc_toolkit.simulation.histograms import sample_photon_counts


def build_inputs():
    """Two seeded raw-count examples, including exactly coincident components."""
    time = np.arange(256) * 0.0625
    profile = generate_gaussian_irf_profile(
        time,
        gaussian_centre_ns=0.6,
        gaussian_fwhm_ns=0.25,
        provenance={"role": "synthetic research example IRF"},
    )
    prepared = prepare_irf(profile, time, resampling="none", registration_offset_ns=0)
    cases = []
    for index, (name, lifetimes) in enumerate(
        (("separated", (0.6, 2.0, 4.8)), ("coincident", (1.5, 1.5, 4.8)))
    ):
        unit = multiexponential_reconvolution_component_counts(
            time,
            prepared.kernel,
            amplitudes=(1, 1, 1),
            lifetimes=lifetimes,
            temporal_shift=0.043,
        )
        amplitudes = 20_000 * np.array([0.25, 0.45, 0.30]) / unit.sum(axis=1)
        truth = TriexponentialParameters(tuple(amplitudes), lifetimes, 0.5, 0.043)
        expected = (amplitudes[:, None] * unit).sum(axis=0) + truth.background
        seed = [37, 3, 20261113, index]
        counts = sample_photon_counts(expected, np.random.default_rng(seed))
        measurement = TCSPCMeasurement(
            time,
            counts,
            MeasurementDataKind.RAW_COUNTS,
            sample_id=name,
            provenance={"role": "synthetic teaching observation", "rng_entropy": seed},
        )
        alternative = alternative_expected = None
        if name == "coincident":
            combined = float(amplitudes[0] + amplitudes[1])
            alternative = TriexponentialParameters(
                (0.9 * combined, 0.1 * combined, float(amplitudes[2])),
                lifetimes,
                truth.background,
                truth.temporal_shift_ns,
            )
            alternative_expected = multiexponential_reconvolution_expected_counts(
                time,
                prepared.kernel,
                amplitudes=alternative.amplitudes,
                lifetimes=alternative.lifetimes_ns,
                background=alternative.background,
                temporal_shift=alternative.temporal_shift_ns,
            )
        cases.append(
            dict(
                name=name,
                measurement=measurement,
                prepared=prepared,
                truth=truth,
                expected=expected,
                alternative=alternative,
                alternative_expected=alternative_expected,
            )
        )
    return cases


def run_example():
    """Fit each observation once and retain every result, including rejections."""
    settings = TriexponentialFitSettings(
        max_seed_starts=16,
        reserved_recovery_starts=4,
        max_optimization_calls=20_000,
        wall_seconds=60,
        policy="compact",
        solver="trf",
    )
    cases = build_inputs()
    print("Experimental K=3 research example with synthetic Poisson observations.")
    for case in cases:
        result = fit_experimental_triexponential_reconvolution(
            case["measurement"],
            prepared_irf=case["prepared"],
            settings=settings,
            lifetime_bounds_ns=(0.1, 10),
            temporal_shift_bounds_ns=(-0.3, 0.3),
        )
        case["result"] = result
        fit = result.fit
        print(f"\n{case['name']} generating parameters {case['truth']}")
        print(f"Returned parameters {fit.parameters}")
        print(
            f"Numerically accepted {fit.numerical.accepted}, reasons {fit.numerical.reasons}"
        )
        print(f"Stopping reason {fit.optimization.stopping_reason}")
        print(
            f"Start terminations {dict(Counter(s.termination for s in fit.optimization.starts))}"
        )
        print(f"Best finite objective {fit.optimization.best_observed_centered_nll}")
        print(
            f"Conditioning {fit.identifiability.conditioning_status}, rank {fit.identifiability.weighted_jacobian_rank}"
        )
        print(f"Condition number {fit.identifiability.condition_number}")
        print(
            f"Zero-amplitude components {fit.identifiability.zero_amplitude_components}"
        )
        print(f"Derivative statuses {fit.identifiability.derivative_statuses}")
        print(
            f"Active bounds {fit.identifiability.active_bounds}, equal pairs {fit.identifiability.equal_lifetime_pairs}"
        )
        print(f"Lifetime ratios {fit.identifiability.lifetime_ratios}")
        print(f"Measurement provenance {dict(result.context.measurement.provenance)}")
        print(
            f"IRF selection {result.context.irf_selection}, provenance {dict(fit.assumptions.source_provenance)}"
        )
        if fit.expected_counts is not None:
            error = np.linalg.norm(
                fit.expected_counts - case["expected"]
            ) / np.linalg.norm(case["expected"])
            print(f"Curve relative L2 error against generating mean {error:.6g}")
        if fit.statistics is not None:
            print(
                f"Centered NLL {fit.centered_nll:.6g}, Poisson deviance {fit.statistics.deviance:.6g}"
            )
            print(
                f"Deviance residual range {np.min(fit.statistics.deviance_residuals):.4g} to {np.max(fit.statistics.deviance_residuals):.4g}"
            )
        if case["alternative"] is not None:
            difference = np.max(np.abs(case["expected"] - case["alternative_expected"]))
            print(
                f"Alternative coincident-component amplitudes {case['alternative'].amplitudes}"
            )
            print(
                f"Maximum expected-curve difference for alternative amplitudes {difference:.6g}"
            )
    print(
        "\nSimilar curves can have different components. Inspect residuals, bounds and conditioning."
    )
    print(
        "Numerical acceptance alone cannot establish global optimality or uniquely resolved populations."
    )
    return cases


def save_figure(path, cases):
    """Plot stored results and component curves without refitting."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 6),
        sharex="col",
        layout="constrained",
        gridspec_kw={"height_ratios": [3, 1]},
    )
    for column, case in enumerate(cases):
        measurement, fit = case["measurement"], case["result"].fit
        time = measurement.time_ns
        curve, residual = axes[:, column]
        curve.scatter(time, measurement.values, s=7, alpha=0.5, label="Observed counts")
        curve.plot(time, case["expected"], "k--", label="Generating mean")
        if fit.expected_counts is not None:
            curve.plot(time, fit.expected_counts, label="Returned fit")
        if fit.component_counts is not None:
            for index, component in enumerate(fit.component_counts):
                curve.plot(
                    time,
                    component,
                    alpha=0.5,
                    linewidth=0.8,
                    label=f"Component {index + 1}",
                )
        if fit.statistics is not None:
            residual.plot(time, fit.statistics.deviance_residuals, linewidth=0.8)
        else:
            residual.text(
                0.5,
                0.5,
                "Residuals unavailable",
                transform=residual.transAxes,
                ha="center",
            )
        residual.axhline(0, color="gray", linewidth=0.8)
        residual.set(xlabel="Time (ns)", ylabel="Deviance residual")
        curve.set(
            ylabel="Counts per bin",
            title=f"{case['name']} — accepted {fit.numerical.accepted}",
        )
        curve.legend(fontsize=7)
    fig.savefig(target, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure", type=Path, help="Optional figure from the two computed fits"
    )
    args = parser.parse_args()
    cases = run_example()
    if args.figure:
        save_figure(args.figure, cases)
