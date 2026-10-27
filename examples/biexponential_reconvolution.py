"""Explicit raw-measurement mono/bi comparison, without an automatic model choice."""

from pathlib import Path

from tcspc_toolkit.irf.preparation import irf_profile_from_sampled_irf, prepare_irf
from tcspc_toolkit.measurement.adapters import compare_experimental_reconvolution
from tcspc_toolkit.measurement.io import (
    load_sampled_irf_csv,
    load_tcspc_measurement_csv,
)


def run_example():
    directory = Path(__file__).resolve().parents[1] / "data" / "examples"
    sampled = load_sampled_irf_csv(
        directory / "experimental_style_irf.csv", time_unit="ns", time_column="time_ns"
    )
    measurement = load_tcspc_measurement_csv(
        directory / "experimental_style_decay.csv",
        time_unit="ps",
        time_column="time_ps",
        data_kind="raw_counts",
    )
    prepared = prepare_irf(
        irf_profile_from_sampled_irf(sampled),
        measurement.time_ns,
        resampling="none",
        registration_offset_ns=0,
    )
    comparison = compare_experimental_reconvolution(
        measurement,
        prepared_irf=prepared,
        lifetime_bounds_ns=(0.1, 8),
        temporal_shift_bounds_ns=(-0.5, 0.5),
    )
    print(
        "Synthetic CSV stand-ins; no experimental truth or trusted reference supplied."
    )
    print(
        f"IRF selection: {comparison.context.irf_selection}; "
        f"preparation flags: {prepared.diagnostics.flags}"
    )
    print(
        f"Mono valid fit: {comparison.mono.valid_fit}; "
        f"lifetime: {comparison.mono.fitted_lifetime_ns:.5g} ns"
    )
    print(f"Bi parameters: {comparison.bi.parameters}")
    optimization = comparison.bi.optimization
    print(
        f"Bi local numerical validation: {optimization.numerical_validation_passed}; "
        f"exhausted budgets: {optimization.exhausted_budgets}"
    )
    print(f"Bi identifiability diagnostics: {comparison.bi.identifiability}")
    print(
        f"Bi minus mono AIC: {comparison.statistics.delta_aic}; "
        f"bin-based BIC: {comparison.statistics.delta_bic_bins}"
    )
    print("Mono lifetime has no upper bound; its background lower bound is 1e-12.")
    print(
        "Information criteria and warning absence do not establish physical correctness."
    )
    return comparison


if __name__ == "__main__":
    run_example()
