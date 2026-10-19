"""Import synthetic experimental-style CSV fixtures, with explicit IRF preparation."""

from pathlib import Path

from tcspc_toolkit import (
    fit_experimental_reconvolution, irf_profile_from_sampled_irf,
    load_sampled_irf_csv, load_tcspc_measurement_csv, prepare_irf,
)


def run_example():
    example_dir = Path(__file__).resolve().parents[1] / "data" / "examples"
    sampled = load_sampled_irf_csv(
        example_dir / "experimental_style_irf.csv", time_unit="ns", time_column="time_ns",
    )
    measurement = load_tcspc_measurement_csv(
        example_dir / "experimental_style_decay.csv", time_unit="ps",
        time_column="time_ps", data_kind="raw_counts",
    )
    source = irf_profile_from_sampled_irf(sampled)
    prepared = prepare_irf(
        source, measurement.time_ns, resampling="none", registration_offset_ns=0.0,
    )  # These fixture grids match after unit conversion; no alignment is inferred.
    fit = fit_experimental_reconvolution(
        measurement, prepared_irf=prepared, temporal_shift_bounds_ns=(-0.5, 0.5),
    )
    print("Synthetic CSV stand-ins, not acquired experimental observations.")
    print(f"Imported {measurement.values.size} raw-count bins; internal time unit: ns")
    print(f"IRF source: {source.source_kind.value}; preparation: no resampling, zero registration")
    print(f"IRF support loss: {prepared.diagnostics.support_loss_fraction:.3g}; "
          f"preparation flags: {prepared.diagnostics.flags}")
    print(f"Estimated lifetime: {fit.fitted_lifetime_ns:.4f} ns; valid fit: {fit.valid_fit}")
    print(f"Poisson deviance: {fit.poisson_deviance:.3f}; boundary hit: {fit.boundary_hit}")
    print("No trusted reference was supplied; no lifetime error or coverage is calculated.")
    if not fit.valid_fit:
        raise RuntimeError(f"Unusable fit: {fit.failure_reason}")
    return measurement, sampled, prepared, fit


if __name__ == "__main__":
    run_example()
