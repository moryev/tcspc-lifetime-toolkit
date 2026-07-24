from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from tcspc_toolkit.fitting import fit_monoexponential_decay
from tcspc_toolkit.simulation import simulate_ideal_decay


def run_simulation(args: Namespace) -> None:
    """Generate and display a synthetic TCSPC decay."""

    time = np.linspace(
        start=args.start_time,
        stop=args.end_time,
        num=args.bins,
    )

    expected_counts, measured_counts = simulate_ideal_decay(
        time=time,
        amplitude=args.amplitude,
        lifetime=args.lifetime,
        background=args.background,
        random_seed=args.random_seed,
    )

    print("Simulation completed.")
    print(f"Number of bins: {args.bins}")
    print(f"First expected counts: {expected_counts[:5]}")
    print(f"First measured counts: {measured_counts[:5]}")


def run_fit(args: Namespace) -> None:
    """Load a CSV file and fit a monoexponential decay."""

    input_path = Path(args.input)

    if not input_path.exists():
        raise SystemExit(
            f"Error: input file does not exist: {input_path}"
        )

    try:
        data = pd.read_csv(input_path)
    except Exception as error:
        raise SystemExit(
            f"Error: could not read CSV file: {error}"
        ) from error

    required_columns = {
        args.time_column,
        args.counts_column,
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        available_columns = ", ".join(data.columns)

        raise SystemExit(
            "Error: missing required CSV column(s): "
            f"{', '.join(sorted(missing_columns))}\n"
            f"Available columns: {available_columns}"
        )

    time = data[args.time_column].to_numpy(dtype=np.float64)
    counts = data[args.counts_column].to_numpy(dtype=np.float64)

    automatic_background = float(np.min(counts))

    automatic_amplitude = float(
        np.max(counts) - automatic_background
    )

    automatic_lifetime = float(
        (np.max(time) - np.min(time)) / 5.0
    )

    initial_amplitude = (
        args.initial_amplitude
        if args.initial_amplitude is not None
        else automatic_amplitude
    )

    initial_lifetime = (
        args.initial_lifetime
        if args.initial_lifetime is not None
        else automatic_lifetime
    )

    initial_background = (
        args.initial_background
        if args.initial_background is not None
        else automatic_background
    )

    initial_guess = (
        initial_amplitude,
        initial_lifetime,
        initial_background,
    )

    if len(time) == 0:
        raise SystemExit("Error: the input CSV file contains no data.")

    if len(time) != len(counts):
        raise SystemExit(
            "Error: time and counts columns have different lengths."
        )

    if not np.all(np.isfinite(time)):
        raise SystemExit(
            f"Error: column '{args.time_column}' contains invalid values."
        )

    if not np.all(np.isfinite(counts)):
        raise SystemExit(
            f"Error: column '{args.counts_column}' contains invalid values."
        )

    try:
        fit_result = fit_monoexponential_decay(
            time=time,
            counts=counts,
            initial_guess=initial_guess,
        )
    except Exception as error:
        print(f"Fit status: failed")
        raise SystemExit(
            f"Error: fitting failed: {error}"
        ) from error

    print(f"Estimated lifetime: {fit_result.lifetime:.2f} ns")
    print(
        "Estimated uncertainty: "
        f"{fit_result.lifetime_std:.2f} ns"
    )
    print("Fit status: success")


def main() -> None:
    parser = ArgumentParser(
        prog="tcspc",
        description="TCSPC Lifetime Toolkit",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ---------------------------------------------------------
    # simulate
    # ---------------------------------------------------------

    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Generate a synthetic TCSPC decay",
    )

    simulate_parser.add_argument(
        "--bins",
        type=int,
        required=True,
        help="Number of time bins",
    )

    simulate_parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Start time in ns",
    )

    simulate_parser.add_argument(
        "--end-time",
        type=float,
        default=20.0,
        help="End time in ns",
    )

    simulate_parser.add_argument(
        "--amplitude",
        type=float,
        default=1000.0,
        help="Decay amplitude",
    )

    simulate_parser.add_argument(
        "--lifetime",
        type=float,
        required=True,
        help="Fluorescence lifetime in ns",
    )

    simulate_parser.add_argument(
        "--background",
        type=float,
        default=5.0,
        help="Constant background counts",
    )

    simulate_parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducible photon sampling",
    )

    simulate_parser.set_defaults(handler=run_simulation)

    # ---------------------------------------------------------
    # fit
    # ---------------------------------------------------------

    fit_parser = subparsers.add_parser(
        "fit",
        help="Fit a monoexponential decay stored in a CSV file",
    )

    fit_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input CSV file",
    )

    fit_parser.add_argument(
        "--time-column",
        type=str,
        default="time_ns",
        help="Name of the CSV column containing time values",
    )

    fit_parser.add_argument(
        "--counts-column",
        type=str,
        default="measured_counts",
        help="Name of the CSV column containing photon counts",
    )

    fit_parser.add_argument(
        "--initial-amplitude",
        type=float,
        default=None,
        help="Initial amplitude guess; derived automatically if omitted",
    )

    fit_parser.add_argument(
        "--initial-lifetime",
        type=float,
        default=None,
        help="Initial lifetime guess in ns; derived automatically if omitted",
    )

    fit_parser.add_argument(
        "--initial-background",
        type=float,
        default=None,
        help="Initial background guess; derived automatically if omitted",
    )

    fit_parser.set_defaults(handler=run_fit)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
