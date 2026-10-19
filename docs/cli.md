# Command-line interface

After [installation](../README.md#installation), invoke `tcspc` or
`python -m tcspc_toolkit`. Both call the same parser.

```bash
tcspc --help
tcspc simulate --help
tcspc fit --help
```

These commands provide unconvolved baselines. For IRF-aware Poisson inference,
use the [library quick start](../examples/quickstart.py) or
[CSV reconvolution example](../examples/experimental_csv.py).

## Simulate

```bash
tcspc simulate --bins 512 --lifetime 2.5 --random-seed 42
```

This prints the bin count and first five expected/observed counts for an ideal
mono-exponential decay plus constant background, sampled with Poisson noise.
It does not convolve an IRF, export a CSV, or accept a JSON manifest.

| Option | Required / default | Meaning |
| --- | --- | --- |
| `--bins` | Required integer | Number of samples in the time grid |
| `--start-time` | `0.0` | Start time in ns |
| `--end-time` | `20.0` | End time in ns; included in the evenly spaced grid |
| `--amplitude` | `1000.0` | Scale of the ideal exponential, not a total photon budget |
| `--lifetime` | Required float | Lifetime in ns |
| `--background` | `5.0` | Constant expected background counts/bin |
| `--random-seed` | Unset | Explicit seed for reproducible Poisson draws |

## Fit a CSV baseline

Run from the checkout root:

```bash
tcspc fit --input data/examples/ideal_tcspc_decay.csv --time-unit ns --data-kind raw_counts
```

This committed synthetic file matches the command's **unconvolved, unweighted
least-squares** model. Typical output is a lifetime of `2.49 ns`, an estimated
uncertainty about `0.0037 ns`, and `Fit status: success` (last digits depend on
the SciPy version). The uncertainty is the
least-squares covariance-derived parameter standard error, not a Poisson Fisher
interval or a model-form uncertainty estimate.

| Option | Required / default | Meaning |
| --- | --- | --- |
| `--input` | Required path | Headered single-histogram CSV |
| `--time-column` | `time_ns` | Time column name; does not infer units |
| `--counts-column` | `measured_counts` | Observation column name |
| `--time-unit` | `ns` | Declared input units: `ps`, `ns`, `us`, `ms`, or `s` |
| `--data-kind` | `raw_counts` | `raw_counts` or `processed_intensity` |
| `--initial-amplitude` | Automatic | Initial exponential amplitude |
| `--initial-lifetime` | Automatic | Initial lifetime in internal ns |
| `--initial-background` | Automatic | Initial constant background |

The canonical importer converts time to ns and validates finite, increasing,
uniform coordinates. Raw counts must be non-negative exact integers within the
`int64` range. Declare fractional/processed observations explicitly; the baseline
fit remains least squares in either case. It does not restore Poisson semantics
to transformed data.

Initial background is the minimum observation, initial amplitude is maximum
minus minimum, and initial lifetime is one fifth of the time span unless supplied.
Poor initialization or a physically inappropriate model can still lead to a
misleading result. Optimizer completion is not scientific validation.

For a compatible user-supplied file with different column names and units:

```bash
tcspc fit --input decay.csv --time-column time_ps --counts-column intensity --time-unit ps --data-kind processed_intensity
```

`decay.csv` in that last command is a user-provided input, not a bundled fixture.
There are no CLI IRF, reconvolution-objective, Bayesian, output-file, or
persistence options. Use the [example index](../examples/README.md) for those
library workflows. `--help` / `-h` is available on each parser.
