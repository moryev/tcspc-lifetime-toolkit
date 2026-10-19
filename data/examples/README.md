# Experimental-style CSV example

All three CSVs in this directory are synthetic demonstration inputs.
For executable import/reconvolution commands, see the
[example index](../../examples/README.md#2-experimental-style-csv-import).

`experimental_style_decay.csv` and `experimental_style_irf.csv` are **synthetic**
demonstration files, not experimental observations. They exercise the same
import and estimation path intended for a compatible exported TCSPC histogram.
No trusted experimental reference lifetime is supplied by these files.

- The decay file has `time_ps,counts`: 81 uniform bins from 0 to 20 ns,
  expressed in picoseconds, with non-negative integer photon counts.
- The separate IRF file has `time_ns,irf`: the same 81-bin physical grid,
  expressed in nanoseconds, with non-negative sampled IRF amplitudes. The
  imported amplitudes are preserved; the fitting adapter normalizes the IRF
  only for its forward model.
- The synthetic generator used a Gaussian IRF centred at 1.0 ns with
  0.6 ns FWHM and amplitude 10, a mono-exponential lifetime of 3.0 ns,
  amplitude 600 on the unit-area-IRF-convolved decay, and a constant
  background of 2 expected counts per bin. Poisson counts were sampled with
  NumPy's `default_rng(42)`.

The generator settings document fixture provenance; they are **not** attached
as ground truth to the imported `TCSPCMeasurement` or passed to its estimator.
The workflow is demonstrated in
[`notebooks/15_experimental_tcspc_workflow.ipynb`](../../notebooks/15_experimental_tcspc_workflow.ipynb).

## Ideal unconvolved baseline

`ideal_tcspc_decay.csv` contains a synthetic ideal mono-exponential decay with
constant background and Poisson observations, without IRF convolution. Its
columns are `time_ns`, `expected_counts`, `measured_counts`, and
`lifetime_true_ns`. Use `measured_counts` as the observation; generating
expectations and the 2.5 ns lifetime column are synthetic provenance.
The [CLI baseline example](../../docs/cli.md#fit-a-csv-baseline) selects those
observation and time columns explicitly through its defaults. This file is not
an experimental measurement or an IRF-aware fitting demonstration.
