# Release readiness and quality gates

**Target: 0.8.0, unreleased.** GitHub Release first; PyPI publication is deferred.
The version reflects substantial work after 0.7.0 without a premature v1.0
API-stability commitment. No tag or release is created by CI.
Issue #13 remains deferred and is not a release blocker.

The [changelog](../CHANGELOG.md) owns release notes and breaking changes.
[Results](results.md) and the [Issue-12 acceptance ledger](issue12_scientific_acceptance.md)
own scientific evidence; passing CI does not establish experimental accuracy
or physical-model validity.

## Local and CI commands

Use a virtual environment and an editable Git checkout:

```bash
python -m pip install -e ".[dev,bayesian]"
python -m pip check
python -m ruff check src tests examples scripts
python scripts/check_format.py
python -m pytest --durations=25 --junitxml=build/reports/pytest.xml
```

The Python 3.12 test job replaces the last command with:

```bash
python -m pytest --durations=25 --junitxml=build/reports/pytest.xml --cov=tcspc_toolkit --cov-branch --cov-report=term-missing --cov-report=xml:build/reports/coverage.xml --cov-report=json:build/reports/coverage.json
```

`python -m pytest` remains the canonical full-suite command. No slow scientific
or architecture tests have been removed. Coverage measures the parent process;
do not enable subprocess auto-instrumentation, which preloads modules and
interferes with import-isolation guards.

For artifacts, start with an empty `dist/` (or move previous builds elsewhere):

```bash
python -m build
python -m twine check --strict dist/*
python scripts/check_distribution.py --dist-dir dist
```

`build` creates an sdist and then a wheel from that sdist. The checker examines
metadata, license, 61-module inventory, and source materials. It installs each
artifact into its own disposable environment outside the checkout and verifies
installed import locations, the ordered 66-name root API, the fixed 38-module
removal list, console entry point, README code and seven quick examples.
It tests base-only optional-dependency isolation, then adds the Bayesian extra
and runs the existing small cross-feature integration smoke. SQLite outputs are
temporary. Neither editable imports nor original-checkout data are used.

## CI scope

The [workflow](../.github/workflows/ci.yml) runs on PRs to `main`, pushes to `main`,
and manual dispatch, with read-only token permissions, dependency caching,
pinned official actions and explicit timeouts.

| Job | Environment | Gate |
| --- | --- | --- |
| Tests | Ubuntu 24.04 / Python 3.11 | Full pytest |
| Tests + coverage | Ubuntu 24.04 / Python 3.12 | Full pytest, branch coverage |
| Quality | Ubuntu 24.04 / Python 3.12 | Ruff lint, formatting ratchet |
| Distribution | Ubuntu 24.04 / Python 3.12 | Build, strict metadata check, isolated wheel/sdist installs |
| Week-7 exact regeneration (manual dispatch) | Windows 2025 / Python 3.12.0, NumPy 2.5.1, SciPy 1.18.0 | Qualified-reference candidate; exact original fingerprint |

Test reports, dependency versions and distribution logs are uploaded for review.
Local Windows results do not qualify Linux or Python 3.11. The first successful
remote run is a required acceptance gate, not something local execution can replace.
Newer Python versions remain declared by `>=3.11` but are not in this CI matrix.

Actions were checked against official release tags and action metadata:
[checkout v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1),
[setup-python v7.0.0](https://github.com/actions/setup-python/releases/tag/v7.0.0),
and [upload-artifact v7.0.2](https://github.com/actions/upload-artifact/releases/tag/v7.0.2).
Updates require an explicit reviewed SHA change; the workflow does not publish.

## Week-7 bitwise and numerical reproducibility

The [second PR #25 diagnostic run](https://github.com/moryev/tcspc-lifetime-toolkit/actions/runs/39629791895)
captured all 810 frozen Week-7 simulations through the existing library path.
Its Python 3.11 and 3.12 test artifacts were compared with the saved Windows
reference trace. The 400-bin sample #770 **Windows realization** remains fixed
in [`tests/fixtures/week7_sample_770_windows_reference.json`](../tests/fixtures/week7_sample_770_windows_reference.json)
with SHA-256 `035f408aa945ef4df47b43ab8c7cada90e05877f5f029323804d1c8a3e1e95ca`.
It is the unchanged input to the fitter regression on every platform; no Linux
hash replaces it.

| Comparison with Windows 11 / Python 3.12.0 / NumPy 2.5.1 / SciPy 1.18.0 | Largest expected-count difference, all 810 × 400 bins | First count difference | First RNG-state difference before a sample |
| --- | ---: | --- | --- |
| Ubuntu / Python 3.11.17 / NumPy 2.4.6 / SciPy 1.17.1 | 1.82 × 10⁻¹² counts | sample 2, bin 17 | sample 2 |
| Ubuntu / Python 3.12.15 / NumPy 2.5.4 / SciPy 1.18.1 | 4.55 × 10⁻¹³ counts | sample 407, bin 16 | sample 499 |

Parameter order and non-realization metadata agree. The ideal-decay and
unnormalized Gaussian IRF differences are at most 1.11 × 10⁻¹⁶; normalized
IRF differences are at most 2.22 × 10⁻¹⁶. Against Linux 3.11, sample 1/bin 399
has expected rate 2.342 × 10⁻¹⁷ counts on Windows and exactly zero on Linux;
both draw zero,
but the NumPy Poisson generator consumes a different random stream, so sample
2 starts from a different RNG state. Against Linux 3.12, the streams entering
sample 407 agree, while bin 16 has expected rates 0.039888822072901696 and
0.03988882207290147: its sampled count changes from zero to one. The RNG
states still agree before sample 408. This is floating-point/Poisson-threshold
sensitivity, not a meaningful difference in the deterministic physical model.

Three distinct checks now apply:

1. **Fixture integrity:** assert the original count-array hash exactly.
2. **Portable simulation:** check the frozen parameter order, raw nonnegative
   integer counts, photon-budget/background metadata, and reference expected
   counts for samples 1, 407 and 770. The per-bin criterion is `rtol=1e-13`,
   `atol=1e-11` counts: the absolute allowance exceeds the largest observed
   cross-platform difference by a factor above five, yet is negligible beside
   one photon or Poisson counting variability.
3. **Exact stochastic regeneration:** run only in the pinned Windows reference
   environment with `TCSPC_VERIFY_WEEK7_REFERENCE=1` and compare the newly
   generated sample #770 with the original hash and array. The dedicated
   `workflow_dispatch` job is an explicitly configured Windows runner
   **candidate**; it becomes a qualified remote reference environment only
   after an actual passing run. A failure requires investigation, not a new
   accepted hash.

Bitwise identity of a long seeded Poisson stream is environment-specific;
scientific numerical equivalence concerns the physical expected-count model
and the fixed realization used to test fitting. Neither establishes
experimental validity. Original Issue-12 references, seeds, fitter tolerances
and frozen protocols are unchanged.

## Lint and formatting policy

Ruff **0.17.1** is pinned in the development extra and `required-version`.
Lint uses `E4,E7,E9,F`; import sorting and root-export sorting are not enabled.
Notebooks, data/reference artifacts and generated build output are excluded from
Ruff; their scientific and import guards remain in pytest.

The initial audit found 22 lint findings across 11 files and 136 unformatted
Python files. Small lint corrections precede the one-time formatting baseline;
there is no repository-wide rewrite.
The posterior module was subsequently formatted when correcting its deferred
annotation lint exceptions, reducing the baseline to 135 files. Its Python AST
is identical to the pre-change source. Two narrow `F821` exceptions preserve
the guarded, strictly deferred sampler dependency; the reporting script's `F841`
exception documents a local variable used inside a pandas query string.

`.ruff-format-baseline.json` records SHA-256 hashes of that reviewed debt.
`check_format.py` runs Ruff on all four Python source/test/example/script scopes.
Only exactly unchanged debt can remain unformatted; new files and edits to old
files must be formatted. Hashes normalize CRLF to LF, with no whitespace-content
normalization. Once a file is formatted, remove its obsolete baseline entry.

The checker has **no refresh command**. Local runs compare exemptions with HEAD;
CI compares with the PR base or the pre-push commit. Existing exemptions may
only be removed, never added or rehashed. A missing baseline in the base commit
allows the initial, explicitly reviewed Issue-15 introduction only. Missing Git
revisions fail rather than bypassing validation.

```bash
python -m ruff format path/to/changed_file.py
python scripts/check_format.py --base-ref origin/main
```

A Ruff upgrade requires a separate reviewed policy migration: change the tool
pin, assess formatting changes, and explicitly revise the ratchet policy if
needed. Do not refresh hashes to make a failed check disappear. Review policy
code/baseline changes just as other CI configuration changes.

## Coverage policy and regression priorities

Stage A Windows/Python 3.12 measurements were **82.34% combined**, **86.88%
statements**, **70.14% branches**. The candidate floor is **80% combined**,
configured for the Python 3.12 job. Final acceptance of that floor is pending
successful Linux CI. The second PR #25 run measured **82.58%** Linux combined
coverage, above the floor, while the former bitwise hash assertion still failed.
Investigate material future differences rather than lowering the floor to pass.

No scientific modules, benchmark code or inconvenient branches are omitted.
Coverage is a change detector, not a certificate of scientific correctness.
Focused added tests cover invalid Poisson inputs, invalid prepared kernels,
Bayesian autocorrelation exceptions, and raw/Pearson residual behavior.
Broader model-mismatch evidence retains its separate reference and population
policies. New tests should address failure semantics or scientific regressions,
not merely raise the aggregate percentage.

## Dependency qualification

| Runtime dependency | Minimum | Rationale |
| --- | --- | --- |
| NumPy | 2.0.0 | `np.trapezoid` is used by the scientific code |
| SciPy | 1.13.0 | NumPy 2 compatible release |
| pandas | 2.2.2 | NumPy 2 compatible wheels |
| Matplotlib | 3.8.4 | NumPy 2 compatible release |
| scikit-learn | 1.4.2 | NumPy 2 compatible release |
| emcee (optional) | >=3.1.6,<4 | Reviewed Bayesian interface |

See [NumPy's API](https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html)
and [ecosystem compatibility tracking](https://github.com/numpy/numpy/issues/26191).
The exact five runtime minima together passed `pip check`, quick start, CSV,
features, ML and uncertainty smokes on Windows/Python 3.12 in the audit.
This is **smoke compatibility**, not complete minimum-dependency qualification.
The normal CI matrix resolves current compatible releases. Full minimum-stack
testing and qualification of other Python/platform combinations remain optional
follow-up work; do not advertise them as completed.

## Separate scientific qualification

Run the genuine Bayesian example separately:

```bash
python examples/bayesian_inference.py
```

Allow approximately 1–2 minutes. Keep its priors, seeds, extension budget and
sampling diagnostics unchanged; a provisional result is not accepted inference.
Execute Notebook 15 from a fresh kernel with the documented inputs and save only
a temporary output copy. Do not overwrite historical notebook outputs or the
quick-start figure during release qualification.

Long reference studies, full Bayesian example sampling and long notebooks are
outside routine CI. The [reproducibility guide](reproducibility.md) owns their
inputs, commands and cost.

## Publication checklist

- Review local validation and the complete diff, including frozen scientific
  contracts and metadata consistency.
- Obtain successful actual GitHub runs on the proposed PR; confirm the Linux
  coverage floor and inspect warnings, reports and dependency versions.
- Obtain separate approval for Git history actions and publication.
- At publication, replace the citation's unreleased message and add the actual
  `date-released`; date the 0.8.0 changelog entry consistently.
- Only then create the approved `v0.8.0` tag/GitHub Release and attach qualified
  artifacts. Rebuild from the approved release commit. PyPI remains deferred.

Current metadata uses version `0.8.0` in `pyproject.toml` and `CITATION.cff`;
the citation deliberately has no release date. No old 0.7.0 date is reused.
