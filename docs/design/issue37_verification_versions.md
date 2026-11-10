# Reviewed fit-only verification versions

## Authorization and scope

This checkpoint implements the reviewed verification-v2 protocol and deterministic
software tests. Verification-v2 execution requires a separate reviewed lock after
the infrastructure is committed. The canonical manifest keeps
`execution_authorized=false` and `execution_commit=null`.

The numerical baseline is commit `cb3d0ad0aff42d8dbb0dc29e455dc1f25db07dcc`.
The immutable original v2 proposal has SHA-256
`3256669aaccdf8f3b7391f12c08ba7b20f67f500867184c26eb3e674b9a3c518`.
The canonical manifest preserves every scientific contract field from that
proposal. Its administrative status records the implemented infrastructure, and
its reporting metadata adds a compact historical verification-v1 baseline.

The baseline contains existing array hashes and fit summaries read from the
sealed v1 inventory and hash-verified results. It records both source hashes.
It is used only by reporting after the searches finish. Generating parameters
and historical estimates never enter optimizer initialization.

## Closed registry

`benchmarks._triexponential_verification.PROTOCOLS` is an immutable mapping of
frozen descriptors. Each descriptor binds the exact scope, manifest schema,
scientific contract hash, canonical manifest path, ordered twelve-task inventory,
output directory and report schema.

| Scope | Scientific contract SHA-256 | Report representation |
|---|---|---|
| `issue37_pr2c1_numerical_verification_v1` | `f53733d261f3cee8fa0ffe1af79455897b186335a709b9b7b1cc3d7d1b842f4b` | Historical schema-less v1 record |
| `issue37_pr2c1b_numerical_verification_v2` | `27c5238c32a9aa388f614dcc2c8a6ef2813baad985e9db3de681a031a4ddeb17` | `issue37.pr2c1b.verification-report.v2` |

The hash covers conditions, construction, IRFs, problems, configurations, tasks,
resources, scheduling, numerical acceptance, environment controls and initial
guesses. Unknown versions, cross-version schemas, changed conditions, streams,
solvers, budgets and task identities fail validation. The output namespace must
match the descriptor. The execution gate additionally compares the entire
manifest's exact byte hash and canonical digest against the approved lock.
Administrative and reporting metadata are therefore also execution-bound.

`--verification` retains v1 as its default manifest. V2 requires the explicit
canonical v2 `--manifest` argument. Fingerprinting, preparation/run receipts,
output-directory checks and report dispatch resolve the selected descriptor.
Pilot and comprehensive development retain their existing scope flags and task
inventories. Every verification task is one compact production fit.

## V2 population and budgets

The canonical protocol is in
`configs/issue37_pr2c1b_numerical_verification_v2.json`.
V00, V01 and V02 retain their reviewed physical definitions, each with one
deterministic expectation and one Poisson observation. Each problem receives
`lbfgsb:compact` and `trf:compact`, giving six problems and twelve fits.
Audits are not applicable to this protocol.

The Poisson entropy is `[37,2,20261101,3,condition_index,0]`. Historical pilot,
comprehensive development and verification-v1 retain branches 0, 1 and 2.
Three deterministic targets reuse inspected development conditions. Comparisons
with v1 are descriptive and require matching time, generating kernel, assumed
kernel, mean and observed-array hashes. Fresh Poisson draws are independent.

Each fit retains sixteen seed starts and four reserved recovery starts,
20,000 optimization model calls including 4,000 reserved for recovery,
512 initialization calls, 1,024 validation calls, 32 assessment calls and
256 diagnostic calls. The descent tolerance remains 0.001.
Each serial worker has a 60-second external deadline, five seconds for
termination and five for cleanup. The maximum worker budget is 720 seconds,
with up to 120 additional cleanup seconds across twelve timeouts.
Preparation and reporting retain their separate limits of 18 and 15 model calls.

## Input and execution identity

Stage 1 seals committed source, exact manifest bytes, the execution-relevant
environment, settings, task inventory and RNG procedure. The enforced source
inventory contains all package Python sources, the execution script and
`pyproject.toml`. The selected manifest must also match the named commit.
Python identity, dependency versions, module origins, numerical-library binaries
and thread/environment controls are enforced as before.

Stage 2 uses a fresh version-specific directory. Preparation writes a durable
marker and six individually hashed input checkpoints, atomically seals the
aggregate inventory, and re-reads every input and task mapping. Workers require
a complete verified inventory. Attempts bind the execution-lock identity and
the inventory identity, plus the exact input payload and task identity.

Local filenames remain hashes of the declared local problem/task IDs. Their
execution and directory identities distinguish versions. Copying a v1 lock,
inventory, input, attempt receipt or worker transport into a v2 store fails
verification. Resume verifies all evidence before any genuinely missing task.
Missing sealed inputs stop resume. Existing failed, timed-out, interrupted and
completed attempts remain immutable. Incomplete preparation and unconfirmed
process cleanup require separate investigation.

## Reporting

V2 retains all twelve attempts, including missing and censored outcomes, and
reports returned/best-finite objectives, raw numerical assessment, optimization
records, component recovery, work and timing. Its final-validation summary adds
probe-status counts, redundant physical candidates, unavailable feasible
directions, promotions, candidate matching, recorded final validation,
detected descent and required-recovery status. Unknown diagnostics remain null.

The existing fitter serializes the final validation record and promotions. It
does not serialize a separate terminal-pass invocation flag or every earlier
validation pass. The report identifies this as `final_candidate_record_only`.
It preserves omitted-required-recovery reasons even when final validation passes.
Numerical acceptance, objective quality and conditional component recovery have
separate fields and denominators.

The v2 historical comparison is bound to the canonical manifest's verified v1
baseline. Objective differences are populated only for matching deterministic
inputs. Poisson outcomes carry `independent_poisson_draw` and no before/after
objective difference. Original pilot comparison remains separately available.
Historical audit completeness remains 0/6 for the pilot. V1 and v2 audit status
is explicitly `not_applicable`.

## Post-commit review sequence

1. Review and locally commit the tested infrastructure and canonical manifest.
2. Use the final runtime with the manifest's one-thread controls. Set the six
   thread counts to `1`, `OMP_DYNAMIC` and `MKL_DYNAMIC` to `FALSE`, and
   `PYTHONHASHSEED` to `0`. Clear `OPENBLAS_CORETYPE`, `PYTHONPATH` and `PYTHONHOME`.
3. Run metadata-only inspection to a fresh pending lock proposal. Verify the new
   commit, committed-source status, manifest/source/environment fingerprints,
   twelve-task inventory, output namespace and resource ceilings.
4. Obtain separate maintainer execution approval. Preserve the pending proposal
   and create an approved lock with the same fingerprint and explicit review.
5. Under that authorization, prepare and verify all inputs, then run, then report.
   Check each command exit code. Preserve durable receipts and prior reports.

The future metadata-only command is

```powershell
.venv\Scripts\python.exe scripts/develop_triexponential.py inspect --verification --manifest configs/issue37_pr2c1b_numerical_verification_v2.json
```

Future `prepare`, `run` and `report` commands use the same manifest and
`--verification`, plus `--directory data/generated/issue37_pr2c1b_numerical_verification_v2`
and `--execution-lock` pointing to the separately approved v2 lock.
This checkpoint produces software-validation evidence. Scientific qualification
requires its own independently reviewed protocol and execution authorization.
