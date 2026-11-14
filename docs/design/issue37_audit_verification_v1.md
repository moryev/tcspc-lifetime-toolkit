# Issue #37 audit-only verification v1 infrastructure

## Closed protocol

The dedicated `--audit-verification` CLI flag recognizes exactly
`issue37_pr2c2_audit_verification_v1`. It is mutually exclusive with `--pilot`
and `--verification`. The private audit-verification adapter holds a read-only
registry containing this one descriptor. Unknown versions fail validation.

The canonical manifest is
`configs/issue37_pr2c2_audit_verification_v1.json`, with SHA-256
`33994d9a4c4d964c8b8c5c3c35a5563f2695087b92d763cd14de0adf003bd8fc`.
Its pinned scientific contract has SHA-256
`e712329cbafca271ce6572e93bc383892d563485f53edbfd0d3b674baf9b0b23`.
The implementation independently pins this digest, ordered task identities,
schema and output directory. Manifest claims cannot replace the compiled
contract. Actual execution additionally requires the exact manifest bytes and
committed source/environment identity in an approved execution lock.

The historical proposal remains at
`docs/validation/issue37_pr2c2_audit_verification_v1_manifest_proposal.json`, with
SHA-256 `b15c7bf828d32a7f0ae4c7c360d177b59243e388275507eb8a0addc73750ec93`.
The canonical file preserves its scientific contract while recording the
implemented infrastructure status and historical proposal reference.
`execution_authorized` remains false in the canonical manifest.

## Inputs, tasks and work

The six problems are `A00:mean:matched`, `A00:poisson-0:matched`,
`A01:mean:matched`, `A01:poisson-0:matched`, `A02:mean:matched` and
`A02:poisson-0:matched`. Each schedules exactly one task by appending `:audit`.
Observation IDs omit `:matched` and are distinct for each deterministic or
Poisson target. A00–A02 preserve the inspected V00–V02 physical definitions,
amplitudes, bounds and IRF assumptions. Poisson entropy is
`[37,2,20261101,4,condition_index,0]`. Branches 0–3 retain their existing roles.

Each audit receives the existing 16 TRF and eight Powell starts, 40,000 total
direct-model calls, 39,742 search calls, 256 validation calls and two assessment
calls. The fixed invocation leases remain 1,656 calls for the first 22 starts
and 1,655 for the final two. The external worker deadline is 120 seconds.
Serial execution gives a 720-second aggregate worker ceiling and a 60-second
aggregate cleanup allowance. Unconfirmed process termination stops execution.

The adapter reuses the existing counted preparation function. Each input
requires three K=1 forward calls to calculate unit-component curves and verify
the declared amplitudes. Mean targets use the weighted sum of those curves.
Poisson targets sample that same constructed expectation. Each observation
identity is prepared once. Six unique identities therefore require at most
18 forward calls and 18 charged component convolutions. Interrupted work keeps
its actual preparation record. This budget is separate from audit-worker work.

The audit dispatch passes only grid, observed values, assumed kernel, bounds
and explicit audit settings to the independent audit. Generating parameters and
historical production estimates are excluded from initialization.

## Evidence and resume

`inspect` performs metadata resolution and source/environment fingerprinting.
Its output is a pending lock with `execution_authorized=false`. The fingerprint
covers every production Python module, the runner, `pyproject.toml`, manifest
bytes, resolved settings/task inventory, interpreter and dependency identities,
loaded numerical libraries and thread controls. Uncommitted source cannot pass
the ordinary execution gate after toggling an authorization field.

`prepare` requires a fresh output directory and an approved matching lock. It
uses the existing immutable input checkpoints and atomic aggregate inventory
seal. Every input and task association is verified before preparation returns.
`run` and each worker reverify the complete inventory before accessing inputs.

The existing durable start receipts, worker receipts, input/execution bindings
and atomic result publication apply to this scope. Missing or corrupted inputs
block execution and resume. Failed, completed, timed-out or interrupted attempts
retain their identities. Resume executes genuinely missing tasks only. An
unconfirmed running/interrupted worker requires reconciliation before further
execution. Cross-scope locks, inputs and attempts fail the same integrity gates.

## Sealed-record reporting

The reporter performs zero physical-model calls. It verifies the lock, complete
input inventory and attempts, then checks successful parent results against
their immutable worker transports. Source numerical assessments and endpoint
checks remain recorded evidence. Reporting provides no additional objective
recomputation.

The report retains all six task rows and all 144 scheduled start rows. Lost
worker results carry unknown start activity and unknown work. A saved worker
payload remains inspectable even if its parent timed out or was interrupted,
while the parent outcome remains visible. Censored runtime comes from the
parent's external-worker timing record.

Per-start trial classification uses complete counters for finite objectives,
valid positive infinity, infeasibility and numerical failures. A null serialized
objective supplies no status inference. Retained derivative diagnostics are
explicitly partial when records were dropped. Raw start records preserve
endpoint and finite-best objectives, solver status, fixed/free coordinates,
actual calls, leases, elapsed time and diagnostic truncation. Task records retain
phase work, charged/completed component work, unknown partial work, exact
assessment/validation candidate identities and scalar-validation details.

Strict completion requires the recorded successful search and all established
assessment criteria. Contradictory completion claims are exposed as
`inconsistent_completion_record`. A finite validated candidate alone is
insufficient. Parent failure/interruption also prevents a completed task verdict.
Production-fit metrics and production-minus-audit gaps have explicit
`not_applicable` status. All attempt denominators retain failures and missing
results. These inspected conditions provide operational development evidence.

## Validation boundaries

Tests enumerate the real protocol through metadata-only calls. Numerical and
store/CLI tests use tiny synthetic inputs and mocked workers. They exercise
preparation, sealing, cross-scope rejection, corrupt transports, immutable
attempts, timeouts, incomplete work, zero-model reporting and closed-registry
enforcement. The numerical audit, production fitters, source physics, IRFs,
existing scientific manifests, root API, SQLite schema and release metadata
retain their established contracts.

## Post-commit handoff

Proposed local commit message is
`feat: add closed audit-only verification infrastructure`.

After maintainer review and a separately authorized commit, establish the
one-thread environment below and generate a new pending lock with `inspect`.
Review its exact committed source, environment, manifest, tasks and limits.
Maintainer execution approval must then be recorded in a distinct approved lock
using the existing authorization schema. Preserve the pending artifact and
verify the ordinary gate against the approved lock before input preparation.

Future PowerShell commands, from the repository root, are shown below. The
approved-lock filename is reserved for that later approval step.
Each command checks the native exit code and parses JSON before creating its
output file. Exclusive UTF-8 file creation preserves existing evidence. Failed
commands leave their durable attempt records in the execution directory.

```powershell
$repo = 'C:\Users\Yevhenii\PycharmProjects\scientific-python-transition'
Set-Location -LiteralPath $repo
$python = Join-Path $repo '.venv\Scripts\python.exe'
$runner = Join-Path $repo 'scripts\develop_triexponential.py'
$manifest = Join-Path $repo 'configs\issue37_pr2c2_audit_verification_v1.json'
$runDirectory = Join-Path $repo 'data\generated\issue37_pr2c2_audit_verification_v1'
$pending = Join-Path $repo 'docs\validation\issue37_pr2c2_audit_verification_v1_execution_lock_proposal.json'
$approved = Join-Path $repo 'docs\validation\issue37_pr2c2_audit_verification_v1_execution_lock_approved.json'
$report = Join-Path $repo 'docs\validation\issue37_pr2c2_audit_verification_v1_results.json'
$prepareReceipt = Join-Path $repo 'docs\validation\issue37_pr2c2_audit_verification_v1_prepare_receipt.json'
$runReceipt = Join-Path $repo 'docs\validation\issue37_pr2c2_audit_verification_v1_run_receipt.json'
$ErrorActionPreference = 'Stop'

$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:VECLIB_MAXIMUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:BLIS_NUM_THREADS = '1'
$env:OMP_DYNAMIC = 'FALSE'
$env:MKL_DYNAMIC = 'FALSE'
$env:PYTHONHASHSEED = '0'
Remove-Item Env:OPENBLAS_CORETYPE,Env:PYTHONPATH,Env:PYTHONHOME -ErrorAction SilentlyContinue

function Save-AuditJson {
    param([string[]]$Arguments, [string]$OutputPath)
    if (Test-Path -LiteralPath $OutputPath) { throw "Output already exists $OutputPath" }
    $lines = & $python $runner @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) { throw "Audit CLI failed with exit code $exitCode" }
    $json = [string]::Join([Environment]::NewLine, [string[]]$lines)
    $parsed = ConvertFrom-Json -InputObject $json -ErrorAction Stop
    if ($null -eq $parsed) { throw 'CLI returned empty JSON' }
    if ($Arguments[0] -eq 'inspect') {
        if ($parsed.execution_authorized -ne $false) { throw 'Expected pending lock' }
        $parsed.maintainer_review = [pscustomobject]@{
            status = 'pending'
            execution_approval = 'awaiting separate maintainer authorization'
        }
        $json = ConvertTo-Json -InputObject $parsed -Depth 100
    }
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($json + [Environment]::NewLine)
    $stream = [IO.File]::Open($OutputPath, [IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $stream.Write($bytes, 0, $bytes.Length) }
    finally { $stream.Dispose() }
}

# Metadata-only preview after the reviewed local commit
Save-AuditJson -Arguments @('inspect', '--audit-verification', '--manifest', $manifest) -OutputPath $pending

# Following separate review of the pending lock and execution authorization
$executionArgs = @('--audit-verification', '--manifest', $manifest,
    '--directory', $runDirectory, '--execution-lock', $approved)
Save-AuditJson -Arguments (@('prepare') + $executionArgs) -OutputPath $prepareReceipt
Save-AuditJson -Arguments (@('run') + $executionArgs) -OutputPath $runReceipt
Save-AuditJson -Arguments (@('report') + $executionArgs) -OutputPath $report
```

Preparation/reporting outputs belong outside the sealed checkpoint directory.
An explicitly reviewed resume uses a fresh receipt filename and the existing
sealed directory. Existing attempts retain their recorded terminal outcomes.
Future execution approval covers the reviewed source and lock identity. Any
subsequent source or environment change requires a fresh lock review.
