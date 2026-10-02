# SQLite persistence contract (Issue #9, schema v1)

The persistence module is an optional boundary around scientific results.
Stage 1 provides schema v1, validated connections, transactions and canonical
hashing. Stage 2 records scientific identities and provenance. Stage 3 records
classical point fits and ML/baseline predictions. Uncertainty, Bayesian
summaries and benchmark metrics do not have adapters yet.

## Scientific entities

The database has sixteen tables, each with a specific role:

| Table | Meaning |
|---|---|
| schema_metadata | Singleton schema and serialization versions plus DDL fingerprint |
| experiment_runs | One recorded workflow, its producing-version evidence and protocol |
| artifacts | References and byte hashes for external files |
| irf_sources | Original IRF trace, source kind, parameters and provenance |
| prepared_irfs | Derived or supplied kernel and explicit preparation history |
| simulation_conditions | Generating physics, separate from any observation |
| measurements | One observed histogram and its declared data kind |
| run_measurements | Membership and train/calibration/test role of an observation |
| model_versions | Reusable estimator/model specification and optional training artifact |
| model_assumptions | Assumed physical decay, observation model and fixed IRF kernel |
| lifetime_references | Trusted experimental references or prior-free pseudo-true projections |
| estimator_results | Shared per-observation point-result and execution identity |
| fit_details | Classical parameters, optimizer and numerical validation |
| uncertainty_results | Intervals, scores, covariance summaries and their validity |
| bayesian_summaries | Posterior, sampler and predictive summaries |
| benchmark_metrics | Aggregate metric facts with explicit grouping and denominators |

model_versions describes a reusable specification. Classical reconvolution
and Bayesian inference need no trained-model artifact. A trained artifact,
representation artifact or training run may be linked for an ML model. Each
actual observation-level execution belongs in estimator_results and may link
to classical, uncertainty and Bayesian detail records.

A synthetic observation may link to simulation_conditions. That table holds
finite-window signal photon budget, detector background per bin, generating
decay components and generating IRF. It never serves as a measurement's
intrinsic truth field. Experimental measurements have no generating-condition
link. Trusted experimental references and pseudo-true likelihood projections
have distinct lifetime_references kinds and foreign-key scopes. Corrected
pseudo-true references receive new versions; historical reference rows remain.
A bi-exponential generating condition has no unique mono-exponential physical
truth. A recorded bi-exponential condition must have both positive component
lifetimes and a secondary detected fraction strictly between zero and one;
missing component truth is not a complete generating condition. The Stage-6.5
projection remains prior-free and model-conditional.

irf_sources records source identity; prepared_irfs records registration,
resampling, normalization and kernel identity. The measurement's attached IRF
is a separate relationship from the generating IRF and the IRF assumed by an
estimator. An imported sampled source is not automatically an independently
measured physical IRF. IRF model relation is declared on each result, since
the same assumption can be matched for one observation and misspecified for
another. The Stage-3 adapter records this relation only when the caller
declares it; equal kernels do not establish a match. Neither a valid kernel
nor good sampling diagnostics establishes
physical model correctness.
The bare_array source representation and supplied_kernel preparation kind
make incomplete historical kernel-only evidence explicit; neither assigns a
measured or generating source kind that the record does not establish.

Raw counts and processed intensity have different measurement data kinds.
Only raw counts have observed_total_counts. Persistence does not apply a
Poisson method to processed values. Generating photon budget, observed total
counts, fitted reconvolution amplitude and background per bin use different
columns and units. Time and lifetime columns are nanoseconds; stored runtimes
are seconds and retain a named measurement scope.
Counts and cardinalities such as bin counts, photon budgets, observed totals,
artifact byte sizes, realization indices, optimizer evaluations, resamples,
Bayesian steps/draws and benchmark denominators must have SQLite integer
storage after affinity conversion. Nullable count fields accept NULL where
their scientific row contract permits it. Continuous background, amplitude,
fraction, probability, coverage, lifetime and runtime fields remain real-valued.

Complete Poisson reconvolution assumptions require a prepared IRF and exactly
one explicit temporal-shift mode: a fixed shift with no bounds, or a lower and
upper bound with no fixed shift and lower < upper. Historical contexts marked
`historical_incomplete` may retain missing or partial shift evidence.

No array, posterior chain, fitted curve, residual profile, model binary or
fitted PCA object is stored as a SQLite BLOB by default. Their rows may link
to external artifacts with path, path base, format, byte hash and locator
metadata. A path alone does not guarantee the file is still available.

## Identity and provenance

Integer primary keys are local database handles. Stable external keys identify
runs, observations, source/prepared IRFs, generating conditions, model
specifications, assumptions and references. Existing sample_id,
condition_id, pair_id, test/regime IDs, prior_policy_id, estimator names and
method IDs remain scoped scientific labels. A result is unique for its run,
measurement, model, nullable assumption and analysis_key. The uniqueness index
normalizes a NULL assumption for this purpose. Hashes validate content; they
do not silently deduplicate distinct observations.

Every new run must declare origin=new, version_state=known and its actual,
nonempty producing_package_version. A released installed package can be
recorded without a Git checkout or source digest; producing_code_revision and
producing_git_commit are optional, but should be recorded when genuinely
available. For editable or otherwise mutable source, the run adapter requires
an actual Git revision or source digest when `editable_source=True` to distinguish code
states that share a package version. It must not substitute the current
importing checkout when importing historical results. Optional revision and
Git fields, when supplied, must be nonblank. Here version_state=known means
the producing package version is known; it does not claim a Git revision exists.
For genuinely incomplete historical evidence only, origin=historical and
version_state=unknown are allowed with an explicit reason; package, code and
Git version fields must then be NULL. Historical runs
with known producing versions must provide them. The schema enforces these
states; callers must supply truthful producing evidence. Full configuration
and dependency version snapshots remain
separate from these version fields.

Issue-4 derives four named, independent random streams from unsigned 64-bit
SHA-256 prefixes. A SQLite INTEGER can store only signed 64-bit values.
The exact nonnegative seed is therefore stored as canonical decimal TEXT
without a leading plus sign or leading zeros, except for "0". This applies
to run, observation, result and uncertainty seed columns. Stream names and
derivation policy remain in run/execution provenance.

Scientific rows are immutable through the Stage-2/3 adapters. Insert is the
default. An explicit `on_duplicate="reuse_identical"` accepts an identical existing
record; a conflicting payload must raise. No silent replacement or default
upsert is planned. Scientific failures are records: fit validity, optimizer
success, numerical validation, interval validity, sampling status and
diagnostic acceptance remain separate fields.
For a shared estimator result, `is_valid=1` requires `status='available'` and a
non-NULL numeric lifetime estimate. It does not require a positive estimate:
finite zero or negative ML predictions remain valid outputs under the current
ML evaluation contract. Estimator-specific physical constraints belong at the
scientific adapter boundary. Invalid results may retain a finite estimate.

## Canonical array hashes

All new array fingerprints use SHA-256 over the canonical byte sequence of a
nonempty, one-dimensional numerical array. Hashes do not depend on native
byte order, array strides, memory order or an incidental narrower dtype.
Shape is separately stored as n_bins (including IRF sources) or grid metadata.
A caller must also compare the array's scientific role and grid; a digest by
itself is not an IRF or observation identity.

The private helpers in persistence.py define these byte representations:

| Array role | Canonical bytes and validation |
|---|---|
| Time grid in ns | C-contiguous little-endian IEEE-754 binary64 (<f8); finite, strictly increasing |
| Raw histogram counts | C-contiguous little-endian signed 64-bit integers (<i8); nonnegative, exactly integral, within int64 range |
| Continuous/processed histogram values | C-contiguous <f8; finite, negative values permitted |
| IRF source samples | C-contiguous <f8; finite and nonnegative; source time grid is hashed separately as a time grid |
| Prepared IRF kernel | C-contiguous <f8; finite and nonnegative; target time grid is hashed separately |

For new floating-array hashes, both signed-zero encodings become positive
zero before hashing. Values that are exactly equal after conversion to
binary64 hash identically. A float32 approximation of a nonrepresentable
decimal and a distinct float64 approximation are not numerically equal at
binary64 precision and need not share a digest. No rounding, smoothing,
resampling, clipping or unit conversion happens inside the hash helpers.
The caller converts physical time units to nanoseconds before hashing.
NaN and infinities are rejected for these source arrays.

Integer count arrays of different signed integer widths or byte order hash
identically if their values agree. Floating count inputs are accepted only
when every value is an exact integer below 2**53, matching
TCSPCMeasurement's raw-count boundary. Values exceeding signed int64 or
fractional counts are rejected. No conversion to floating point precedes
raw-count hashing.

Frozen Issue-4 fingerprints used SHA-256 directly on contiguous little-endian
<i8 observed counts and <f8 expected counts, time grids and IRF kernels.
The raw-count helper retains exactly that rule. A separate legacy float
helper retains the original <f8 byte rule, including any historical
signed-zero bits, for comparison with already saved Issue-4 fingerprints.
Never rewrite a frozen hash with the new signed-zero-normalized hash or
replace historical files to make a hash match. Store a legacy fingerprint in
explicit provenance alongside a new canonical fingerprint when both are
needed. The existing Stage-5/6 files and Stage-6.5 correction remain frozen.

An artifact's sha256 is the hash of the whole external file, not the hash of
an in-memory array extracted from it. These hashes have different meanings.

## Scalar and JSON serialization

Frequently grouped or filtered quantities have relational columns: lifetime,
photon budget, background, test/regime, estimator and prior policy identity,
IRF source kind, status, uncertainty method, nominal level and failure
denominators. benchmark_metrics is a metric fact table with those explicit
dimensions; it is not a general entity-attribute-value database.

Flexible configuration, provenance and small auxiliary diagnostics use
canonical JSON: sorted string keys, UTF-8, compact separators and no JSON
NaN/Infinity tokens. Stage-2 adapters convert supported dataclasses, enums,
tuples, mappings and NumPy scalars. They reject unsupported objects
and unapproved arrays instead of stringifying or pickling them. Configuration
hashes are SHA-256 of those canonical UTF-8 JSON bytes, with the documented
serialization version.

For result scalar fields, Python None maps to SQL NULL. A known source NaN
or infinity maps to NULL and is named in nonfinite_fields_json so it remains
distinguishable from None. Finite values from rejected or invalid results are
retained. Required identity/configuration values reject nonfinite inputs.
SQLite CHECK constraints enforce structural relationships and ordinary range
rules; Stage-3 result adapters enforce finite scientific scalars and
cross-table type consistency. pandas may display SQL NULL as NaN.

## Connection and transaction behavior

initialize_database(path) creates or validates the entire schema-v1 state.
For an in-memory database, pass an existing sqlite3.Connection to
initialize_database(connection) and keep it open. The function refuses an
unrelated or incomplete database and does not perform migrations.
Successful initialization sets both the singleton metadata row and SQLite
user_version=1. It checks each toolkit-owned table, named index, metadata
entry and DDL fingerprint. Additional user tables, views and non-unique query
indexes are allowed and excluded from the v1 fingerprint. A user-created
UNIQUE index on a toolkit table is rejected because it adds a write constraint.
Any trigger defined on a toolkit table is rejected because it can change
persistence behavior; triggers on user tables or views are allowed. Missing or
altered toolkit-owned objects still make the database incompatible.
Reopening a compatible database does not rewrite it.
The four constraint corrections in this document amend the unreleased schema
v1 definition. They are not a migration or schema v2: a database made from the
earlier provisional Stage-1 v1 DDL has a different toolkit definition and must
be recreated before use with this version.

Schema creation runs in a transaction. On a DDL, metadata or validation
failure, no partially initialized schema is committed. SQLite may already
have created an empty file for a path-based attempt; atomic schema state
does not promise filesystem-level removal of that file.

connect_database(path, readonly=False) opens only an existing compatible
database, using SQLite mode=rw or mode=ro, and never creates one. A read-only
connection also sets query_only. Every connection enables and verifies
PRAGMA foreign_keys=ON before transaction work. Connections have a
five-second lock timeout and autocommit at the SQLite level, leaving the
explicit transaction helper in charge of write units.

transaction(connection) uses BEGIN IMMEDIATE and commits at the top level.
Within an active transaction, it creates a uniquely named SAVEPOINT;
failure rolls back only that savepoint before propagating the exception.
Commit failure rolls back the top-level transaction. Callers close their
connections. It is an error to pass a connection already in a transaction
to initialize_database, since foreign-key activation must happen first.

No transaction encloses numerical fitting, training or sampling. Stage-2/3
multi-row adapters use the transaction/savepoint helper for short, atomic inserts.
Parameterized SQL will bind all user-supplied values. Scientific failures
are valid data; SQL and serialization errors cause rollback.

## Stage-2 recording boundary

Stage 2 exposes narrowly named record functions in `tcspc_toolkit.persistence`:
`record_run`, `record_artifact`, `record_simulation_condition`,
`record_issue4_condition`, `record_measurement`, `record_synthetic_observation`,
`record_benchmark_observation`, `link_run_measurement`, `record_irf_source`,
`record_legacy_irf_source`, `record_prepared_irf`, `record_supplied_kernel`,
`record_model_version`, `record_model_assumption`,
`record_trusted_lifetime_reference`, `record_pseudo_true_reference` and
`record_issue4_pseudo_true_reference`.
`bayesian_model_configuration` serializes reusable prior/sampler settings
without the per-observation random seed. These functions return local SQLite
IDs, not reconstructed scientific objects. They do not alter numerical modules.

Stable keys (`run_key`, `measurement_key`, source/preparation/condition/model/
assumption/reference keys and `artifact_key`) are explicit, caller-supplied
identities. A key should include a stable project/dataset/session namespace
when local labels can collide. `sample_id`, `condition_id`, `pair_id`,
`test_id`, `prior_policy_id` and estimator names are scoped descriptive labels,
not globally unique keys. In particular, an observation keeps one
`measurement_key` across later estimator, IRF-assumption, prior and uncertainty
analyses. Independent Poisson realizations need distinct measurement keys even
when their condition and local `sample_id` agree. No key is inferred from a
histogram hash, local label or hidden counter, and no retry suffix is invented.

`record_run` takes the producing package version explicitly. Git or code
revision is optional for a released package without `.git`; mutable editable
source requires an actual revision when declared. Genuinely unknown historical
producers require an explicit reason and retain NULL version fields. The
adapter never substitutes the importing checkout's version. Configurations
are canonical JSON with SHA-256 of the UTF-8 bytes; unsigned seeds remain
canonical decimal TEXT, including values above SQLite's signed int64 range.

`record_artifact` registers an external path, format/kind, file-byte SHA-256,
optional byte size and small locator metadata (for example shape/dtype). If
the byte hash is omitted, the function reads the existing file to compute it;
it never copies or embeds the file. A relative `database_directory` path is
resolved against the database file's directory; in-memory databases require
absolute artifact paths. An explicit byte hash can document an unavailable
historical file. File-byte, array-content and configuration hashes are
different evidence.

`record_simulation_condition` accepts explicit mono-, bi- or other generating
physics, including photon budget, background, shift and generating prepared
IRF. `record_issue4_condition` consumes the existing matched/mismatch Issue-4
condition objects after verifying the linked source and full prepared-IRF
identity, not merely an equal generating kernel. Additional
generating parameters remain in canonical JSON. No generating lifetime is
read from a measurement's metadata or its `sample_id`; a synthetic observation
may have no condition. Bi-exponential components stay in
`simulation_conditions`, never as a fabricated mono-exponential truth.

`record_measurement` consumes `TCSPCMeasurement`; the two synthetic observation
functions consume explicit arrays or one documented raw row from
`BenchmarkMeasurements`/`GeneralizationTestMeasurements`. They do not infer
raw counts from `BenchmarkDataset`, whose histograms may be processed. Raw
counts pass the scientific count guard, retain the unmodified histogram hash
and get an observed total. Processed intensity may contain fractional or
negative values, has a distinct hash role and has no observed photon total.
Full time/histogram arrays remain external. An experimental observation cannot
acquire a generating condition. `link_run_measurement` stores the caller's
data role and local test/regime/pair/dataset labels without rewriting their
training, calibration, frozen-evaluation or demo meaning.

`record_irf_source` accepts `SampledIRF`, `IRFProfile` and successful
`LeadingEdgeIRFResult`. Imported sampled, synthetic Gaussian/EMG and
leading-edge proxy sources keep distinct source kinds; a failed proxy does
not create a source. An estimated proxy requires the measurement from which
it was derived. A bare legacy array can be recorded only as an explicitly
unknown source kind. `record_prepared_irf` consumes `PreparedIRF`, preserves
its diagnostic and operation history, and can atomically register its source
when given a source key. One source may have multiple preparations on distinct
target grids. `record_supplied_kernel` is the explicit incomplete-provenance
path for a historical unit-area kernel. Generating IRF (`simulation_conditions`),
measurement-attached IRF (`measurements`) and inference-assumed prepared IRF
(`model_assumptions`) are independent relationships. Registration/resampling
is not a fitted temporal shift. Deliberate IRF misspecification is the relation
between generating/attached and assumed IRFs, not a fabricated source kind;
per-result relation labels are recorded by the Stage-3 result adapters.
When an existing IRF ID is supplied or an IRF key is reused, matching grid and
sample hashes are necessary but not sufficient. The adapter also checks source
kind, parameters, metadata and provenance, or the linked source and available
preparation operations/diagnostics, respectively. The equivalent
`SampledIRF`-to-imported-`IRFProfile` representation conversion remains valid;
missing estimation-result-only diagnostics are not fabricated from a source
profile.

`record_model_version` registers a reusable classical, baseline, ML or
Bayesian estimator specification. ML may link a training run and external
trained/representation artifacts; classical and Bayesian specifications need
no such artifacts. Prior policy labels alone are insufficient: Bayesian
configuration should contain the actual prior and reusable sampler settings.
At the `record_model_version` boundary, the direct Bayesian
`sampler.random_seed` field (or the seed of a directly supplied
`BayesianSamplingConfig`) is removed before configuration hashing and storage.
That recognized sampling seed identifies an execution, not reusable
prior/sampler policy. Other sampler settings remain, as do seed-named fields
at unknown or nested paths; they are not silently classified as execution
seeds. ML training seeds remain part of ML model configuration and may
identify distinct trained reusable models.
`record_model_assumption` records physical decay/observation/background and
shift policy with the assumed prepared IRF separately. Complete Poisson
reconvolution contexts require one fixed or bounded shift mode; incomplete
historical contexts may lack evidence. If given a Bayesian prior object, the
adapter checks a matching fixed shift or Bayesian bounded-prior support within
the physical assumption's bounds. A narrower prior remains distinct from the
physical shift range; prior identity is not physical IRF provenance.

`record_trusted_lifetime_reference` requires an experimental measurement and
cannot create synthetic truth. `record_pseudo_true_reference` requires a
generating condition and mono-exponential Poisson model assumption; it stores
a prior-free model-conditional projection and optional diagnostics. Corrections
use a new stable key/version and may point to the same-scope prior reference
with `supersedes_reference_id`; the older row is not overwritten. The Issue-4
wrapper accepts `PseudoTrueReference`, verifies its local condition/assumption
labels against the linked rows, and retains its numerical audit in validation
JSON; non-finite optional diagnostics become JSON null with named fields. Ordinary
known mono truth and bi-exponential component lifetimes remain in generating
conditions. Every Stage-2 duplicate key raises by default; identical reuse
compares the complete normalized payload, including parsed canonical JSON.
Changed content under a stable key raises rather than upserts.
When a pseudo-true projection provides a temporal shift, a complete fixed-shift
assumption requires agreement within `math.isclose` tolerances (`rtol=1e-7`,
`atol=1e-12` ns); a complete bounded/free-shift assumption requires an inclusive
in-bounds value. Explicitly incomplete historical assumptions do not invent a
missing shift policy. This check also applies to the direct Issue-4 wrapper and
precedes duplicate reuse; the projection shift is not generating truth.

## Stage-3 point-result boundary

One `estimator_results` row identifies one point analysis by
`(run_id, measurement_id, model_id, assumption_id or NULL, analysis_key)`.
The schema's `COALESCE(assumption_id, 0)` uniqueness index makes the NULL case
unique too. The run must already contain the measurement in `run_measurements`.
The model ID names a reusable specification, not a fitted observation; the
analysis key distinguishes intentional repeat analyses, not automatic retries.
Neither local `sample_id`, truth/reference lifetime, DataFrame index nor fitted
curve determines this identity. An IRF relation is `unspecified`, `matched` or
`deliberately_misspecified`; a non-unspecified relation requires an explicit
assumption. Proxy/estimated is an IRF source kind, not a fourth relation value.

`record_reconvolution_fit` accepts `ReconvolutionFitResult` or
`ReconvolutionCurveResult`, requires a classical model specification with an
explicit `configuration_json["objective"]` of `poisson` or `least_squares` and an
explicit mono-exponential physical assumption, and atomically writes the point
row plus one `fit_details` extension. The point summary is
`fitted_lifetime_ns`. The lower
result's `success` supplies its overall valid/source-success state; its
optimizer-reported success and numerical validation remain separate. The
curve result supplies `valid_fit`, optimizer success, boundary and recovery
flags, starting/fitted parameters, Poisson NLL/deviance and failure/exception
details. An invalid fit may retain a finite lifetime and diagnostics with
`status='failed'` and `is_valid=0`. Unavailable lower-level initial guesses,
Poisson metrics, boundary assessment and runtime stay NULL rather than being
recomputed. Valid classical fits must retain finite physical fitted parameters
and a positive lifetime, consistent with their source fit contract. The fitted
curve is not stored as a BLOB.

The optimization objective belongs to the reusable `model_versions`
configuration; `model_assumptions` separately records the physical observation
and forward-model assumptions. An explicit `poisson_reconvolution` or
`least_squares_reconvolution` observation model must agree with the model
objective. A Poisson objective requires a raw-count measurement; least squares
accepts raw counts or processed intensity. Complete reconvolution assumptions
require a prepared IRF. When an assumption links one, its prepared **target**
grid must match the measurement's bin count, canonical time-grid hash, start,
and step; the IRF source may have a different original grid. A valid classical
fit's temporal shift must agree with a complete fixed-shift assumption within
`math.isclose(rtol=1e-7, atol=1e-12 ns)` or lie within inclusive bounded-shift
limits. Failed fits retain their diagnostic shift even when outside those
limits. Explicitly incomplete historical assumptions do not supply a missing
shift policy. These context checks precede duplicate reuse.

The curve result's `runtime_ms` is converted to seconds with scope
`curve_fit_phase_after_initialization`: the source timer begins after initial
guess construction and includes more than just optimizer work. It is neither
an exact optimizer-only time nor a whole-call time, so `optimizer_seconds` and
`call_seconds` remain NULL. The lower result carries no runtime. Benchmark-wide
timing is not divided among observations. Classical and batch prediction rows
have no invented execution seed.

`record_regression_predictions` accepts `RegressionBenchmarkResult` plus an
explicit ordered `measurement_ids` sequence and a matching ML or baseline
model specification. `record_baseline_predictions` accepts the actual
constant-mean or mean-arrival prediction array with its named baseline model.
Both create only `estimator_results` rows, with summary
`predicted_lifetime_ns` and zero-based `prediction_index` in each row's
execution JSON. The ordered sequence must match prediction count, contain no
duplicate measurement ID, and refer to existing run memberships; unordered
containers such as sets are rejected. No DataFrame-index, `sample_id`, truth or
representation inference occurs. Representation identity comes from
`model_versions`. Aggregate errors, relative errors
and benchmark metrics are not copied into point rows. Official benchmark and
baseline outputs require finite predictions, but zero and negative finite
predictions remain valid under current ML evaluation semantics. Batch rows
have no invented per-sample runtime or seed.

`record_scalar_prediction` is the explicit already-fitted single-output path
for an ML or baseline model, including experimental measurements without
generating truth. Its caller supplies the source type, status/validity, and
any genuinely measured runtime scope or execution seed. A valid generic point
requires an available finite estimate, but not positivity; an invalid row may
retain a finite estimate or record a nonfinite one as NULL. Source NaN,
positive infinity and negative infinity in result scalars become SQL NULL
with `"nan"`, `"+inf"` and `"-inf"` entries, respectively, in the owning row's
`nonfinite_fields_json`. A missing source value is NULL without such an entry.

All Stage-3 writes use the Stage-2 duplicate spelling: default
`on_duplicate="raise"`; `reuse_identical` compares the complete normalized
persisted payload. Classical reuse checks both the point row and its required
`fit_details` row. A conflict or failure at any point in a batch or composed
classical write rolls back that write unit, including within an outer
savepoint. No row is silently updated or suffixed.

## Query and compatibility boundaries

The first schema supports joins across observations, generating conditions,
model specifications, assumptions, IRF source/preparation, per-observation
results, uncertainty outputs and aggregate metrics. Queries must select a
declared target and reference version. Coverage should report attempted,
valid and covering denominators separately. Importing numerical modules
must not import persistence or initialize a database.

Version 1 detects incompatible databases and stops. Migrations, database
merging, automatic external artifact storage, broad public API exports and
full dataclass reconstruction are future work. A focused SQLite example will
be fast and deterministic; Bayesian persistence will be verified with
deterministic result fixtures, without live MCMC or an emcee dependency.
