# SQLite persistence contract (Issue #9, schema v1)

The persistence module is an optional boundary around scientific results.
This document fixes the schema and serialization rules before object adapters
are added. Stage 1 provides database initialization, validated connections,
transactions, schema constraints, and private array-hashing helpers. It does
not store measurements or estimator results yet.

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
truth. The Stage-6.5 projection remains prior-free and model-conditional.

irf_sources records source identity; prepared_irfs records registration,
resampling, normalization and kernel identity. The measurement's attached IRF
is a separate relationship from the generating IRF and the IRF assumed by an
estimator. An imported sampled source is not automatically an independently
measured physical IRF. IRF model relation is declared on each result, since
the same assumption can be matched for one observation and misspecified for
another. Neither a valid kernel nor good sampling diagnostics establishes
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
available. For editable or otherwise mutable source, the later run adapter
should require an actual Git revision or source digest to distinguish code
states that share a package version. It must not substitute the current
importing checkout when importing historical results. Optional revision and
Git fields, when supplied, must be nonblank. Here version_state=known means
the producing package version is known; it does not claim a Git revision exists.
For genuinely incomplete historical evidence only, origin=historical and
version_state=unknown are allowed with an explicit reason; package, code and
Git version fields must then be NULL. Historical runs
with known producing versions must provide them. The schema enforces these
states; later adapters will verify that the supplied values describe the
actual producer. Full configuration and dependency version snapshots remain
separate from these version fields.

Issue-4 derives four named, independent random streams from unsigned 64-bit
SHA-256 prefixes. A SQLite INTEGER can store only signed 64-bit values.
The exact nonnegative seed is therefore stored as canonical decimal TEXT
without a leading plus sign or leading zeros, except for "0". This applies
to run, observation, result and uncertainty seed columns. Stream names and
derivation policy remain in run/execution provenance.

Scientific rows are immutable through the planned v1 adapters. Insert is the
default. An explicit reuse-identical option may accept an identical existing
record; a conflicting payload must raise. No silent replacement or default
upsert is planned. Scientific failures are records: fit validity, optimizer
success, numerical validation, interval validity, sampling status and
diagnostic acceptance remain separate fields.

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
NaN/Infinity tokens. Later adapters will explicitly convert supported enums,
tuples, mappings and NumPy scalars. They will reject unsupported objects
and unapproved arrays instead of stringifying or pickling them. Configuration
hashes are SHA-256 of those canonical UTF-8 JSON bytes, with the documented
serialization version.

For result scalar fields, Python None maps to SQL NULL. A known source NaN
or infinity maps to NULL and is named in nonfinite_fields_json so it remains
distinguishable from None. Finite values from rejected or invalid results are
retained. Required identity/configuration values reject nonfinite inputs.
SQLite CHECK constraints enforce structural relationships and ordinary range
rules; later adapters must also enforce finite scientific scalars and
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

No transaction encloses numerical fitting, training or sampling. Later
adapters will finish computation before their short, atomic inserts.
Parameterized SQL will bind all user-supplied values. Scientific failures
are valid data; SQL and serialization errors cause rollback.

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
