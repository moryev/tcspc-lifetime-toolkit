"""Optional SQLite schema and connection support for TCSPC result persistence.

This module does not import, initialize, or alter the numerical workflows.
Scientific-object adapters are intentionally deferred beyond Issue-9 Stage 1.
"""

from __future__ import annotations

import hashlib
import itertools
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.typing import ArrayLike


SCHEMA_VERSION = 1
_SCHEMA_NAME = "tcspc_lifetime_toolkit"
_SERIALIZATION_VERSION = 1
_SAVEPOINT_NUMBERS = itertools.count(1)


class PersistenceSchemaError(ValueError):
    """An existing database is not a compatible TCSPC persistence database."""


# All statements belong to one schema version. Existing databases are validated
# before use; CREATE IF NOT EXISTS would conceal an incomplete or altered schema.
_TABLES: tuple[tuple[str, str], ...] = (
    (
        "schema_metadata",
        """
        CREATE TABLE schema_metadata (
            singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
            schema_name TEXT NOT NULL CHECK (schema_name = 'tcspc_lifetime_toolkit'),
            schema_version INTEGER NOT NULL CHECK (schema_version = 1),
            serialization_version INTEGER NOT NULL
                CHECK (serialization_version = 1),
            definition_sha256 TEXT NOT NULL CHECK (length(definition_sha256) = 64),
            created_at_utc TEXT NOT NULL
        )
        """,
    ),
    (
        "experiment_runs",
        """
        CREATE TABLE experiment_runs (
            run_id INTEGER PRIMARY KEY CHECK (run_id > 0),
            run_key TEXT NOT NULL UNIQUE CHECK (length(trim(run_key)) > 0),
            run_type TEXT NOT NULL CHECK (length(trim(run_type)) > 0),
            status TEXT NOT NULL
                CHECK (status IN ('complete', 'partial', 'failed')),
            recorded_at_utc TEXT NOT NULL,
            origin TEXT NOT NULL CHECK (origin IN ('new', 'historical')),
            version_state TEXT NOT NULL CHECK (version_state IN ('known', 'unknown')),
            producing_package_version TEXT,
            producing_code_revision TEXT,
            producing_git_commit TEXT,
            producing_version_unknown_reason TEXT,
            config_json TEXT NOT NULL,
            config_sha256 TEXT NOT NULL CHECK (length(config_sha256) = 64),
            protocol_id TEXT,
            protocol_version TEXT,
            profile TEXT,
            base_seed_decimal TEXT,
            seed_namespace TEXT,
            dependency_versions_json TEXT NOT NULL DEFAULT '{}',
            seed_policy_json TEXT NOT NULL DEFAULT '{}',
            notes TEXT,
            source_run_id INTEGER REFERENCES experiment_runs(run_id)
                ON DELETE RESTRICT,
            CHECK (
                (version_state = 'known'
                    AND producing_package_version IS NOT NULL
                    AND length(trim(producing_package_version)) > 0
                    AND (producing_code_revision IS NULL
                        OR length(trim(producing_code_revision)) > 0)
                    AND (producing_git_commit IS NULL
                        OR length(trim(producing_git_commit)) > 0)
                    AND producing_version_unknown_reason IS NULL)
                OR
                (origin = 'historical'
                    AND version_state = 'unknown'
                    AND producing_package_version IS NULL
                    AND producing_code_revision IS NULL
                    AND producing_git_commit IS NULL
                    AND producing_version_unknown_reason IS NOT NULL
                    AND length(trim(producing_version_unknown_reason)) > 0)
            ),
            CHECK (
                base_seed_decimal IS NULL OR (
                    length(base_seed_decimal) > 0
                    AND base_seed_decimal NOT GLOB '*[^0-9]*'
                    AND (base_seed_decimal = '0'
                        OR base_seed_decimal GLOB '[1-9]*')
                )
            )
        )
        """,
    ),
    (
        "artifacts",
        """
        CREATE TABLE artifacts (
            artifact_id INTEGER PRIMARY KEY CHECK (artifact_id > 0),
            artifact_key TEXT NOT NULL UNIQUE
                CHECK (length(trim(artifact_key)) > 0),
            producing_run_id INTEGER REFERENCES experiment_runs(run_id)
                ON DELETE RESTRICT,
            path TEXT NOT NULL CHECK (length(trim(path)) > 0),
            path_base TEXT NOT NULL
                CHECK (path_base IN ('database_directory', 'absolute')),
            artifact_kind TEXT NOT NULL CHECK (length(trim(artifact_kind)) > 0),
            format TEXT NOT NULL CHECK (length(trim(format)) > 0),
            sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
            byte_size INTEGER CHECK (byte_size IS NULL OR byte_size >= 0),
            locator_json TEXT NOT NULL DEFAULT '{}'
        )
        """,
    ),
    (
        "irf_sources",
        """
        CREATE TABLE irf_sources (
            irf_source_id INTEGER PRIMARY KEY CHECK (irf_source_id > 0),
            source_key TEXT NOT NULL UNIQUE CHECK (length(trim(source_key)) > 0),
            source_representation TEXT NOT NULL
                CHECK (source_representation IN
                    ('sampled_irf', 'profile', 'bare_array')),
            source_kind TEXT CHECK (source_kind IS NULL OR source_kind IN
                ('synthetic_gaussian', 'synthetic_emg', 'imported_sampled',
                 'leading_edge_estimate')),
            n_bins INTEGER NOT NULL CHECK (n_bins > 1),
            source_grid_sha256 TEXT NOT NULL
                CHECK (length(source_grid_sha256) = 64),
            source_values_sha256 TEXT NOT NULL
                CHECK (length(source_values_sha256) = 64),
            source_artifact_id INTEGER REFERENCES artifacts(artifact_id)
                ON DELETE RESTRICT,
            derived_from_measurement_id INTEGER REFERENCES measurements(measurement_id)
                ON DELETE RESTRICT,
            source_parameters_json TEXT NOT NULL DEFAULT '{}',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            provenance_json TEXT NOT NULL DEFAULT '{}',
            CHECK (
                (source_representation = 'bare_array' AND source_kind IS NULL)
                OR (source_representation <> 'bare_array'
                    AND source_kind IS NOT NULL)
            )
        )
        """,
    ),
    (
        "prepared_irfs",
        """
        CREATE TABLE prepared_irfs (
            prepared_irf_id INTEGER PRIMARY KEY CHECK (prepared_irf_id > 0),
            preparation_key TEXT NOT NULL UNIQUE
                CHECK (length(trim(preparation_key)) > 0),
            irf_source_id INTEGER REFERENCES irf_sources(irf_source_id)
                ON DELETE RESTRICT,
            preparation_kind TEXT NOT NULL
                CHECK (preparation_kind IN
                    ('explicit', 'legacy_same_grid_normalization',
                     'supplied_kernel')),
            time_grid_sha256 TEXT NOT NULL
                CHECK (length(time_grid_sha256) = 64),
            kernel_sha256 TEXT NOT NULL CHECK (length(kernel_sha256) = 64),
            kernel_artifact_id INTEGER REFERENCES artifacts(artifact_id)
                ON DELETE RESTRICT,
            n_bins INTEGER NOT NULL CHECK (n_bins > 1),
            time_start_ns REAL NOT NULL,
            time_step_ns REAL NOT NULL CHECK (time_step_ns > 0),
            registration_offset_ns REAL,
            resampling_method TEXT
                CHECK (resampling_method IS NULL OR resampling_method IN
                    ('none', 'linear')),
            support_loss_fraction REAL
                CHECK (support_loss_fraction IS NULL
                    OR support_loss_fraction BETWEEN 0 AND 1),
            normalization_factor REAL
                CHECK (normalization_factor IS NULL
                    OR normalization_factor > 0),
            source_fwhm_ns REAL,
            target_fwhm_ns REAL,
            target_bins_per_fwhm REAL,
            diagnostics_json TEXT NOT NULL DEFAULT '{}',
            operations_json TEXT NOT NULL DEFAULT '[]',
            flags_json TEXT NOT NULL DEFAULT '[]',
            CHECK (
                preparation_kind = 'supplied_kernel'
                OR irf_source_id IS NOT NULL
            ),
            CHECK (
                preparation_kind <> 'explicit'
                OR (registration_offset_ns IS NOT NULL
                    AND resampling_method IS NOT NULL
                    AND support_loss_fraction IS NOT NULL
                    AND normalization_factor IS NOT NULL)
            )
        )
        """,
    ),
    (
        "simulation_conditions",
        """
        CREATE TABLE simulation_conditions (
            condition_pk INTEGER PRIMARY KEY CHECK (condition_pk > 0),
            condition_key TEXT NOT NULL UNIQUE
                CHECK (length(trim(condition_key)) > 0),
            condition_id TEXT NOT NULL CHECK (length(trim(condition_id)) > 0),
            origin_run_id INTEGER REFERENCES experiment_runs(run_id)
                ON DELETE RESTRICT,
            generating_model TEXT NOT NULL
                CHECK (generating_model IN
                    ('monoexponential', 'biexponential', 'other')),
            generating_irf_id INTEGER REFERENCES prepared_irfs(prepared_irf_id)
                ON DELETE RESTRICT,
            mono_lifetime_ns REAL,
            primary_lifetime_ns REAL,
            secondary_lifetime_ns REAL,
            secondary_detected_fraction REAL,
            signal_photon_count INTEGER NOT NULL
                CHECK (signal_photon_count > 0),
            background_per_bin REAL NOT NULL
                CHECK (background_per_bin >= 0),
            true_temporal_shift_ns REAL NOT NULL,
            true_reconvolution_amplitude REAL,
            expected_counts_sha256 TEXT
                CHECK (expected_counts_sha256 IS NULL
                    OR length(expected_counts_sha256) = 64),
            expected_counts_artifact_id INTEGER REFERENCES artifacts(artifact_id)
                ON DELETE RESTRICT,
            generating_parameters_json TEXT NOT NULL DEFAULT '{}',
            CHECK (
                (generating_model = 'monoexponential'
                    AND mono_lifetime_ns IS NOT NULL
                    AND mono_lifetime_ns > 0
                    AND primary_lifetime_ns IS NULL
                    AND secondary_lifetime_ns IS NULL
                    AND secondary_detected_fraction IS NULL)
                OR
                (generating_model = 'biexponential'
                    AND mono_lifetime_ns IS NULL
                    AND primary_lifetime_ns > 0
                    AND secondary_lifetime_ns > 0
                    AND secondary_detected_fraction >= 0
                    AND secondary_detected_fraction < 1)
                OR
                (generating_model = 'other'
                    AND mono_lifetime_ns IS NULL)
            )
        )
        """,
    ),
    (
        "measurements",
        """
        CREATE TABLE measurements (
            measurement_id INTEGER PRIMARY KEY CHECK (measurement_id > 0),
            measurement_key TEXT NOT NULL UNIQUE
                CHECK (length(trim(measurement_key)) > 0),
            origin_run_id INTEGER REFERENCES experiment_runs(run_id)
                ON DELETE RESTRICT,
            condition_pk INTEGER REFERENCES simulation_conditions(condition_pk)
                ON DELETE RESTRICT,
            attached_irf_source_id INTEGER REFERENCES irf_sources(irf_source_id)
                ON DELETE RESTRICT,
            histogram_artifact_id INTEGER REFERENCES artifacts(artifact_id)
                ON DELETE RESTRICT,
            source_type TEXT NOT NULL
                CHECK (source_type IN ('synthetic', 'experimental', 'unknown')),
            data_kind TEXT NOT NULL
                CHECK (data_kind IN ('raw_counts', 'processed_intensity')),
            sample_id TEXT,
            n_bins INTEGER NOT NULL CHECK (n_bins > 1),
            time_start_ns REAL NOT NULL,
            time_step_ns REAL NOT NULL CHECK (time_step_ns > 0),
            time_grid_sha256 TEXT NOT NULL
                CHECK (length(time_grid_sha256) = 64),
            values_sha256 TEXT NOT NULL CHECK (length(values_sha256) = 64),
            observed_total_counts INTEGER,
            observation_seed_decimal TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            provenance_json TEXT NOT NULL DEFAULT '{}',
            CHECK (source_type <> 'experimental' OR condition_pk IS NULL),
            CHECK (
                (data_kind = 'raw_counts'
                    AND observed_total_counts IS NOT NULL
                    AND observed_total_counts >= 0)
                OR
                (data_kind = 'processed_intensity'
                    AND observed_total_counts IS NULL)
            ),
            CHECK (
                observation_seed_decimal IS NULL OR (
                    length(observation_seed_decimal) > 0
                    AND observation_seed_decimal NOT GLOB '*[^0-9]*'
                    AND (observation_seed_decimal = '0'
                        OR observation_seed_decimal GLOB '[1-9]*')
                )
            )
        )
        """,
    ),
    (
        "run_measurements",
        """
        CREATE TABLE run_measurements (
            run_id INTEGER NOT NULL REFERENCES experiment_runs(run_id)
                ON DELETE RESTRICT,
            measurement_id INTEGER NOT NULL REFERENCES measurements(measurement_id)
                ON DELETE RESTRICT,
            data_role TEXT NOT NULL CHECK (length(trim(data_role)) > 0),
            test_id TEXT,
            regime_id TEXT,
            pair_id TEXT,
            realization_index INTEGER
                CHECK (realization_index IS NULL OR realization_index >= 0),
            dataset_key TEXT,
            membership_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (run_id, measurement_id)
        )
        """,
    ),
    (
        "model_versions",
        """
        CREATE TABLE model_versions (
            model_id INTEGER PRIMARY KEY CHECK (model_id > 0),
            model_key TEXT NOT NULL UNIQUE CHECK (length(trim(model_key)) > 0),
            estimator_name TEXT NOT NULL
                CHECK (length(trim(estimator_name)) > 0),
            family TEXT NOT NULL
                CHECK (family IN ('classical', 'ml', 'bayesian', 'baseline')),
            configuration_json TEXT NOT NULL,
            configuration_sha256 TEXT NOT NULL
                CHECK (length(configuration_sha256) = 64),
            representation_id TEXT,
            prior_policy_id TEXT,
            implementation_version TEXT,
            training_run_id INTEGER REFERENCES experiment_runs(run_id)
                ON DELETE RESTRICT,
            trained_model_artifact_id INTEGER REFERENCES artifacts(artifact_id)
                ON DELETE RESTRICT,
            representation_artifact_id INTEGER REFERENCES artifacts(artifact_id)
                ON DELETE RESTRICT,
            provenance_json TEXT NOT NULL DEFAULT '{}'
        )
        """,
    ),
    (
        "model_assumptions",
        """
        CREATE TABLE model_assumptions (
            assumption_id INTEGER PRIMARY KEY CHECK (assumption_id > 0),
            assumption_key TEXT NOT NULL UNIQUE
                CHECK (length(trim(assumption_key)) > 0),
            source_assumption_id TEXT,
            assumed_decay_model TEXT NOT NULL
                CHECK (length(trim(assumed_decay_model)) > 0),
            observation_model TEXT NOT NULL
                CHECK (length(trim(observation_model)) > 0),
            background_convention TEXT NOT NULL
                CHECK (length(trim(background_convention)) > 0),
            prepared_irf_id INTEGER REFERENCES prepared_irfs(prepared_irf_id)
                ON DELETE RESTRICT,
            context_completeness TEXT NOT NULL
                CHECK (context_completeness IN ('complete', 'historical_incomplete')),
            temporal_shift_lower_ns REAL,
            temporal_shift_upper_ns REAL,
            fixed_temporal_shift_ns REAL,
            configuration_json TEXT NOT NULL DEFAULT '{}',
            provenance_json TEXT NOT NULL DEFAULT '{}',
            CHECK (
                temporal_shift_lower_ns IS NULL
                OR temporal_shift_upper_ns IS NULL
                OR temporal_shift_lower_ns < temporal_shift_upper_ns
            ),
            CHECK (
                context_completeness <> 'complete'
                OR observation_model <> 'poisson_reconvolution'
                OR prepared_irf_id IS NOT NULL
            )
        )
        """,
    ),
    (
        "lifetime_references",
        """
        CREATE TABLE lifetime_references (
            reference_id INTEGER PRIMARY KEY CHECK (reference_id > 0),
            reference_key TEXT NOT NULL UNIQUE
                CHECK (length(trim(reference_key)) > 0),
            reference_kind TEXT NOT NULL
                CHECK (reference_kind IN
                    ('trusted_experimental', 'pseudo_true_mono')),
            reference_version TEXT NOT NULL
                CHECK (length(trim(reference_version)) > 0),
            measurement_id INTEGER REFERENCES measurements(measurement_id)
                ON DELETE RESTRICT,
            condition_pk INTEGER REFERENCES simulation_conditions(condition_pk)
                ON DELETE RESTRICT,
            assumption_id INTEGER REFERENCES model_assumptions(assumption_id)
                ON DELETE RESTRICT,
            source_artifact_id INTEGER REFERENCES artifacts(artifact_id)
                ON DELETE RESTRICT,
            supersedes_reference_id INTEGER REFERENCES lifetime_references(reference_id)
                ON DELETE RESTRICT,
            lifetime_ns REAL NOT NULL CHECK (lifetime_ns > 0),
            projection_amplitude REAL,
            projection_background_per_bin REAL,
            projection_temporal_shift_ns REAL,
            projection_poisson_nll REAL,
            validation_json TEXT NOT NULL DEFAULT '{}',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            provenance_json TEXT NOT NULL DEFAULT '{}',
            CHECK (
                (reference_kind = 'trusted_experimental'
                    AND measurement_id IS NOT NULL
                    AND condition_pk IS NULL
                    AND assumption_id IS NULL)
                OR
                (reference_kind = 'pseudo_true_mono'
                    AND measurement_id IS NULL
                    AND condition_pk IS NOT NULL
                    AND assumption_id IS NOT NULL)
            ),
            CHECK (
                supersedes_reference_id IS NULL
                OR supersedes_reference_id <> reference_id
            )
        )
        """,
    ),
    (
        "estimator_results",
        """
        CREATE TABLE estimator_results (
            result_id INTEGER PRIMARY KEY CHECK (result_id > 0),
            run_id INTEGER NOT NULL,
            measurement_id INTEGER NOT NULL,
            model_id INTEGER NOT NULL REFERENCES model_versions(model_id)
                ON DELETE RESTRICT,
            assumption_id INTEGER REFERENCES model_assumptions(assumption_id)
                ON DELETE RESTRICT,
            analysis_key TEXT NOT NULL DEFAULT 'default'
                CHECK (length(trim(analysis_key)) > 0),
            point_summary TEXT NOT NULL CHECK (length(trim(point_summary)) > 0),
            status TEXT NOT NULL
                CHECK (status IN ('available', 'failed', 'unavailable')),
            is_valid INTEGER NOT NULL CHECK (is_valid IN (0, 1)),
            irf_model_relation TEXT NOT NULL DEFAULT 'unspecified'
                CHECK (irf_model_relation IN
                    ('unspecified', 'matched', 'deliberately_misspecified')),
            source_result_type TEXT NOT NULL
                CHECK (length(trim(source_result_type)) > 0),
            lifetime_estimate_ns REAL,
            failure_reason TEXT,
            runtime_seconds REAL
                CHECK (runtime_seconds IS NULL OR runtime_seconds >= 0),
            runtime_scope TEXT,
            random_seed_decimal TEXT,
            nonfinite_fields_json TEXT NOT NULL DEFAULT '{}',
            execution_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (run_id, measurement_id)
                REFERENCES run_measurements(run_id, measurement_id)
                ON DELETE RESTRICT,
            CHECK (
                is_valid = 0 OR
                (status = 'available'
                    AND lifetime_estimate_ns IS NOT NULL
                    AND lifetime_estimate_ns > 0)
            ),
            CHECK (
                random_seed_decimal IS NULL OR (
                    length(random_seed_decimal) > 0
                    AND random_seed_decimal NOT GLOB '*[^0-9]*'
                    AND (random_seed_decimal = '0'
                        OR random_seed_decimal GLOB '[1-9]*')
                )
            )
        )
        """,
    ),
    (
        "fit_details",
        """
        CREATE TABLE fit_details (
            result_id INTEGER PRIMARY KEY REFERENCES estimator_results(result_id)
                ON DELETE RESTRICT,
            source_success INTEGER CHECK (source_success IN (0, 1)),
            optimizer_reported_success INTEGER
                CHECK (optimizer_reported_success IN (0, 1)),
            valid_fit INTEGER NOT NULL CHECK (valid_fit IN (0, 1)),
            numerical_validation_passed INTEGER
                CHECK (numerical_validation_passed IN (0, 1)),
            recovery_attempted INTEGER NOT NULL DEFAULT 0
                CHECK (recovery_attempted IN (0, 1)),
            boundary_hit INTEGER CHECK (boundary_hit IN (0, 1)),
            fitted_amplitude REAL,
            fitted_background_per_bin REAL,
            fitted_temporal_shift_ns REAL,
            initial_amplitude REAL,
            initial_lifetime_ns REAL,
            initial_background_per_bin REAL,
            initial_temporal_shift_ns REAL,
            poisson_nll REAL,
            poisson_deviance REAL,
            max_coordinate_descent_nll REAL,
            optimizer_status INTEGER,
            optimizer_message TEXT,
            optimizer_nfev INTEGER
                CHECK (optimizer_nfev IS NULL OR optimizer_nfev >= 0),
            optimizer_njev INTEGER
                CHECK (optimizer_njev IS NULL OR optimizer_njev >= 0),
            optimizer_seconds REAL
                CHECK (optimizer_seconds IS NULL OR optimizer_seconds >= 0),
            call_seconds REAL
                CHECK (call_seconds IS NULL OR call_seconds >= 0),
            exception_message TEXT,
            diagnostics_json TEXT NOT NULL DEFAULT '{}',
            nonfinite_fields_json TEXT NOT NULL DEFAULT '{}'
        )
        """,
    ),
    (
        "uncertainty_results",
        """
        CREATE TABLE uncertainty_results (
            uncertainty_id INTEGER PRIMARY KEY CHECK (uncertainty_id > 0),
            result_id INTEGER NOT NULL REFERENCES estimator_results(result_id)
                ON DELETE RESTRICT,
            calibration_run_id INTEGER REFERENCES experiment_runs(run_id)
                ON DELETE RESTRICT,
            samples_artifact_id INTEGER REFERENCES artifacts(artifact_id)
                ON DELETE RESTRICT,
            method_id TEXT NOT NULL CHECK (length(trim(method_id)) > 0),
            output_kind TEXT NOT NULL CHECK (output_kind IN
                ('prediction_interval', 'uncertainty_score',
                 'covariance_summary', 'credible_interval')),
            interval_kind TEXT,
            nominal_coverage REAL
                CHECK (nominal_coverage IS NULL
                    OR nominal_coverage > 0 AND nominal_coverage < 1),
            method_config_json TEXT NOT NULL,
            method_config_sha256 TEXT NOT NULL
                CHECK (length(method_config_sha256) = 64),
            interpretation TEXT NOT NULL CHECK (length(trim(interpretation)) > 0),
            calibration_scope TEXT NOT NULL
                CHECK (length(trim(calibration_scope)) > 0),
            is_valid INTEGER NOT NULL CHECK (is_valid IN (0, 1)),
            lower_ns REAL,
            upper_ns REAL,
            uncertainty_score REAL,
            reported_std_ns REAL,
            resample_median_ns REAL,
            n_requested INTEGER
                CHECK (n_requested IS NULL OR n_requested >= 0),
            n_valid INTEGER CHECK (n_valid IS NULL OR n_valid >= 0),
            refit_failure_rate REAL
                CHECK (refit_failure_rate IS NULL
                    OR refit_failure_rate BETWEEN 0 AND 1),
            runtime_seconds REAL
                CHECK (runtime_seconds IS NULL OR runtime_seconds >= 0),
            random_seed_decimal TEXT,
            failure_reason TEXT,
            diagnostics_json TEXT NOT NULL DEFAULT '{}',
            nonfinite_fields_json TEXT NOT NULL DEFAULT '{}',
            CHECK (
                n_requested IS NULL OR n_valid IS NULL
                OR n_valid <= n_requested
            ),
            CHECK (
                (output_kind IN ('prediction_interval', 'credible_interval')
                    AND interval_kind IS NOT NULL
                    AND nominal_coverage IS NOT NULL
                    AND uncertainty_score IS NULL
                    AND (is_valid = 0 OR
                        (lower_ns IS NOT NULL AND upper_ns IS NOT NULL
                         AND lower_ns <= upper_ns)))
                OR
                (output_kind = 'uncertainty_score'
                    AND interval_kind IS NULL
                    AND nominal_coverage IS NULL
                    AND lower_ns IS NULL AND upper_ns IS NULL
                    AND (is_valid = 0 OR
                        (uncertainty_score IS NOT NULL
                         AND uncertainty_score >= 0)))
                OR
                (output_kind = 'covariance_summary'
                    AND interval_kind IS NULL
                    AND nominal_coverage IS NULL
                    AND lower_ns IS NULL AND upper_ns IS NULL
                    AND uncertainty_score IS NULL
                    AND (is_valid = 0 OR
                        (reported_std_ns IS NOT NULL
                         AND reported_std_ns >= 0)))
            ),
            CHECK (
                random_seed_decimal IS NULL OR (
                    length(random_seed_decimal) > 0
                    AND random_seed_decimal NOT GLOB '*[^0-9]*'
                    AND (random_seed_decimal = '0'
                        OR random_seed_decimal GLOB '[1-9]*')
                )
            )
        )
        """,
    ),
    (
        "bayesian_summaries",
        """
        CREATE TABLE bayesian_summaries (
            result_id INTEGER PRIMARY KEY REFERENCES estimator_results(result_id)
                ON DELETE RESTRICT,
            posterior_artifact_id INTEGER REFERENCES artifacts(artifact_id)
                ON DELETE RESTRICT,
            predictive_artifact_id INTEGER REFERENCES artifacts(artifact_id)
                ON DELETE RESTRICT,
            sampling_status TEXT NOT NULL CHECK (sampling_status IN
                ('success', 'insufficient_sampling',
                 'initialization_failed', 'numerical_failure')),
            diagnostics_accepted INTEGER NOT NULL
                CHECK (diagnostics_accepted IN (0, 1)),
            lifetime_mean_ns REAL,
            mean_acceptance_fraction REAL,
            minimum_effective_samples REAL,
            production_steps INTEGER CHECK (production_steps IS NULL
                OR production_steps >= 0),
            retained_samples INTEGER CHECK (retained_samples IS NULL
                OR retained_samples >= 0),
            extension_count INTEGER CHECK (extension_count IS NULL
                OR extension_count >= 0),
            minimum_autocorrelation_multiples REAL,
            maximum_autocorrelation_relative_change REAL,
            maximum_ensemble_mean_difference_sd REAL,
            maximum_ensemble_median_difference_sd REAL,
            lifetime_background_correlation REAL,
            lifetime_shift_correlation REAL,
            initialization_seconds REAL,
            sampling_seconds REAL,
            diagnostic_seconds REAL,
            inference_total_seconds REAL,
            call_seconds REAL,
            diagnostic_failure_reasons_json TEXT NOT NULL DEFAULT '[]',
            parameter_summaries_json TEXT NOT NULL DEFAULT '[]',
            correlation_json TEXT NOT NULL DEFAULT '[]',
            ppc_status TEXT CHECK (ppc_status IS NULL OR ppc_status IN
                ('not_requested', 'success', 'not_available', 'failed')),
            ppc_n_draws INTEGER CHECK (ppc_n_draws IS NULL
                OR ppc_n_draws >= 0),
            ppc_runtime_seconds REAL,
            ppc_deviance_tail_probability REAL,
            ppc_discrepancies_json TEXT NOT NULL DEFAULT '{}',
            nonfinite_fields_json TEXT NOT NULL DEFAULT '{}',
            CHECK (
                diagnostics_accepted = 0 OR sampling_status = 'success'
            )
        )
        """,
    ),
    (
        "benchmark_metrics",
        """
        CREATE TABLE benchmark_metrics (
            metric_id INTEGER PRIMARY KEY CHECK (metric_id > 0),
            run_id INTEGER NOT NULL REFERENCES experiment_runs(run_id)
                ON DELETE RESTRICT,
            model_id INTEGER REFERENCES model_versions(model_id)
                ON DELETE RESTRICT,
            assumption_id INTEGER REFERENCES model_assumptions(assumption_id)
                ON DELETE RESTRICT,
            condition_pk INTEGER REFERENCES simulation_conditions(condition_pk)
                ON DELETE RESTRICT,
            reference_id INTEGER REFERENCES lifetime_references(reference_id)
                ON DELETE RESTRICT,
            source_artifact_id INTEGER REFERENCES artifacts(artifact_id)
                ON DELETE RESTRICT,
            metric_name TEXT NOT NULL CHECK (length(trim(metric_name)) > 0),
            metric_unit TEXT NOT NULL CHECK (length(trim(metric_unit)) > 0),
            scope_sha256 TEXT NOT NULL CHECK (length(scope_sha256) = 64),
            scope_json TEXT NOT NULL,
            value_status TEXT NOT NULL
                CHECK (value_status IN ('finite', 'undefined')),
            metric_value REAL,
            n_attempted INTEGER NOT NULL CHECK (n_attempted >= 0),
            n_valid INTEGER CHECK (n_valid IS NULL
                OR n_valid BETWEEN 0 AND n_attempted),
            n_contributing INTEGER CHECK (n_contributing IS NULL
                OR n_contributing BETWEEN 0 AND n_attempted),
            denominator_kind TEXT NOT NULL
                CHECK (length(trim(denominator_kind)) > 0),
            test_id TEXT,
            regime_id TEXT,
            method_id TEXT,
            nominal_coverage REAL
                CHECK (nominal_coverage IS NULL
                    OR nominal_coverage > 0 AND nominal_coverage < 1),
            reference_kind TEXT CHECK (reference_kind IS NULL OR
                reference_kind IN ('generating_mono', 'primary_component',
                    'trusted_experimental', 'pseudo_true_mono')),
            reference_version TEXT,
            signal_photon_count INTEGER
                CHECK (signal_photon_count IS NULL
                    OR signal_photon_count > 0),
            background_per_bin REAL
                CHECK (background_per_bin IS NULL
                    OR background_per_bin >= 0),
            nonfinite_fields_json TEXT NOT NULL DEFAULT '{}',
            UNIQUE (run_id, scope_sha256, metric_name),
            CHECK (
                (value_status = 'finite' AND metric_value IS NOT NULL)
                OR (value_status = 'undefined' AND metric_value IS NULL)
            )
        )
        """,
    ),
)

_INDEXES: tuple[tuple[str, str], ...] = (
    ("idx_artifacts_run_kind", """
        CREATE INDEX idx_artifacts_run_kind
        ON artifacts(producing_run_id, artifact_kind)
    """),
    ("idx_irf_sources_kind", """
        CREATE INDEX idx_irf_sources_kind ON irf_sources(source_kind)
    """),
    ("idx_irf_sources_measurement", """
        CREATE INDEX idx_irf_sources_measurement
        ON irf_sources(derived_from_measurement_id)
    """),
    ("idx_prepared_irfs_source", """
        CREATE INDEX idx_prepared_irfs_source
        ON prepared_irfs(irf_source_id)
    """),
    ("idx_conditions_photon_background", """
        CREATE INDEX idx_conditions_photon_background
        ON simulation_conditions(signal_photon_count, background_per_bin)
    """),
    ("idx_measurements_condition", """
        CREATE INDEX idx_measurements_condition ON measurements(condition_pk)
    """),
    ("idx_measurements_attached_irf", """
        CREATE INDEX idx_measurements_attached_irf
        ON measurements(attached_irf_source_id)
    """),
    ("idx_measurements_sample", """
        CREATE INDEX idx_measurements_sample ON measurements(sample_id)
    """),
    ("idx_run_measurements_role", """
        CREATE INDEX idx_run_measurements_role
        ON run_measurements(run_id, data_role, test_id, regime_id)
    """),
    ("idx_run_measurements_measurement", """
        CREATE INDEX idx_run_measurements_measurement
        ON run_measurements(measurement_id, run_id)
    """),
    ("idx_models_family_estimator", """
        CREATE INDEX idx_models_family_estimator
        ON model_versions(family, estimator_name)
    """),
    ("idx_models_training_run", """
        CREATE INDEX idx_models_training_run
        ON model_versions(training_run_id)
    """),
    ("idx_assumptions_irf", """
        CREATE INDEX idx_assumptions_irf
        ON model_assumptions(prepared_irf_id)
    """),
    ("idx_references_condition", """
        CREATE INDEX idx_references_condition
        ON lifetime_references(reference_kind, condition_pk, assumption_id)
    """),
    ("idx_references_measurement", """
        CREATE INDEX idx_references_measurement
        ON lifetime_references(measurement_id)
    """),
    ("uq_results_identity", """
        CREATE UNIQUE INDEX uq_results_identity
        ON estimator_results(
            run_id, measurement_id, model_id,
            COALESCE(assumption_id, 0), analysis_key
        )
    """),
    ("idx_results_run_model", """
        CREATE INDEX idx_results_run_model
        ON estimator_results(run_id, model_id, measurement_id)
    """),
    ("idx_results_measurement", """
        CREATE INDEX idx_results_measurement
        ON estimator_results(measurement_id, run_id)
    """),
    ("idx_results_assumption", """
        CREATE INDEX idx_results_assumption
        ON estimator_results(assumption_id)
    """),
    ("uq_uncertainty_identity", """
        CREATE UNIQUE INDEX uq_uncertainty_identity
        ON uncertainty_results(
            result_id, method_id, output_kind, method_config_sha256,
            COALESCE(nominal_coverage, -1)
        )
    """),
    ("idx_uncertainty_method", """
        CREATE INDEX idx_uncertainty_method
        ON uncertainty_results(method_id, output_kind, nominal_coverage)
    """),
    ("idx_benchmark_regime", """
        CREATE INDEX idx_benchmark_regime
        ON benchmark_metrics(run_id, test_id, regime_id, model_id)
    """),
    ("idx_benchmark_method", """
        CREATE INDEX idx_benchmark_method
        ON benchmark_metrics(run_id, method_id, nominal_coverage)
    """),
)


def _normal_sql(statement: str) -> str:
    return " ".join(statement.strip().rstrip(";").split())


def _definition_sha256() -> str:
    statements = (
        *(statement for _, statement in _TABLES),
        *(statement for _, statement in _INDEXES),
    )
    source = "\n".join(_normal_sql(statement) for statement in statements)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _configure_connection(connection: sqlite3.Connection, *, readonly: bool) -> None:
    if connection.in_transaction:
        raise ValueError("configure foreign keys before starting a transaction")
    connection.execute("PRAGMA foreign_keys = ON")
    enabled = connection.execute("PRAGMA foreign_keys").fetchone()
    if enabled is None or enabled[0] != 1:
        raise PersistenceSchemaError("SQLite foreign-key enforcement is unavailable")
    if readonly:
        connection.execute("PRAGMA query_only = ON")


def _validate_schema(connection: sqlite3.Connection) -> None:
    actual = {
        (kind, name): _normal_sql(sql)
        for kind, name, sql in connection.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
        )
    }
    for name, statement in _TABLES:
        if actual.get(("table", name)) != _normal_sql(statement):
            raise PersistenceSchemaError(f"missing or altered table: {name}")
    for name, statement in _INDEXES:
        if actual.get(("index", name)) != _normal_sql(statement):
            raise PersistenceSchemaError(f"missing or altered index: {name}")

    owned_tables = {name.casefold(): name for name, _ in _TABLES}
    owned_indexes = {name.casefold() for name, _ in _INDEXES}
    for trigger_name, table_name in connection.execute(
        "SELECT name, tbl_name FROM sqlite_master WHERE type = 'trigger'"
    ):
        if table_name.casefold() in owned_tables:
            raise PersistenceSchemaError(
                f"unexpected trigger on toolkit table: {trigger_name}"
            )
    for index_name, table_name in connection.execute(
        "SELECT name, tbl_name FROM sqlite_master "
        "WHERE type = 'index' AND sql IS NOT NULL"
    ):
        canonical_table_name = owned_tables.get(table_name.casefold())
        if canonical_table_name is None or index_name.casefold() in owned_indexes:
            continue
        # Extra non-unique query indexes are harmless. A UNIQUE index adds a
        # write constraint to a toolkit table and changes persistence semantics.
        for index in connection.execute(f'PRAGMA index_list("{canonical_table_name}")'):
            if index[1].casefold() == index_name.casefold() and index[2]:
                raise PersistenceSchemaError(
                    f"unexpected unique index on toolkit table: {index_name}"
                )

    rows = connection.execute(
        "SELECT singleton_id, schema_name, schema_version, "
        "serialization_version, definition_sha256 FROM schema_metadata"
    ).fetchall()
    if len(rows) != 1 or tuple(rows[0]) != (
        1, _SCHEMA_NAME, SCHEMA_VERSION, _SERIALIZATION_VERSION,
        _definition_sha256(),
    ):
        raise PersistenceSchemaError("schema metadata is absent or incompatible")
    version = connection.execute("PRAGMA user_version").fetchone()
    if version is None or version[0] != SCHEMA_VERSION:
        raise PersistenceSchemaError("SQLite user_version is incompatible")


def _execute_schema_statement(
    connection: sqlite3.Connection, statement: str
) -> None:
    connection.execute(statement)


def initialize_database(
    database: str | Path | sqlite3.Connection,
) -> None:
    """Atomically initialize schema state or validate an existing v1 database.

    Pass an open connection for an in-memory database. A failed initialization
    may leave an empty SQLite file when a path was supplied, but never a
    partially initialized schema.
    """
    own_connection = not isinstance(database, sqlite3.Connection)
    if own_connection:
        if str(database) == ":memory:":
            raise ValueError("pass an open sqlite3.Connection for an in-memory database")
        connection = sqlite3.connect(database, timeout=5.0, isolation_level=None)
    else:
        connection = database
    try:
        _configure_connection(connection, readonly=False)
        with transaction(connection):
            objects = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' AND type IN "
                "('table', 'view', 'index', 'trigger') LIMIT 1"
            ).fetchone()
            current_user_version = connection.execute(
                "PRAGMA user_version"
            ).fetchone()[0]
            if objects is not None:
                _validate_schema(connection)
                return
            if current_user_version != 0:
                raise PersistenceSchemaError(
                    "empty database has an unrelated user_version"
                )

            for _, statement in (*_TABLES, *_INDEXES):
                _execute_schema_statement(connection, statement)
            connection.execute(
                "INSERT INTO schema_metadata "
                "(singleton_id, schema_name, schema_version, "
                "serialization_version, definition_sha256, created_at_utc) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    1, _SCHEMA_NAME, SCHEMA_VERSION, _SERIALIZATION_VERSION,
                    _definition_sha256(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            _validate_schema(connection)
    finally:
        if own_connection:
            connection.close()


def connect_database(
    path: str | Path, *, readonly: bool = False
) -> sqlite3.Connection:
    """Open and validate an existing database without creating or upgrading it."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    mode = "ro" if readonly else "rw"
    connection = sqlite3.connect(
        f"{resolved.as_uri()}?mode={mode}",
        uri=True,
        timeout=5.0,
        isolation_level=None,
    )
    try:
        _configure_connection(connection, readonly=readonly)
        _validate_schema(connection)
        connection.row_factory = sqlite3.Row
        return connection
    except BaseException:
        connection.close()
        raise


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[None]:
    """Commit one write unit, using a savepoint inside an existing transaction."""
    if connection.in_transaction:
        name = f"tcspc_stage1_sp_{next(_SAVEPOINT_NUMBERS)}"
        connection.execute(f"SAVEPOINT {name}")
        try:
            yield
        except BaseException:
            connection.execute(f"ROLLBACK TO SAVEPOINT {name}")
            connection.execute(f"RELEASE SAVEPOINT {name}")
            raise
        else:
            connection.execute(f"RELEASE SAVEPOINT {name}")
    else:
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            connection.commit()
        except BaseException:
            connection.rollback()
            raise


def _numeric_vector(values: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values, not {array.dtype}")
    return array


def _canonical_float64_bytes(
    values: ArrayLike, *, name: str, nonnegative: bool = False,
    increasing: bool = False,
) -> bytes:
    source = _numeric_vector(values, name=name)
    try:
        canonical = np.array(source, dtype="<f8", order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} cannot be represented as float64") from exc
    if not np.all(np.isfinite(canonical)):
        raise ValueError(f"{name} must contain only finite float64 values")
    if nonnegative and np.any(canonical < 0):
        raise ValueError(f"{name} must be nonnegative")
    if increasing and (canonical.size < 2 or np.any(np.diff(canonical) <= 0)):
        raise ValueError(f"{name} must be strictly increasing")
    canonical[canonical == 0] = 0.0  # one representation for +0 and -0
    return canonical.tobytes(order="C")


def _canonical_raw_count_bytes(values: ArrayLike) -> bytes:
    source = _numeric_vector(values, name="raw counts")
    if np.any(source < 0):
        raise ValueError("raw counts must be nonnegative")
    if source.dtype.kind == "f":
        if not np.all(np.isfinite(source)):
            raise ValueError("raw counts must be finite")
        as_float = np.asarray(source, dtype=np.float64)
        if np.any(as_float >= 2**53) or np.any(as_float != np.rint(as_float)):
            raise ValueError("floating raw counts must be exact integers below 2**53")
    elif source.dtype.kind == "u" and np.any(source > np.iinfo(np.int64).max):
        raise ValueError("raw counts exceed int64 range")
    canonical = np.ascontiguousarray(source, dtype="<i8")
    return canonical.tobytes(order="C")


def _hash_time_grid_ns(values: ArrayLike) -> str:
    return hashlib.sha256(_canonical_float64_bytes(
        values, name="time grid ns", increasing=True
    )).hexdigest()


def _hash_raw_counts(values: ArrayLike) -> str:
    return hashlib.sha256(_canonical_raw_count_bytes(values)).hexdigest()


def _hash_continuous_histogram_values(values: ArrayLike) -> str:
    return hashlib.sha256(_canonical_float64_bytes(
        values, name="continuous histogram values"
    )).hexdigest()


def _hash_irf_source_values(values: ArrayLike) -> str:
    return hashlib.sha256(_canonical_float64_bytes(
        values, name="IRF source values", nonnegative=True
    )).hexdigest()


def _hash_prepared_kernel(values: ArrayLike) -> str:
    return hashlib.sha256(_canonical_float64_bytes(
        values, name="prepared IRF kernel", nonnegative=True
    )).hexdigest()


def _hash_legacy_issue4_float64_array(values: ArrayLike) -> str:
    """Retain frozen Issue-4 SHA-256 of contiguous little-endian f8 bytes."""
    source = _numeric_vector(values, name="legacy Issue-4 array")
    canonical = np.ascontiguousarray(source, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()
