"""Optional SQLite schema, connections, and scientific-record adapters.

Numerical workflows do not import or initialize this persistence boundary.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Literal

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
            byte_size INTEGER CHECK (byte_size IS NULL OR
                (typeof(byte_size) = 'integer' AND byte_size >= 0)),
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
            n_bins INTEGER NOT NULL
                CHECK (typeof(n_bins) = 'integer' AND n_bins > 1),
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
            n_bins INTEGER NOT NULL
                CHECK (typeof(n_bins) = 'integer' AND n_bins > 1),
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
                CHECK (typeof(signal_photon_count) = 'integer'
                    AND signal_photon_count > 0),
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
                    AND primary_lifetime_ns IS NOT NULL
                    AND primary_lifetime_ns > 0
                    AND secondary_lifetime_ns IS NOT NULL
                    AND secondary_lifetime_ns > 0
                    AND secondary_detected_fraction IS NOT NULL
                    AND secondary_detected_fraction > 0
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
            n_bins INTEGER NOT NULL
                CHECK (typeof(n_bins) = 'integer' AND n_bins > 1),
            time_start_ns REAL NOT NULL,
            time_step_ns REAL NOT NULL CHECK (time_step_ns > 0),
            time_grid_sha256 TEXT NOT NULL
                CHECK (length(time_grid_sha256) = 64),
            values_sha256 TEXT NOT NULL CHECK (length(values_sha256) = 64),
            observed_total_counts INTEGER CHECK (observed_total_counts IS NULL
                OR typeof(observed_total_counts) = 'integer'),
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
                CHECK (realization_index IS NULL OR
                    (typeof(realization_index) = 'integer'
                        AND realization_index >= 0)),
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
            ),
            CHECK (
                context_completeness <> 'complete'
                OR observation_model <> 'poisson_reconvolution'
                OR (
                    (fixed_temporal_shift_ns IS NOT NULL
                        AND temporal_shift_lower_ns IS NULL
                        AND temporal_shift_upper_ns IS NULL)
                    OR
                    (fixed_temporal_shift_ns IS NULL
                        AND temporal_shift_lower_ns IS NOT NULL
                        AND temporal_shift_upper_ns IS NOT NULL
                        AND temporal_shift_lower_ns < temporal_shift_upper_ns)
                )
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
                    AND typeof(lifetime_estimate_ns) IN ('integer', 'real'))
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
            optimizer_status INTEGER CHECK (optimizer_status IS NULL
                OR typeof(optimizer_status) = 'integer'),
            optimizer_message TEXT,
            optimizer_nfev INTEGER
                CHECK (optimizer_nfev IS NULL OR
                    (typeof(optimizer_nfev) = 'integer'
                        AND optimizer_nfev >= 0)),
            optimizer_njev INTEGER
                CHECK (optimizer_njev IS NULL OR
                    (typeof(optimizer_njev) = 'integer'
                        AND optimizer_njev >= 0)),
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
                CHECK (n_requested IS NULL OR
                    (typeof(n_requested) = 'integer' AND n_requested >= 0)),
            n_valid INTEGER CHECK (n_valid IS NULL OR
                (typeof(n_valid) = 'integer' AND n_valid >= 0)),
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
            production_steps INTEGER CHECK (production_steps IS NULL OR
                (typeof(production_steps) = 'integer' AND production_steps >= 0)),
            retained_samples INTEGER CHECK (retained_samples IS NULL OR
                (typeof(retained_samples) = 'integer' AND retained_samples >= 0)),
            extension_count INTEGER CHECK (extension_count IS NULL OR
                (typeof(extension_count) = 'integer' AND extension_count >= 0)),
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
            ppc_n_draws INTEGER CHECK (ppc_n_draws IS NULL OR
                (typeof(ppc_n_draws) = 'integer' AND ppc_n_draws >= 0)),
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
            n_attempted INTEGER NOT NULL
                CHECK (typeof(n_attempted) = 'integer' AND n_attempted >= 0),
            n_valid INTEGER CHECK (n_valid IS NULL OR
                (typeof(n_valid) = 'integer'
                    AND n_valid BETWEEN 0 AND n_attempted)),
            n_contributing INTEGER CHECK (n_contributing IS NULL OR
                (typeof(n_contributing) = 'integer'
                    AND n_contributing BETWEEN 0 AND n_attempted)),
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
                CHECK (signal_photon_count IS NULL OR
                    (typeof(signal_photon_count) = 'integer'
                        AND signal_photon_count > 0)),
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


# Stage-2 adapters. The numerical modules never import this optional boundary.
DuplicatePolicy = Literal["raise", "reuse_identical"]


class PersistenceConflictError(ValueError):
    """A stable key already identifies a different scientific record."""


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _optional_text(value: str | None, name: str) -> str | None:
    return None if value is None else _required_text(value, name)


def _finite_scalar(
    value: Any, name: str, *, positive: bool = False, nonnegative: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0 or nonnegative and result < 0:
        raise ValueError(f"{name} is outside its allowed range")
    return result


def _optional_scalar(value: Any, name: str, **kwargs: bool) -> float | None:
    return None if value is None else _finite_scalar(value, name, **kwargs)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or result > 2**63 - 1:
        raise ValueError(f"{name} must be between {minimum} and SQLite int64 maximum")
    return result


def _optional_integer(value: Any, name: str, *, minimum: int = 0) -> int | None:
    return None if value is None else _integer(value, name, minimum=minimum)


def _decimal_seed(value: int | str | None, name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer or canonical decimal text")
    if isinstance(value, (int, np.integer)):
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
        return str(int(value))
    if not isinstance(value, str) or not value or not value.isascii() or not value.isdecimal():
        raise ValueError(f"{name} must be canonical decimal text")
    if len(value) > 1 and value[0] == "0":
        raise ValueError(f"{name} must not have leading zeros")
    return value


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
    return normalized


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, np.generic):
        converted = value.item()
        if isinstance(converted, np.generic):
            raise TypeError(f"unsupported NumPy scalar: {value.dtype}")
        return _json_value(converted)
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON mapping keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON values must be finite")
        return value
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def _mapping_json(value: Mapping[str, Any] | None, name: str) -> str:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return _canonical_json(value)


def _configuration_json(value: Any) -> tuple[str, str]:
    if not isinstance(value, Mapping) and not (is_dataclass(value) and not isinstance(value, type)):
        raise TypeError("configuration must be a mapping or dataclass instance")
    serialized = _canonical_json(value)
    return serialized, hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _utc_timestamp(value: datetime | str) -> str:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("recorded_at_utc must be an ISO timestamp") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("recorded_at_utc must include a timezone")
    return value.astimezone(timezone.utc).isoformat()


def _same_payload_value(column: str, left: Any, right: Any) -> bool:
    if column.endswith("_json"):
        try:
            return _canonical_json(json.loads(left)) == _canonical_json(json.loads(right))
        except (TypeError, ValueError):
            return False
    return left == right


def _record_row(
    connection: sqlite3.Connection, *, table: str, key_columns: tuple[str, ...],
    payload: dict[str, Any], return_column: str | None,
    on_duplicate: DuplicatePolicy,
) -> int | tuple[int, ...]:
    if on_duplicate not in ("raise", "reuse_identical"):
        raise ValueError("on_duplicate must be 'raise' or 'reuse_identical'")
    if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise ValueError("foreign keys must be enabled before recording")
    # Identifiers are private literals supplied by the adapters; values are bound.
    columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
    expected = set(columns) - ({return_column} if return_column else set())
    if set(payload) != expected:
        raise RuntimeError(f"incomplete {table} adapter payload")
    where = " AND ".join(f"{column} = ?" for column in key_columns)
    with transaction(connection):
        cursor = connection.execute(
            f"SELECT * FROM {table} WHERE {where}",
            tuple(payload[column] for column in key_columns),
        )
        found = cursor.fetchone()
        if found is not None:
            existing = dict(zip((item[0] for item in cursor.description), found))
            if on_duplicate == "raise":
                raise PersistenceConflictError(f"duplicate {table} stable key")
            if not all(_same_payload_value(column, existing[column], value)
                       for column, value in payload.items()):
                raise PersistenceConflictError(f"conflicting {table} stable key")
            return (int(existing[return_column]) if return_column else
                    tuple(int(existing[column]) for column in key_columns))
        names = tuple(payload)
        placeholders = ", ".join("?" for _ in names)
        inserted = connection.execute(
            f"INSERT INTO {table} ({', '.join(names)}) VALUES ({placeholders})",
            tuple(payload.values()),
        )
        return (int(inserted.lastrowid) if return_column else
                tuple(int(payload[column]) for column in key_columns))


def _require_row(connection: sqlite3.Connection, table: str, id_column: str, row_id: int) -> sqlite3.Row | tuple:
    row = connection.execute(
        f"SELECT * FROM {table} WHERE {id_column} = ?", (row_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown {table}.{id_column}: {row_id}")
    return row


def _uniform_grid(time_ns: ArrayLike) -> tuple[np.ndarray, int, float, float, str]:
    from tcspc_toolkit.preprocessing import validate_time_axis

    time = validate_time_axis(time_ns)
    return time, int(time.size), float(time[0]), float(np.diff(time)[0]), _hash_time_grid_ns(time)


def record_run(
    connection: sqlite3.Connection, *, run_key: str, run_type: str,
    status: Literal["complete", "partial", "failed"],
    recorded_at_utc: datetime | str, origin: Literal["new", "historical"],
    configuration: Mapping[str, Any] | Any,
    package_version: str | None = None, code_revision: str | None = None,
    git_commit: str | None = None, version_state: Literal["known", "unknown"] = "known",
    version_unknown_reason: str | None = None, editable_source: bool = False,
    protocol_id: str | None = None, protocol_version: str | None = None,
    profile: str | None = None, base_seed: int | str | None = None,
    seed_namespace: str | None = None,
    dependency_versions: Mapping[str, Any] | None = None,
    seed_policy: Mapping[str, Any] | None = None,
    notes: str | None = None, source_run_id: int | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Record caller-supplied producing evidence without runtime version guessing."""
    if status not in ("complete", "partial", "failed") or origin not in ("new", "historical"):
        raise ValueError("invalid run status or origin")
    if version_state == "known":
        package_version = _required_text(package_version, "package_version")
        if version_unknown_reason is not None:
            raise ValueError("known producing version cannot have an unknown reason")
        if editable_source and code_revision is None and git_commit is None:
            raise ValueError("editable source requires an actual Git or code revision")
    elif version_state == "unknown":
        if origin != "historical" or any(
            item is not None for item in (package_version, code_revision, git_commit)
        ):
            raise ValueError("only historical records may have wholly unknown producer evidence")
        _required_text(version_unknown_reason, "version_unknown_reason")
        if editable_source:
            raise ValueError("unknown historical producer cannot assert editable source")
    else:
        raise ValueError("invalid version_state")
    config_json, config_sha256 = _configuration_json(configuration)
    payload = {
        "run_key": _required_text(run_key, "run_key"),
        "run_type": _required_text(run_type, "run_type"),
        "status": status,
        "recorded_at_utc": _utc_timestamp(recorded_at_utc),
        "origin": origin,
        "version_state": version_state,
        "producing_package_version": package_version,
        "producing_code_revision": _optional_text(code_revision, "code_revision"),
        "producing_git_commit": _optional_text(git_commit, "git_commit"),
        "producing_version_unknown_reason": version_unknown_reason,
        "config_json": config_json,
        "config_sha256": config_sha256,
        "protocol_id": _optional_text(protocol_id, "protocol_id"),
        "protocol_version": _optional_text(protocol_version, "protocol_version"),
        "profile": _optional_text(profile, "profile"),
        "base_seed_decimal": _decimal_seed(base_seed, "base_seed"),
        "seed_namespace": _optional_text(seed_namespace, "seed_namespace"),
        "dependency_versions_json": _mapping_json(dependency_versions, "dependency_versions"),
        "seed_policy_json": _mapping_json(seed_policy, "seed_policy"),
        "notes": notes,
        "source_run_id": _optional_integer(source_run_id, "source_run_id", minimum=1),
    }
    return _record_row(
        connection, table="experiment_runs", key_columns=("run_key",),
        payload=payload, return_column="run_id", on_duplicate=on_duplicate,
    )


def _database_directory(connection: sqlite3.Connection) -> Path | None:
    for _, name, path in connection.execute("PRAGMA database_list"):
        if name == "main" and path:
            return Path(path).resolve().parent
    return None


def _artifact_file_path(
    connection: sqlite3.Connection, path: str | Path, path_base: str,
) -> Path:
    chosen = Path(path)
    if path_base == "absolute":
        if not chosen.is_absolute():
            raise ValueError("absolute artifact path must be absolute")
        return chosen
    if path_base != "database_directory" or chosen.is_absolute():
        raise ValueError("database_directory artifact path must be relative")
    directory = _database_directory(connection)
    if directory is None:
        raise ValueError("in-memory databases require absolute artifact paths")
    return directory / chosen


def _hash_file_bytes(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def record_artifact(
    connection: sqlite3.Connection, *, artifact_key: str, path: str | Path,
    path_base: Literal["database_directory", "absolute"], artifact_kind: str,
    format: str, sha256: str | None = None, byte_size: int | None = None,
    producing_run_id: int | None = None,
    locator: Mapping[str, Any] | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Register an external file; compute its byte hash only when omitted."""
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("path must be nonempty")
    file_path = _artifact_file_path(connection, path, path_base)
    if sha256 is None:
        sha256, actual_size = _hash_file_bytes(file_path)
        if byte_size is not None and byte_size != actual_size:
            raise ValueError("byte_size conflicts with the external file")
        byte_size = actual_size
    payload = {
        "artifact_key": _required_text(artifact_key, "artifact_key"),
        "producing_run_id": _optional_integer(producing_run_id, "producing_run_id", minimum=1),
        "path": str(path), "path_base": path_base,
        "artifact_kind": _required_text(artifact_kind, "artifact_kind"),
        "format": _required_text(format, "format"),
        "sha256": _sha256(sha256, "sha256"),
        "byte_size": _optional_integer(byte_size, "byte_size"),
        "locator_json": _mapping_json(locator, "locator"),
    }
    return _record_row(
        connection, table="artifacts", key_columns=("artifact_key",),
        payload=payload, return_column="artifact_id", on_duplicate=on_duplicate,
    )


def record_irf_source(
    connection: sqlite3.Connection, source: Any, *, source_key: str,
    source_artifact_id: int | None = None,
    derived_from_measurement_id: int | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Record a SampledIRF, IRFProfile or successful leading-edge result."""
    from tcspc_toolkit.irf import IRFProfile, IRFSourceKind
    from tcspc_toolkit.irf_estimation import LeadingEdgeIRFResult
    from tcspc_toolkit.measurements import SampledIRF

    supplied_source = source
    estimate_diagnostics = None
    if isinstance(source, LeadingEdgeIRFResult):
        if source.profile is None:
            raise ValueError("failed leading-edge estimation has no IRF source")
        estimate_diagnostics = source.diagnostics
        source = source.profile
    if isinstance(source, SampledIRF):
        representation, kind = "sampled_irf", IRFSourceKind.IMPORTED_SAMPLED.value
        parameters: Mapping[str, Any] = {}
    elif isinstance(source, IRFProfile):
        representation, kind = "profile", source.source_kind.value
        parameters = source.source_parameters
    else:
        raise TypeError("source must be SampledIRF, IRFProfile or LeadingEdgeIRFResult")
    if kind == IRFSourceKind.LEADING_EDGE_ESTIMATE.value and derived_from_measurement_id is None:
        raise ValueError("leading-edge proxy requires its source measurement ID")
    provenance = dict(source.provenance)
    if estimate_diagnostics is not None:
        provenance["leading_edge_diagnostics"] = estimate_diagnostics
    payload = {
        "source_key": _required_text(source_key, "source_key"),
        "source_representation": representation,
        "source_kind": kind,
        "n_bins": _integer(len(source.time_ns), "n_bins", minimum=2),
        "source_grid_sha256": _hash_time_grid_ns(source.time_ns),
        "source_values_sha256": _hash_irf_source_values(source.values),
        "source_artifact_id": _optional_integer(source_artifact_id, "source_artifact_id", minimum=1),
        "derived_from_measurement_id": _optional_integer(
            derived_from_measurement_id, "derived_from_measurement_id", minimum=1,
        ),
        "source_parameters_json": _mapping_json(parameters, "source_parameters"),
        "metadata_json": _mapping_json(source.metadata, "metadata"),
        "provenance_json": _mapping_json(provenance, "provenance"),
    }
    if on_duplicate == "reuse_identical":
        if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            raise ValueError("foreign keys must be enabled before recording")
        with transaction(connection):
            existing = connection.execute(
                "SELECT irf_source_id, source_representation FROM irf_sources "
                "WHERE source_key = ?", (payload["source_key"],),
            ).fetchone()
            if existing is not None:
                source_id = int(existing[0])
                try:
                    _verify_irf_source(connection, source_id, supplied_source)
                except ValueError as exc:
                    raise PersistenceConflictError("conflicting irf_sources stable key") from exc
                cursor = connection.execute(
                    "SELECT * FROM irf_sources WHERE irf_source_id = ?", (source_id,),
                )
                row = dict(zip((item[0] for item in cursor.description), cursor.fetchone()))
                if not all(
                    _same_payload_value(column, row[column], value)
                    for column, value in payload.items()
                    if column != "source_representation"
                ):
                    raise PersistenceConflictError("conflicting irf_sources stable key")
                return source_id
            return _record_row(
                connection, table="irf_sources", key_columns=("source_key",),
                payload=payload, return_column="irf_source_id", on_duplicate=on_duplicate,
            )
    return _record_row(
        connection, table="irf_sources", key_columns=("source_key",),
        payload=payload, return_column="irf_source_id", on_duplicate=on_duplicate,
    )


def record_legacy_irf_source(
    connection: sqlite3.Connection, *, source_key: str,
    time_ns: ArrayLike, values: ArrayLike,
    source_artifact_id: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Label a bare historical array without assigning a measured source kind."""
    time = _numeric_vector(time_ns, name="legacy IRF time")
    samples = _numeric_vector(values, name="legacy IRF values")
    if time.shape != samples.shape or time.size < 2:
        raise ValueError("legacy IRF arrays must have matching lengths above one")
    payload = {
        "source_key": _required_text(source_key, "source_key"),
        "source_representation": "bare_array", "source_kind": None,
        "n_bins": int(time.size),
        "source_grid_sha256": _hash_time_grid_ns(time),
        "source_values_sha256": _hash_irf_source_values(samples),
        "source_artifact_id": _optional_integer(source_artifact_id, "source_artifact_id", minimum=1),
        "derived_from_measurement_id": None,
        "source_parameters_json": "{}",
        "metadata_json": _mapping_json(metadata, "metadata"),
        "provenance_json": _mapping_json(provenance, "provenance"),
    }
    return _record_row(
        connection, table="irf_sources", key_columns=("source_key",),
        payload=payload, return_column="irf_source_id", on_duplicate=on_duplicate,
    )


def _verify_irf_source(
    connection: sqlite3.Connection, source_id: int, source: Any,
) -> None:
    from tcspc_toolkit.irf import IRFProfile, IRFSourceKind
    from tcspc_toolkit.irf_estimation import LeadingEdgeIRFResult
    from tcspc_toolkit.measurements import SampledIRF

    diagnostics = None
    if isinstance(source, LeadingEdgeIRFResult):
        if source.profile is None:
            raise ValueError("failed leading-edge estimation has no IRF source")
        diagnostics = source.diagnostics
        source = source.profile
    if not isinstance(source, (IRFProfile, SampledIRF)):
        raise TypeError("source must be an IRFProfile or SampledIRF")
    cursor = connection.execute(
        "SELECT source_representation, source_kind, n_bins, source_grid_sha256, "
        "source_values_sha256, source_parameters_json, metadata_json, provenance_json "
        "FROM irf_sources WHERE irf_source_id = ?", (source_id,),
    )
    found = cursor.fetchone()
    if found is None:
        raise ValueError("IRF source ID does not identify the supplied source")
    row = dict(zip((item[0] for item in cursor.description), found))
    kind = source.source_kind.value if isinstance(source, IRFProfile) else "imported_sampled"
    representations = (
        {"sampled_irf", "profile"}
        if kind == IRFSourceKind.IMPORTED_SAMPLED.value else {"profile"}
    )
    parameters = source.source_parameters if isinstance(source, IRFProfile) else {}
    provenance = dict(source.provenance)
    stored_provenance = json.loads(row["provenance_json"])
    if (
        diagnostics is None and kind == IRFSourceKind.LEADING_EDGE_ESTIMATE.value
        and "leading_edge_diagnostics" not in provenance
    ):
        # A PreparedIRF carries its source profile but not the estimation result.
        stored_provenance.pop("leading_edge_diagnostics", None)
    elif diagnostics is not None:
        provenance["leading_edge_diagnostics"] = diagnostics
    if (
        row["source_representation"] not in representations
        or row["source_kind"] != kind
        or row["n_bins"] != len(source.time_ns)
        or row["source_grid_sha256"] != _hash_time_grid_ns(source.time_ns)
        or row["source_values_sha256"] != _hash_irf_source_values(source.values)
        or not _same_payload_value(
            "source_parameters_json", row["source_parameters_json"], _mapping_json(parameters, "source_parameters"),
        )
        or not _same_payload_value(
            "metadata_json", row["metadata_json"], _mapping_json(source.metadata, "metadata"),
        )
        or _canonical_json(stored_provenance) != _canonical_json(provenance)
    ):
        raise ValueError("IRF source ID does not identify the supplied source")


def _verify_prepared_irf(
    connection: sqlite3.Connection, prepared_id: int, prepared: Any,
) -> None:
    from tcspc_toolkit.irf_preparation import PreparedIRF

    if not isinstance(prepared, PreparedIRF):
        raise TypeError("prepared must be a PreparedIRF")
    cursor = connection.execute(
        "SELECT * FROM prepared_irfs WHERE prepared_irf_id = ?", (prepared_id,),
    )
    found = cursor.fetchone()
    if found is None:
        raise ValueError("prepared IRF ID does not identify the supplied prepared IRF")
    row = dict(zip((item[0] for item in cursor.description), found))
    if row["preparation_kind"] not in ("explicit", "legacy_same_grid_normalization") or row["irf_source_id"] is None:
        raise ValueError("prepared IRF ID does not identify the supplied prepared IRF")
    try:
        _verify_irf_source(connection, row["irf_source_id"], prepared.source)
    except ValueError as exc:
        raise ValueError("prepared IRF ID does not identify the supplied prepared IRF") from exc
    expected = _prepared_payload(
        preparation_key=row["preparation_key"], source_id=row["irf_source_id"],
        time=prepared.time_ns, kernel=prepared.kernel, kind=row["preparation_kind"],
        diagnostics=prepared.diagnostics, kernel_artifact_id=row["kernel_artifact_id"],
        preparation_provenance=None,
    )
    actual_diagnostics = json.loads(row["diagnostics_json"])
    if isinstance(actual_diagnostics, dict):
        # Supplemental caller provenance is not present on PreparedIRF itself.
        actual_diagnostics.pop("preparation_provenance", None)
    if (
        _canonical_json(actual_diagnostics) != expected["diagnostics_json"]
        or any(
            not _same_payload_value(column, row[column], value)
            for column, value in expected.items()
            if column != "diagnostics_json"
        )
    ):
        raise ValueError("prepared IRF ID does not identify the supplied prepared IRF")


def _prepared_payload(
    *, preparation_key: str, source_id: int | None, time: np.ndarray,
    kernel: ArrayLike, kind: str, diagnostics: Any,
    kernel_artifact_id: int | None, preparation_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _, n_bins, time_start, time_step, grid_sha = _uniform_grid(time)
    if np.asarray(kernel).shape != (n_bins,):
        raise ValueError("prepared kernel must match its target time grid")
    kernel_sha = _hash_prepared_kernel(kernel)
    if diagnostics is None:
        diagnostic_values: dict[str, Any] = {}
        operations: tuple[str, ...] = ()
        flags = ("source_provenance_incomplete",) if source_id is None else ()
    else:
        diagnostic_values = _json_value(diagnostics)
        operations = diagnostics.operations
        flags = diagnostics.flags
    if preparation_provenance is not None:
        diagnostic_values["preparation_provenance"] = _json_value(preparation_provenance)
    return {
        "preparation_key": _required_text(preparation_key, "preparation_key"),
        "irf_source_id": source_id,
        "preparation_kind": kind,
        "time_grid_sha256": grid_sha,
        "kernel_sha256": kernel_sha,
        "kernel_artifact_id": _optional_integer(kernel_artifact_id, "kernel_artifact_id", minimum=1),
        "n_bins": n_bins,
        "time_start_ns": time_start,
        "time_step_ns": time_step,
        "registration_offset_ns": (
            None if diagnostics is None else
            _finite_scalar(diagnostics.registration_offset_ns, "registration_offset_ns")
        ),
        "resampling_method": None if diagnostics is None else diagnostics.resampling_method,
        "support_loss_fraction": (
            None if diagnostics is None else
            _finite_scalar(diagnostics.support_loss_fraction, "support_loss_fraction", nonnegative=True)
        ),
        "normalization_factor": (
            None if diagnostics is None else
            _finite_scalar(diagnostics.normalization_factor, "normalization_factor", positive=True)
        ),
        "source_fwhm_ns": None if diagnostics is None else _optional_scalar(
            diagnostics.source_fwhm_ns, "source_fwhm_ns", positive=True,
        ),
        "target_fwhm_ns": None if diagnostics is None else _optional_scalar(
            diagnostics.target_fwhm_ns, "target_fwhm_ns", positive=True,
        ),
        "target_bins_per_fwhm": None if diagnostics is None else _optional_scalar(
            diagnostics.target_bins_per_fwhm, "target_bins_per_fwhm", positive=True,
        ),
        "diagnostics_json": _canonical_json(diagnostic_values),
        "operations_json": _canonical_json(operations),
        "flags_json": _canonical_json(flags),
    }


def record_prepared_irf(
    connection: sqlite3.Connection, prepared: Any, *, preparation_key: str,
    irf_source_id: int | None = None, source_key: str | None = None,
    source_artifact_id: int | None = None,
    kernel_artifact_id: int | None = None,
    preparation_kind: Literal["explicit", "legacy_same_grid_normalization"] = "explicit",
    preparation_provenance: Mapping[str, Any] | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Record a PreparedIRF and, when keyed, its source in one transaction."""
    from tcspc_toolkit.irf_preparation import PreparedIRF

    if not isinstance(prepared, PreparedIRF):
        raise TypeError("prepared must be a PreparedIRF")
    if (irf_source_id is None) == (source_key is None):
        raise ValueError("provide exactly one of irf_source_id or source_key")
    if preparation_kind not in ("explicit", "legacy_same_grid_normalization"):
        raise ValueError("invalid preparation_kind for PreparedIRF")
    with transaction(connection):
        source_id = (
            record_irf_source(
                connection, prepared.source, source_key=source_key,
                source_artifact_id=source_artifact_id, on_duplicate=on_duplicate,
            ) if source_key is not None else
            _integer(irf_source_id, "irf_source_id", minimum=1)
        )
        _verify_irf_source(connection, source_id, prepared.source)
        payload = _prepared_payload(
            preparation_key=preparation_key, source_id=source_id,
            time=prepared.time_ns, kernel=prepared.kernel, kind=preparation_kind,
            diagnostics=prepared.diagnostics, kernel_artifact_id=kernel_artifact_id,
            preparation_provenance=preparation_provenance,
        )
        return _record_row(
            connection, table="prepared_irfs", key_columns=("preparation_key",),
            payload=payload, return_column="prepared_irf_id", on_duplicate=on_duplicate,
        )


def record_supplied_kernel(
    connection: sqlite3.Connection, *, preparation_key: str,
    time_ns: ArrayLike, kernel: ArrayLike, irf_source_id: int | None = None,
    kernel_artifact_id: int | None = None,
    provenance: Mapping[str, Any] | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Record a unit-area historical kernel with explicitly incomplete source evidence."""
    time, _, _, _, _ = _uniform_grid(time_ns)
    values = np.asarray(kernel, dtype=np.float64)
    if values.shape != time.shape:
        raise ValueError("supplied kernel must match its grid")
    _hash_prepared_kernel(values)
    area = float(np.trapezoid(values, x=time))
    if not math.isfinite(area) or not np.isclose(area, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("supplied kernel must have unit trapezoidal area")
    source_id = _optional_integer(irf_source_id, "irf_source_id", minimum=1)
    payload = _prepared_payload(
        preparation_key=preparation_key, source_id=source_id,
        time=time, kernel=values, kind="supplied_kernel", diagnostics=None,
        kernel_artifact_id=kernel_artifact_id, preparation_provenance=provenance,
    )
    return _record_row(
        connection, table="prepared_irfs", key_columns=("preparation_key",),
        payload=payload, return_column="prepared_irf_id", on_duplicate=on_duplicate,
    )


def record_simulation_condition(
    connection: sqlite3.Connection, *, condition_key: str, condition_id: str,
    generating_model: Literal["monoexponential", "biexponential", "other"],
    signal_photon_count: int, background_per_bin: float,
    true_temporal_shift_ns: float,
    generating_irf_id: int | None = None, origin_run_id: int | None = None,
    mono_lifetime_ns: float | None = None,
    primary_lifetime_ns: float | None = None,
    secondary_lifetime_ns: float | None = None,
    secondary_detected_fraction: float | None = None,
    true_reconvolution_amplitude: float | None = None,
    expected_counts_sha256: str | None = None,
    expected_counts_artifact_id: int | None = None,
    generating_parameters: Mapping[str, Any] | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Record generating physics supplied explicitly, without inspecting observation metadata."""
    if generating_model not in ("monoexponential", "biexponential", "other"):
        raise ValueError("unknown generating_model")
    mono = _optional_scalar(mono_lifetime_ns, "mono_lifetime_ns", positive=True)
    primary = _optional_scalar(primary_lifetime_ns, "primary_lifetime_ns", positive=True)
    secondary = _optional_scalar(secondary_lifetime_ns, "secondary_lifetime_ns", positive=True)
    fraction = _optional_scalar(secondary_detected_fraction, "secondary_detected_fraction")
    if generating_model == "monoexponential" and (
        mono is None or any(item is not None for item in (primary, secondary, fraction))
    ):
        raise ValueError("mono-exponential truth requires only mono_lifetime_ns")
    if generating_model == "biexponential" and (
        mono is not None or primary is None or secondary is None
        or fraction is None or not 0 < fraction < 1
    ):
        raise ValueError("bi-exponential truth requires two lifetimes and a detected fraction")
    if generating_model == "other" and mono is not None:
        raise ValueError("other generating model must not claim mono-exponential truth")
    payload = {
        "condition_key": _required_text(condition_key, "condition_key"),
        "condition_id": _required_text(condition_id, "condition_id"),
        "origin_run_id": _optional_integer(origin_run_id, "origin_run_id", minimum=1),
        "generating_model": generating_model,
        "generating_irf_id": _optional_integer(generating_irf_id, "generating_irf_id", minimum=1),
        "mono_lifetime_ns": mono,
        "primary_lifetime_ns": primary,
        "secondary_lifetime_ns": secondary,
        "secondary_detected_fraction": fraction,
        "signal_photon_count": _integer(signal_photon_count, "signal_photon_count", minimum=1),
        "background_per_bin": _finite_scalar(background_per_bin, "background_per_bin", nonnegative=True),
        "true_temporal_shift_ns": _finite_scalar(true_temporal_shift_ns, "true_temporal_shift_ns"),
        "true_reconvolution_amplitude": _optional_scalar(
            true_reconvolution_amplitude, "true_reconvolution_amplitude", nonnegative=True,
        ),
        "expected_counts_sha256": None if expected_counts_sha256 is None else _sha256(
            expected_counts_sha256, "expected_counts_sha256",
        ),
        "expected_counts_artifact_id": _optional_integer(
            expected_counts_artifact_id, "expected_counts_artifact_id", minimum=1,
        ),
        "generating_parameters_json": _mapping_json(generating_parameters, "generating_parameters"),
    }
    return _record_row(
        connection, table="simulation_conditions", key_columns=("condition_key",),
        payload=payload, return_column="condition_pk", on_duplicate=on_duplicate,
    )


def record_issue4_condition(
    connection: sqlite3.Connection, condition: Any, *, condition_key: str,
    generating_irf_id: int, origin_run_id: int | None = None,
    true_reconvolution_amplitude: float | None = None,
    generating_parameters: Mapping[str, Any] | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Adapt matched-model or mismatch Issue-4 generating-condition objects."""
    from tcspc_toolkit.bayesian_evaluation import BayesianClassicalCondition
    from tcspc_toolkit.bayesian_mismatch_evaluation import MismatchCondition

    if not isinstance(condition, (BayesianClassicalCondition, MismatchCondition)):
        raise TypeError("condition must be an Issue-4 generating condition")
    generating_id = _integer(generating_irf_id, "generating_irf_id", minimum=1)
    try:
        _verify_prepared_irf(connection, generating_id, condition.generating_irf)
    except ValueError as exc:
        raise ValueError("generating_irf_id does not identify the condition's generating IRF") from exc
    extras = {} if generating_parameters is None else dict(generating_parameters)
    if isinstance(condition, BayesianClassicalCondition):
        model = "monoexponential"
        mono, primary, secondary, fraction = condition.true_lifetime_ns, None, None, None
        extras["n_repeats"] = condition.n_repeats
    else:
        model = condition.generating_model
        mono = condition.mono_generating_lifetime_ns
        primary = condition.primary_lifetime_ns if model == "biexponential" else None
        secondary = condition.secondary_lifetime_ns
        fraction = condition.secondary_detected_fraction
        extras.update({
            "mechanism": condition.mechanism,
            "generating_irf_label": condition.generating_irf_id,
        })
    return record_simulation_condition(
        connection, condition_key=condition_key, condition_id=condition.condition_id,
        generating_model=model, signal_photon_count=condition.signal_photon_count,
        background_per_bin=condition.background_per_bin,
        true_temporal_shift_ns=condition.true_temporal_shift_ns,
        generating_irf_id=generating_id, origin_run_id=origin_run_id,
        mono_lifetime_ns=mono, primary_lifetime_ns=primary,
        secondary_lifetime_ns=secondary, secondary_detected_fraction=fraction,
        true_reconvolution_amplitude=true_reconvolution_amplitude,
        generating_parameters=extras, on_duplicate=on_duplicate,
    )


def _record_observation(
    connection: sqlite3.Connection, *, measurement_key: str,
    time_ns: ArrayLike, values: ArrayLike,
    data_kind: Literal["raw_counts", "processed_intensity"],
    source_type: Literal["synthetic", "experimental", "unknown"],
    condition_pk: int | None, origin_run_id: int | None,
    attached_irf_source_id: int | None, histogram_artifact_id: int | None,
    sample_id: str | None, observation_seed: int | str | None,
    metadata: Mapping[str, Any] | None, provenance: Mapping[str, Any] | None,
    on_duplicate: DuplicatePolicy,
) -> int:
    from tcspc_toolkit.preprocessing import validate_histogram

    time, n_bins, time_start, time_step, grid_sha = _uniform_grid(time_ns)
    if data_kind == "raw_counts":
        # The existing scientific guard checks raw Poisson-count semantics.
        validate_histogram(time=time, counts=values)
        raw_dtype = np.asarray(values).dtype
        if raw_dtype.kind == "f" and raw_dtype.itemsize > 8:
            raise ValueError("raw counts with wider-than-float64 dtype need explicit conversion")
        count_bytes = _canonical_raw_count_bytes(values)
        counts = np.frombuffer(count_bytes, dtype="<i8")
        observed_total = int(sum(map(int, counts)))
        if observed_total > 2**63 - 1:
            raise ValueError("observed total exceeds SQLite int64 range")
        values_sha = hashlib.sha256(count_bytes).hexdigest()
    elif data_kind == "processed_intensity":
        processed = np.asarray(values, dtype=np.float64)
        if processed.shape != time.shape:
            raise ValueError("processed values must match the time grid")
        values_sha = _hash_continuous_histogram_values(processed)
        observed_total = None
    else:
        raise ValueError("data_kind must be raw_counts or processed_intensity")
    if source_type not in ("synthetic", "experimental", "unknown"):
        raise ValueError("invalid source_type")
    if source_type == "experimental" and condition_pk is not None:
        raise ValueError("experimental measurements cannot have generating truth")
    payload = {
        "measurement_key": _required_text(measurement_key, "measurement_key"),
        "origin_run_id": _optional_integer(origin_run_id, "origin_run_id", minimum=1),
        "condition_pk": _optional_integer(condition_pk, "condition_pk", minimum=1),
        "attached_irf_source_id": _optional_integer(
            attached_irf_source_id, "attached_irf_source_id", minimum=1,
        ),
        "histogram_artifact_id": _optional_integer(
            histogram_artifact_id, "histogram_artifact_id", minimum=1,
        ),
        "source_type": source_type,
        "data_kind": data_kind,
        "sample_id": _optional_text(sample_id, "sample_id"),
        "n_bins": n_bins,
        "time_start_ns": time_start,
        "time_step_ns": time_step,
        "time_grid_sha256": grid_sha,
        "values_sha256": values_sha,
        "observed_total_counts": observed_total,
        "observation_seed_decimal": _decimal_seed(observation_seed, "observation_seed"),
        "metadata_json": _mapping_json(metadata, "metadata"),
        "provenance_json": _mapping_json(provenance, "provenance"),
    }
    return _record_row(
        connection, table="measurements", key_columns=("measurement_key",),
        payload=payload, return_column="measurement_id", on_duplicate=on_duplicate,
    )


def record_measurement(
    connection: sqlite3.Connection, measurement: Any, *, measurement_key: str,
    source_type: Literal["experimental", "synthetic"] = "experimental",
    condition_pk: int | None = None, origin_run_id: int | None = None,
    attached_irf_source_id: int | None = None,
    attached_irf_source_key: str | None = None,
    histogram_artifact_id: int | None = None,
    observation_seed: int | str | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Record a TCSPCMeasurement with its explicit raw or processed meaning."""
    from tcspc_toolkit.measurements import MeasurementDataKind, TCSPCMeasurement

    if not isinstance(measurement, TCSPCMeasurement):
        raise TypeError("measurement must be a TCSPCMeasurement")
    if measurement.irf is None and (
        attached_irf_source_id is not None or attached_irf_source_key is not None
    ):
        raise ValueError("measurement has no attached IRF to identify")
    if measurement.irf is not None and (
        (attached_irf_source_id is None) == (attached_irf_source_key is None)
    ):
        raise ValueError("attached IRF requires exactly one source ID or source key")
    with transaction(connection):
        if attached_irf_source_key is not None:
            source_id = record_irf_source(
                connection, measurement.irf, source_key=attached_irf_source_key,
                on_duplicate=on_duplicate,
            )
        else:
            source_id = attached_irf_source_id
        if source_id is not None:
            _verify_irf_source(connection, source_id, measurement.irf)
        values = (
            measurement.require_raw_counts()
            if measurement.data_kind is MeasurementDataKind.RAW_COUNTS
            else measurement.values
        )
        return _record_observation(
            connection, measurement_key=measurement_key,
            time_ns=measurement.time_ns, values=values,
            data_kind=measurement.data_kind.value, source_type=source_type,
            condition_pk=condition_pk, origin_run_id=origin_run_id,
            attached_irf_source_id=source_id,
            histogram_artifact_id=histogram_artifact_id,
            sample_id=measurement.sample_id, observation_seed=observation_seed,
            metadata=measurement.metadata, provenance=measurement.provenance,
            on_duplicate=on_duplicate,
        )


def record_synthetic_observation(
    connection: sqlite3.Connection, *, measurement_key: str,
    time_ns: ArrayLike, values: ArrayLike,
    data_kind: Literal["raw_counts", "processed_intensity"],
    condition_pk: int | None = None, origin_run_id: int | None = None,
    attached_irf_source_id: int | None = None,
    histogram_artifact_id: int | None = None,
    sample_id: str | None = None,
    observation_seed: int | str | None = None,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Record one synthetic histogram without inferring truth from its metadata."""
    return _record_observation(
        connection, measurement_key=measurement_key,
        time_ns=time_ns, values=values, data_kind=data_kind,
        source_type="synthetic", condition_pk=condition_pk,
        origin_run_id=origin_run_id,
        attached_irf_source_id=attached_irf_source_id,
        histogram_artifact_id=histogram_artifact_id,
        sample_id=sample_id, observation_seed=observation_seed,
        metadata=metadata, provenance=provenance, on_duplicate=on_duplicate,
    )


def record_benchmark_observation(
    connection: sqlite3.Connection, measurements: Any, *, sample_index: int,
    measurement_key: str, condition_pk: int | None = None,
    origin_run_id: int | None = None, sample_id: str | None = None,
    observation_seed: int | str | None = None,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Select one documented raw observation from a benchmark measurement carrier."""
    from tcspc_toolkit.generalization_datasets import GeneralizationTestMeasurements
    from tcspc_toolkit.ml_evaluation import BenchmarkMeasurements

    if not isinstance(measurements, (BenchmarkMeasurements, GeneralizationTestMeasurements)):
        raise TypeError("expected BenchmarkMeasurements or GeneralizationTestMeasurements")
    index = _integer(sample_index, "sample_index")
    if index >= len(measurements.X_histograms):
        raise IndexError("sample_index is outside the benchmark measurements")
    return record_synthetic_observation(
        connection, measurement_key=measurement_key, time_ns=measurements.time,
        values=measurements.X_histograms[index], data_kind="raw_counts",
        condition_pk=condition_pk, origin_run_id=origin_run_id,
        sample_id=sample_id, observation_seed=observation_seed,
        metadata=metadata, provenance=provenance, on_duplicate=on_duplicate,
    )


def link_run_measurement(
    connection: sqlite3.Connection, *, run_id: int, measurement_id: int,
    data_role: str, test_id: str | None = None, regime_id: str | None = None,
    pair_id: str | None = None, realization_index: int | None = None,
    dataset_key: str | None = None,
    membership: Mapping[str, Any] | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> tuple[int, int]:
    """Preserve a caller-declared train, calibration, evaluation or demo role."""
    payload = {
        "run_id": _integer(run_id, "run_id", minimum=1),
        "measurement_id": _integer(measurement_id, "measurement_id", minimum=1),
        "data_role": _required_text(data_role, "data_role"),
        "test_id": _optional_text(test_id, "test_id"),
        "regime_id": _optional_text(regime_id, "regime_id"),
        "pair_id": _optional_text(pair_id, "pair_id"),
        "realization_index": _optional_integer(realization_index, "realization_index"),
        "dataset_key": _optional_text(dataset_key, "dataset_key"),
        "membership_json": _mapping_json(membership, "membership"),
    }
    return _record_row(
        connection, table="run_measurements",
        key_columns=("run_id", "measurement_id"), payload=payload,
        return_column=None, on_duplicate=on_duplicate,
    )


def bayesian_model_configuration(priors: Any, sampler: Any) -> dict[str, Any]:
    """Build reusable Bayesian configuration, excluding execution random seed."""
    from tcspc_toolkit.bayesian import BayesianPriorConfig
    from tcspc_toolkit.bayesian_sampling import BayesianSamplingConfig

    if not isinstance(priors, BayesianPriorConfig) or not isinstance(sampler, BayesianSamplingConfig):
        raise TypeError("expected BayesianPriorConfig and BayesianSamplingConfig")
    return _without_bayesian_sampling_seed({"priors": priors, "sampler": sampler})


def _without_bayesian_sampling_seed(value: Any) -> dict[str, Any]:
    """Omit only the recognized Bayesian sampler execution-seed field."""
    from tcspc_toolkit.bayesian_sampling import BayesianSamplingConfig

    normalized = _json_value(value)
    if not isinstance(normalized, dict):
        raise TypeError("Bayesian configuration must be a mapping or dataclass")
    if isinstance(value, BayesianSamplingConfig):
        normalized.pop("random_seed", None)
    elif isinstance(value, Mapping) and isinstance(
        value.get("sampler"), (BayesianSamplingConfig, Mapping)
    ):
        # The direct sampler.random_seed path is execution-specific. Other
        # seed-named fields have no such established meaning and are retained.
        normalized["sampler"].pop("random_seed", None)
    return normalized


def record_model_version(
    connection: sqlite3.Connection, *, model_key: str, estimator_name: str,
    family: Literal["classical", "ml", "bayesian", "baseline"],
    configuration: Mapping[str, Any] | Any,
    representation_id: str | None = None, prior_policy_id: str | None = None,
    implementation_version: str | None = None,
    training_run_id: int | None = None,
    trained_model_artifact_id: int | None = None,
    representation_artifact_id: int | None = None,
    provenance: Mapping[str, Any] | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Register one reusable estimator specification, not a fitted observation."""
    if family not in ("classical", "ml", "bayesian", "baseline"):
        raise ValueError("invalid model family")
    reusable_configuration = (
        _without_bayesian_sampling_seed(configuration)
        if family == "bayesian" else configuration
    )
    config_json, config_sha = _configuration_json(reusable_configuration)
    if family == "bayesian" and config_json == "{}":
        raise ValueError("Bayesian specification needs explicit prior/sampler configuration")
    payload = {
        "model_key": _required_text(model_key, "model_key"),
        "estimator_name": _required_text(estimator_name, "estimator_name"),
        "family": family,
        "configuration_json": config_json,
        "configuration_sha256": config_sha,
        "representation_id": _optional_text(representation_id, "representation_id"),
        "prior_policy_id": _optional_text(prior_policy_id, "prior_policy_id"),
        "implementation_version": _optional_text(implementation_version, "implementation_version"),
        "training_run_id": _optional_integer(training_run_id, "training_run_id", minimum=1),
        "trained_model_artifact_id": _optional_integer(
            trained_model_artifact_id, "trained_model_artifact_id", minimum=1,
        ),
        "representation_artifact_id": _optional_integer(
            representation_artifact_id, "representation_artifact_id", minimum=1,
        ),
        "provenance_json": _mapping_json(provenance, "provenance"),
    }
    return _record_row(
        connection, table="model_versions", key_columns=("model_key",),
        payload=payload, return_column="model_id", on_duplicate=on_duplicate,
    )


def record_model_assumption(
    connection: sqlite3.Connection, *, assumption_key: str,
    assumed_decay_model: str, observation_model: str,
    background_convention: str,
    context_completeness: Literal["complete", "historical_incomplete"],
    prepared_irf_id: int | None = None,
    source_assumption_id: str | None = None,
    fixed_temporal_shift_ns: float | None = None,
    temporal_shift_lower_ns: float | None = None,
    temporal_shift_upper_ns: float | None = None,
    configuration: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    bayesian_priors: Any | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Record physical inference assumptions independently of estimator priors."""
    from tcspc_toolkit.bayesian import BayesianPriorConfig

    decay = _required_text(assumed_decay_model, "assumed_decay_model")
    observation = _required_text(observation_model, "observation_model")
    if context_completeness not in ("complete", "historical_incomplete"):
        raise ValueError("invalid context_completeness")
    fixed = _optional_scalar(fixed_temporal_shift_ns, "fixed_temporal_shift_ns")
    lower = _optional_scalar(temporal_shift_lower_ns, "temporal_shift_lower_ns")
    upper = _optional_scalar(temporal_shift_upper_ns, "temporal_shift_upper_ns")
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("temporal shift bounds must be ordered")
    complete_poisson = context_completeness == "complete" and observation == "poisson_reconvolution"
    if complete_poisson:
        if prepared_irf_id is None:
            raise ValueError("complete Poisson assumption needs a prepared IRF")
        fixed_mode = fixed is not None and lower is None and upper is None
        bounded_mode = fixed is None and lower is not None and upper is not None
        if not (fixed_mode or bounded_mode):
            raise ValueError("complete Poisson assumption needs exactly one shift mode")
    if bayesian_priors is not None:
        if not isinstance(bayesian_priors, BayesianPriorConfig):
            raise TypeError("bayesian_priors must be a BayesianPriorConfig")
        if complete_poisson:
            if fixed is not None and bayesian_priors.fixed_temporal_shift_ns != fixed:
                raise ValueError("fixed physical shift conflicts with Bayesian prior")
            if fixed is None and (
                bayesian_priors.temporal_shift_prior is None
                or bayesian_priors.temporal_shift_prior.lower_ns < lower
                or bayesian_priors.temporal_shift_prior.upper_ns > upper
            ):
                raise ValueError("Bayesian shift support exceeds physical assumption bounds")
    payload = {
        "assumption_key": _required_text(assumption_key, "assumption_key"),
        "source_assumption_id": _optional_text(source_assumption_id, "source_assumption_id"),
        "assumed_decay_model": decay,
        "observation_model": observation,
        "background_convention": _required_text(background_convention, "background_convention"),
        "prepared_irf_id": _optional_integer(prepared_irf_id, "prepared_irf_id", minimum=1),
        "context_completeness": context_completeness,
        "temporal_shift_lower_ns": lower,
        "temporal_shift_upper_ns": upper,
        "fixed_temporal_shift_ns": fixed,
        "configuration_json": _mapping_json(configuration, "configuration"),
        "provenance_json": _mapping_json(provenance, "provenance"),
    }
    return _record_row(
        connection, table="model_assumptions", key_columns=("assumption_key",),
        payload=payload, return_column="assumption_id", on_duplicate=on_duplicate,
    )


def _reference_payload(
    *, reference_key: str, reference_kind: str, reference_version: str,
    measurement_id: int | None, condition_pk: int | None,
    assumption_id: int | None, lifetime_ns: float,
    source_artifact_id: int | None, supersedes_reference_id: int | None,
    projection_amplitude: float | None,
    projection_background_per_bin: float | None,
    projection_temporal_shift_ns: float | None,
    projection_poisson_nll: float | None,
    validation: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "reference_key": _required_text(reference_key, "reference_key"),
        "reference_kind": reference_kind,
        "reference_version": _required_text(reference_version, "reference_version"),
        "measurement_id": measurement_id,
        "condition_pk": condition_pk,
        "assumption_id": assumption_id,
        "source_artifact_id": _optional_integer(source_artifact_id, "source_artifact_id", minimum=1),
        "supersedes_reference_id": _optional_integer(
            supersedes_reference_id, "supersedes_reference_id", minimum=1,
        ),
        "lifetime_ns": _finite_scalar(lifetime_ns, "lifetime_ns", positive=True),
        "projection_amplitude": _optional_scalar(
            projection_amplitude, "projection_amplitude", nonnegative=True,
        ),
        "projection_background_per_bin": _optional_scalar(
            projection_background_per_bin, "projection_background_per_bin", nonnegative=True,
        ),
        "projection_temporal_shift_ns": _optional_scalar(
            projection_temporal_shift_ns, "projection_temporal_shift_ns",
        ),
        "projection_poisson_nll": _optional_scalar(projection_poisson_nll, "projection_poisson_nll"),
        "validation_json": _mapping_json(validation, "validation"),
        "metadata_json": _mapping_json(metadata, "metadata"),
        "provenance_json": _mapping_json(provenance, "provenance"),
    }


def _check_superseded_reference(
    connection: sqlite3.Connection, *, supersedes_reference_id: int | None,
    reference_kind: str, reference_version: str,
    measurement_id: int | None, condition_pk: int | None,
    assumption_id: int | None,
) -> None:
    if supersedes_reference_id is None:
        return
    old = connection.execute(
        "SELECT reference_kind, reference_version, measurement_id, condition_pk, "
        "assumption_id FROM lifetime_references WHERE reference_id = ?",
        (supersedes_reference_id,),
    ).fetchone()
    if old is None or (
        old[0] != reference_kind or old[1] == reference_version
        or old[2] != measurement_id or old[3] != condition_pk
        or old[4] != assumption_id
    ):
        raise ValueError("superseded reference must have the same role and scope, with a prior version")


def record_trusted_lifetime_reference(
    connection: sqlite3.Connection, *, reference_key: str,
    reference_version: str, measurement_id: int, lifetime_ns: float,
    source_artifact_id: int | None = None,
    supersedes_reference_id: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Attach a trusted external lifetime only to an experimental observation."""
    measured_id = _integer(measurement_id, "measurement_id", minimum=1)
    row = connection.execute(
        "SELECT source_type, condition_pk FROM measurements WHERE measurement_id = ?",
        (measured_id,),
    ).fetchone()
    if row is None or tuple(row) != ("experimental", None):
        raise ValueError("trusted experimental reference requires an experimental measurement")
    _check_superseded_reference(
        connection, supersedes_reference_id=supersedes_reference_id,
        reference_kind="trusted_experimental", reference_version=reference_version,
        measurement_id=measured_id, condition_pk=None, assumption_id=None,
    )
    payload = _reference_payload(
        reference_key=reference_key, reference_kind="trusted_experimental",
        reference_version=reference_version, measurement_id=measured_id,
        condition_pk=None, assumption_id=None, lifetime_ns=lifetime_ns,
        source_artifact_id=source_artifact_id,
        supersedes_reference_id=supersedes_reference_id,
        projection_amplitude=None, projection_background_per_bin=None,
        projection_temporal_shift_ns=None, projection_poisson_nll=None,
        validation=None, metadata=metadata, provenance=provenance,
    )
    return _record_row(
        connection, table="lifetime_references", key_columns=("reference_key",),
        payload=payload, return_column="reference_id", on_duplicate=on_duplicate,
    )


def _validate_assumed_temporal_shift(
    shift: float, *, context_completeness: str,
    fixed_shift: float | None, lower_shift: float | None,
    upper_shift: float | None, label: str,
) -> None:
    """Check a valid inferred shift against an explicitly complete physical policy."""
    if context_completeness != "complete":
        return
    if fixed_shift is not None:
        if not math.isclose(shift, fixed_shift, rel_tol=1e-7, abs_tol=1e-12):
            raise ValueError(f"{label} shift conflicts with the fixed model assumption")
    elif lower_shift is None or upper_shift is None or not lower_shift <= shift <= upper_shift:
        raise ValueError(f"{label} shift lies outside the model assumption bounds")


def record_pseudo_true_reference(
    connection: sqlite3.Connection, *, reference_key: str,
    reference_version: str, condition_pk: int, assumption_id: int,
    lifetime_ns: float, projection_amplitude: float | None = None,
    projection_background_per_bin: float | None = None,
    projection_temporal_shift_ns: float | None = None,
    projection_poisson_nll: float | None = None,
    source_artifact_id: int | None = None,
    supersedes_reference_id: int | None = None,
    validation: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Record a prior-free, model-conditional mono-exponential projection."""
    condition_id = _integer(condition_pk, "condition_pk", minimum=1)
    assumed_id = _integer(assumption_id, "assumption_id", minimum=1)
    _require_row(connection, "simulation_conditions", "condition_pk", condition_id)
    assumption = connection.execute(
        "SELECT assumed_decay_model, observation_model, context_completeness, "
        "fixed_temporal_shift_ns, temporal_shift_lower_ns, temporal_shift_upper_ns "
        "FROM model_assumptions "
        "WHERE assumption_id = ?", (assumed_id,),
    ).fetchone()
    if assumption is None or tuple(assumption[:2]) != ("monoexponential", "poisson_reconvolution"):
        raise ValueError("pseudo-true mono projection requires a mono Poisson assumption")
    shift = _optional_scalar(projection_temporal_shift_ns, "projection_temporal_shift_ns")
    if shift is not None:
        _validate_assumed_temporal_shift(
            shift, context_completeness=assumption[2],
            fixed_shift=assumption[3], lower_shift=assumption[4],
            upper_shift=assumption[5], label="projection",
        )
    _check_superseded_reference(
        connection, supersedes_reference_id=supersedes_reference_id,
        reference_kind="pseudo_true_mono", reference_version=reference_version,
        measurement_id=None, condition_pk=condition_id, assumption_id=assumed_id,
    )
    payload = _reference_payload(
        reference_key=reference_key, reference_kind="pseudo_true_mono",
        reference_version=reference_version, measurement_id=None,
        condition_pk=condition_id, assumption_id=assumed_id,
        lifetime_ns=lifetime_ns, source_artifact_id=source_artifact_id,
        supersedes_reference_id=supersedes_reference_id,
        projection_amplitude=projection_amplitude,
        projection_background_per_bin=projection_background_per_bin,
        projection_temporal_shift_ns=projection_temporal_shift_ns,
        projection_poisson_nll=projection_poisson_nll,
        validation=validation, metadata=metadata, provenance=provenance,
    )
    return _record_row(
        connection, table="lifetime_references", key_columns=("reference_key",),
        payload=payload, return_column="reference_id", on_duplicate=on_duplicate,
    )


def record_issue4_pseudo_true_reference(
    connection: sqlite3.Connection, reference: Any, *, reference_key: str,
    reference_version: str, condition_pk: int, assumption_id: int,
    source_artifact_id: int | None = None,
    supersedes_reference_id: int | None = None,
    provenance: Mapping[str, Any] | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Record an Issue-4 projection with its scoped IDs and numerical audit."""
    from tcspc_toolkit.bayesian_mismatch_evaluation import PseudoTrueReference

    if not isinstance(reference, PseudoTrueReference):
        raise TypeError("reference must be a PseudoTrueReference")
    condition = connection.execute(
        "SELECT condition_id FROM simulation_conditions WHERE condition_pk = ?",
        (condition_pk,),
    ).fetchone()
    assumption = connection.execute(
        "SELECT source_assumption_id FROM model_assumptions WHERE assumption_id = ?",
        (assumption_id,),
    ).fetchone()
    if condition is None or condition[0] != reference.condition_id:
        raise ValueError("condition_pk does not match the pseudo-true source condition")
    if assumption is None or assumption[0] != reference.assumption_id:
        raise ValueError("assumption_id does not match the pseudo-true source assumption")

    scalar_columns = {
        "condition_id", "assumption_id", "amplitude", "lifetime_ns",
        "background_per_bin", "temporal_shift_ns", "poisson_nll",
    }
    diagnostics: dict[str, Any] = {}
    nonfinite_fields: list[str] = []
    for item in fields(reference):
        if item.name in scalar_columns:
            continue
        value = getattr(reference, item.name)
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            diagnostics[item.name] = None
            nonfinite_fields.append(item.name)
        else:
            diagnostics[item.name] = value
    diagnostics["nonfinite_fields"] = nonfinite_fields
    return record_pseudo_true_reference(
        connection, reference_key=reference_key, reference_version=reference_version,
        condition_pk=condition_pk, assumption_id=assumption_id,
        lifetime_ns=reference.lifetime_ns,
        projection_amplitude=reference.amplitude,
        projection_background_per_bin=reference.background_per_bin,
        projection_temporal_shift_ns=reference.temporal_shift_ns,
        projection_poisson_nll=reference.poisson_nll,
        source_artifact_id=source_artifact_id,
        supersedes_reference_id=supersedes_reference_id,
        validation=diagnostics, provenance=provenance,
        on_duplicate=on_duplicate,
    )


# Stage-3 point results. Source dataclasses stay in their scientific modules.
def _result_number(
    value: Any, name: str, nonfinite: dict[str, str], *, nonnegative: bool = False,
) -> float | None:
    """Keep finite result scalars relational and classify known nonfinite values."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric or None")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric or None") from exc
    if math.isnan(number):
        nonfinite[name] = "nan"
        return None
    if math.isinf(number):
        nonfinite[name] = "+inf" if number > 0 else "-inf"
        return None
    if nonnegative and number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _result_flag(value: Any, name: str) -> int:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return int(value)


def _optional_result_flag(value: Any, name: str) -> int | None:
    return None if value is None else _result_flag(value, name)


def _optional_diagnostic_text(value: Any, name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be text or None")
    return value


def _optional_optimizer_status(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("optimizer_status must be an integer or None")
    status = int(value)
    if not -(2**63) <= status <= 2**63 - 1:
        raise ValueError("optimizer_status exceeds SQLite int64 range")
    return status


def _estimator_payload(
    *, run_id: int, measurement_id: int, model_id: int,
    assumption_id: int | None, analysis_key: str, point_summary: str,
    status: Literal["available", "failed", "unavailable"], is_valid: bool,
    source_result_type: str, irf_model_relation: str,
    lifetime_estimate_ns: Any, failure_reason: str | None,
    runtime_seconds: Any, runtime_scope: str | None,
    random_seed: int | str | None, execution: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if status not in ("available", "failed", "unavailable"):
        raise ValueError("invalid estimator result status")
    if irf_model_relation not in (
        "unspecified", "matched", "deliberately_misspecified",
    ):
        raise ValueError("invalid IRF model relation")
    valid = _result_flag(is_valid, "is_valid")
    assumed_id = _optional_integer(assumption_id, "assumption_id", minimum=1)
    if assumed_id is None and irf_model_relation != "unspecified":
        raise ValueError("an IRF model relation requires a physical assumption")
    if (runtime_seconds is None) != (runtime_scope is None):
        raise ValueError("runtime_seconds and runtime_scope must be supplied together")
    nonfinite: dict[str, str] = {}
    lifetime = _result_number(lifetime_estimate_ns, "lifetime_estimate_ns", nonfinite)
    runtime = _result_number(
        runtime_seconds, "runtime_seconds", nonfinite, nonnegative=True,
    )
    if valid and (status != "available" or lifetime is None):
        raise ValueError("valid result requires an available finite lifetime estimate")
    return {
        "run_id": _integer(run_id, "run_id", minimum=1),
        "measurement_id": _integer(measurement_id, "measurement_id", minimum=1),
        "model_id": _integer(model_id, "model_id", minimum=1),
        "assumption_id": assumed_id,
        "analysis_key": _required_text(analysis_key, "analysis_key"),
        "point_summary": _required_text(point_summary, "point_summary"),
        "status": status,
        "is_valid": valid,
        "irf_model_relation": irf_model_relation,
        "source_result_type": _required_text(source_result_type, "source_result_type"),
        "lifetime_estimate_ns": lifetime,
        "failure_reason": _optional_diagnostic_text(failure_reason, "failure_reason"),
        "runtime_seconds": runtime,
        "runtime_scope": _optional_text(runtime_scope, "runtime_scope"),
        "random_seed_decimal": _decimal_seed(random_seed, "random_seed"),
        "nonfinite_fields_json": _canonical_json(nonfinite),
        "execution_json": _mapping_json(execution, "execution"),
    }


def _require_result_model(
    connection: sqlite3.Connection, model_id: int, *, families: tuple[str, ...],
) -> str:
    row = connection.execute(
        "SELECT family, estimator_name FROM model_versions WHERE model_id = ?",
        (_integer(model_id, "model_id", minimum=1),),
    ).fetchone()
    if row is None or row[0] not in families:
        raise ValueError(f"model_id must identify a {', '.join(families)} specification")
    return str(row[1])


def _record_estimator_row(
    connection: sqlite3.Connection, payload: dict[str, Any], *,
    on_duplicate: DuplicatePolicy,
) -> tuple[int, bool]:
    """Insert or compare one NULL-safe run/measurement/model/assumption analysis."""
    if on_duplicate not in ("raise", "reuse_identical"):
        raise ValueError("on_duplicate must be 'raise' or 'reuse_identical'")
    if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise ValueError("foreign keys must be enabled before recording")
    columns = [row[1] for row in connection.execute("PRAGMA table_info(estimator_results)")]
    if set(payload) != set(columns) - {"result_id"}:
        raise RuntimeError("incomplete estimator_results adapter payload")
    with transaction(connection):
        membership = connection.execute(
            "SELECT 1 FROM run_measurements WHERE run_id = ? AND measurement_id = ?",
            (payload["run_id"], payload["measurement_id"]),
        ).fetchone()
        if membership is None:
            raise ValueError("run_measurements membership is required for a result")
        cursor = connection.execute(
            "SELECT * FROM estimator_results WHERE run_id = ? AND measurement_id = ? "
            "AND model_id = ? AND assumption_id IS ? AND analysis_key = ?",
            (payload["run_id"], payload["measurement_id"], payload["model_id"],
             payload["assumption_id"], payload["analysis_key"]),
        )
        found = cursor.fetchone()
        if found is not None:
            existing = dict(zip((item[0] for item in cursor.description), found))
            if on_duplicate == "raise":
                raise PersistenceConflictError("duplicate estimator_results identity")
            if not all(
                _same_payload_value(column, existing[column], value)
                for column, value in payload.items()
            ):
                raise PersistenceConflictError("conflicting estimator_results identity")
            return int(existing["result_id"]), False
        names = tuple(payload)
        placeholders = ", ".join("?" for _ in names)
        inserted = connection.execute(
            f"INSERT INTO estimator_results ({', '.join(names)}) VALUES ({placeholders})",
            tuple(payload.values()),
        )
        return int(inserted.lastrowid), True


def _classical_fit_payloads(
    result: Any, *, run_id: int, measurement_id: int, model_id: int,
    assumption_id: int, analysis_key: str, irf_model_relation: str,
    execution: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from tcspc_toolkit.classical_evaluation import ReconvolutionCurveResult
    from tcspc_toolkit.fitting import ReconvolutionFitResult

    if isinstance(result, ReconvolutionFitResult):
        valid = _result_flag(result.success, "success")
        lifetime = result.lifetime
        amplitude, background, shift = (
            result.amplitude, result.background, result.temporal_shift,
        )
        initial = (None, None, None, None)
        optimizer_success = result.optimizer_reported_success
        boundary_hit = None
        poisson_nll = poisson_deviance = None
        failure_reason = exception_message = None
        runtime_seconds = runtime_scope = None
        source_type = "tcspc_toolkit.fitting.ReconvolutionFitResult"
    elif isinstance(result, ReconvolutionCurveResult):
        valid = _result_flag(result.valid_fit, "valid_fit")
        lifetime = result.fitted_lifetime_ns
        amplitude, background, shift = (
            result.fitted_amplitude, result.fitted_background,
            result.fitted_temporal_shift_ns,
        )
        initial = (
            result.initial_amplitude, result.initial_lifetime_ns,
            result.initial_background, result.initial_temporal_shift_ns,
        )
        optimizer_success = result.optimizer_success
        boundary_hit = result.boundary_hit
        poisson_nll, poisson_deviance = result.poisson_nll, result.poisson_deviance
        failure_reason, exception_message = result.failure_reason, result.exception_message
        try:
            runtime_seconds = float(result.runtime_ms) / 1000.0
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("runtime_ms must be numeric") from exc
        runtime_scope = "curve_fit_phase_after_initialization"
        source_type = "tcspc_toolkit.classical_evaluation.ReconvolutionCurveResult"
    else:
        raise TypeError("result must be ReconvolutionFitResult or ReconvolutionCurveResult")

    numerical = _optional_result_flag(
        result.numerical_validation_passed, "numerical_validation_passed",
    )
    if valid and numerical == 0:
        raise ValueError("a numerically rejected classical fit cannot be valid")
    point = _estimator_payload(
        run_id=run_id, measurement_id=measurement_id, model_id=model_id,
        assumption_id=assumption_id, analysis_key=analysis_key,
        point_summary="fitted_lifetime_ns",
        status="available" if valid else "failed", is_valid=bool(valid),
        source_result_type=source_type, irf_model_relation=irf_model_relation,
        lifetime_estimate_ns=lifetime, failure_reason=failure_reason,
        runtime_seconds=runtime_seconds, runtime_scope=runtime_scope,
        random_seed=None, execution=execution,
    )
    if valid and point["lifetime_estimate_ns"] <= 0:
        raise ValueError("a valid classical fit must have a positive lifetime")
    details_nonfinite: dict[str, str] = {}
    details = {
        "result_id": 0,  # Replaced after the point row is inserted.
        "source_success": valid,
        "optimizer_reported_success": _optional_result_flag(
            optimizer_success, "optimizer_reported_success",
        ),
        "valid_fit": valid,
        "numerical_validation_passed": numerical,
        "recovery_attempted": _result_flag(result.recovery_attempted, "recovery_attempted"),
        "boundary_hit": _optional_result_flag(boundary_hit, "boundary_hit"),
        "fitted_amplitude": _result_number(
            amplitude, "fitted_amplitude", details_nonfinite,
        ),
        "fitted_background_per_bin": _result_number(
            background, "fitted_background_per_bin", details_nonfinite,
        ),
        "fitted_temporal_shift_ns": _result_number(
            shift, "fitted_temporal_shift_ns", details_nonfinite,
        ),
        "initial_amplitude": _result_number(
            initial[0], "initial_amplitude", details_nonfinite,
        ),
        "initial_lifetime_ns": _result_number(
            initial[1], "initial_lifetime_ns", details_nonfinite,
        ),
        "initial_background_per_bin": _result_number(
            initial[2], "initial_background_per_bin", details_nonfinite,
        ),
        "initial_temporal_shift_ns": _result_number(
            initial[3], "initial_temporal_shift_ns", details_nonfinite,
        ),
        "poisson_nll": _result_number(poisson_nll, "poisson_nll", details_nonfinite),
        "poisson_deviance": _result_number(
            poisson_deviance, "poisson_deviance", details_nonfinite,
        ),
        "max_coordinate_descent_nll": _result_number(
            result.max_coordinate_descent_nll,
            "max_coordinate_descent_nll", details_nonfinite,
        ),
        "optimizer_status": _optional_optimizer_status(result.optimizer_status),
        "optimizer_message": _optional_diagnostic_text(
            result.optimizer_message, "optimizer_message",
        ),
        "optimizer_nfev": _optional_integer(result.optimizer_nfev, "optimizer_nfev"),
        "optimizer_njev": _optional_integer(result.optimizer_njev, "optimizer_njev"),
        "optimizer_seconds": None,
        "call_seconds": None,
        "exception_message": _optional_diagnostic_text(
            exception_message, "exception_message",
        ),
        "diagnostics_json": "{}",
        "nonfinite_fields_json": _canonical_json(details_nonfinite),
    }
    if valid and (
        details["optimizer_reported_success"] == 0
        or details["fitted_amplitude"] is None
        or details["fitted_amplitude"] < 0
        or details["fitted_background_per_bin"] is None
        or details["fitted_background_per_bin"] < 0
        or details["fitted_temporal_shift_ns"] is None
    ):
        raise ValueError("a valid classical fit requires finite physical parameters")
    return point, details


def record_reconvolution_fit(
    connection: sqlite3.Connection, result: Any, *, run_id: int,
    measurement_id: int, model_id: int, assumption_id: int,
    analysis_key: str = "default", irf_model_relation: str = "unspecified",
    execution: Mapping[str, Any] | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Atomically record a classical point result and its fit-only details."""
    run = _integer(run_id, "run_id", minimum=1)
    measured = _integer(measurement_id, "measurement_id", minimum=1)
    model = _integer(model_id, "model_id", minimum=1)
    assumed = _integer(assumption_id, "assumption_id", minimum=1)
    if connection.execute(
        "SELECT 1 FROM run_measurements WHERE run_id = ? AND measurement_id = ?",
        (run, measured),
    ).fetchone() is None:
        raise ValueError("run_measurements membership is required for a result")
    model_row = connection.execute(
        "SELECT family, configuration_json FROM model_versions WHERE model_id = ?",
        (model,),
    ).fetchone()
    if model_row is None or model_row[0] != "classical":
        raise ValueError("model_id must identify a classical specification")
    configuration = json.loads(model_row[1])
    objective = configuration.get("objective") if isinstance(configuration, dict) else None
    if not isinstance(objective, str) or objective not in ("poisson", "least_squares"):
        raise ValueError("classical model configuration requires a recognized objective")
    assumption = connection.execute(
        "SELECT assumed_decay_model, observation_model, context_completeness, "
        "prepared_irf_id, fixed_temporal_shift_ns, temporal_shift_lower_ns, "
        "temporal_shift_upper_ns FROM model_assumptions WHERE assumption_id = ?",
        (assumed,),
    ).fetchone()
    if assumption is None or assumption[0] != "monoexponential":
        raise ValueError("reconvolution fit requires a monoexponential model assumption")
    observation_objective = {
        "poisson_reconvolution": "poisson",
        "least_squares_reconvolution": "least_squares",
    }.get(assumption[1])
    if observation_objective is not None and objective != observation_objective:
        raise ValueError("classical model objective conflicts with observation assumption")
    if assumption[2] == "complete" and assumption[3] is None:
        raise ValueError("complete reconvolution assumption requires a prepared IRF")
    measurement = connection.execute(
        "SELECT data_kind, n_bins, time_grid_sha256, time_start_ns, time_step_ns "
        "FROM measurements WHERE measurement_id = ?", (measured,),
    ).fetchone()
    if measurement is None:
        raise ValueError("measurement_id must identify a stored measurement")
    if objective == "poisson" and measurement[0] != "raw_counts":
        raise ValueError("Poisson reconvolution requires a raw-count measurement")
    if assumption[3] is not None:
        prepared = connection.execute(
            "SELECT n_bins, time_grid_sha256, time_start_ns, time_step_ns "
            "FROM prepared_irfs WHERE prepared_irf_id = ?", (assumption[3],),
        ).fetchone()
        if prepared is None or tuple(measurement[1:]) != tuple(prepared):
            raise ValueError("prepared IRF target grid differs from measurement time grid")
    point, details = _classical_fit_payloads(
        result, run_id=run, measurement_id=measured,
        model_id=model, assumption_id=assumed,
        analysis_key=analysis_key, irf_model_relation=irf_model_relation,
        execution=execution,
    )
    if point["is_valid"]:
        _validate_assumed_temporal_shift(
            details["fitted_temporal_shift_ns"],
            context_completeness=assumption[2], fixed_shift=assumption[4],
            lower_shift=assumption[5], upper_shift=assumption[6], label="fitted",
        )
    with transaction(connection):
        result_id, inserted = _record_estimator_row(
            connection, point, on_duplicate=on_duplicate,
        )
        details["result_id"] = result_id
        if inserted:
            _record_row(
                connection, table="fit_details", key_columns=("result_id",),
                payload=details, return_column=None, on_duplicate="raise",
            )
        else:
            cursor = connection.execute(
                "SELECT * FROM fit_details WHERE result_id = ?", (result_id,),
            )
            found = cursor.fetchone()
            if found is None:
                raise PersistenceConflictError("reused classical result lacks fit_details")
            existing = dict(zip((item[0] for item in cursor.description), found))
            if not all(
                _same_payload_value(column, existing[column], value)
                for column, value in details.items()
            ):
                raise PersistenceConflictError("conflicting fit_details for reused result")
        return result_id


def _prediction_vector(predictions: ArrayLike) -> np.ndarray:
    source = _numeric_vector(predictions, name="predictions")
    try:
        values = np.asarray(source, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("predictions must be representable as float64") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("benchmark and baseline predictions must be finite")
    return values


def _ordered_measurement_ids(
    measurement_ids: Sequence[int], *, n_predictions: int,
) -> tuple[int, ...]:
    if isinstance(measurement_ids, np.ndarray):
        if measurement_ids.ndim != 1:
            raise ValueError("measurement_ids must be a one-dimensional ordered sequence")
        supplied = tuple(measurement_ids)
    elif isinstance(measurement_ids, Sequence) and not isinstance(
        measurement_ids, (str, bytes),
    ):
        supplied = tuple(measurement_ids)
    else:
        raise ValueError("measurement_ids must be an ordered sequence of IDs")
    if len(supplied) != n_predictions:
        raise ValueError("measurement_ids length must equal prediction count")
    ids = tuple(_integer(item, "measurement_id", minimum=1) for item in supplied)
    if len(set(ids)) != len(ids):
        raise ValueError("measurement_ids must not repeat within one prediction batch")
    return ids


def _record_prediction_batch(
    connection: sqlite3.Connection, *, predictions: np.ndarray,
    measurement_ids: tuple[int, ...], run_id: int, model_id: int,
    assumption_id: int | None, analysis_key: str, source_result_type: str,
    irf_model_relation: str, on_duplicate: DuplicatePolicy,
) -> tuple[int, ...]:
    # Normalize before opening the write transaction; the ordered index is
    # evidence of the supplied array-to-measurement alignment.
    payloads = tuple(
        _estimator_payload(
            run_id=run_id, measurement_id=measurement_id, model_id=model_id,
            assumption_id=assumption_id, analysis_key=analysis_key,
            point_summary="predicted_lifetime_ns", status="available",
            is_valid=True, source_result_type=source_result_type,
            irf_model_relation=irf_model_relation,
            lifetime_estimate_ns=prediction, failure_reason=None,
            runtime_seconds=None, runtime_scope=None, random_seed=None,
            execution={"prediction_index": index},
        )
        for index, (measurement_id, prediction) in enumerate(
            zip(measurement_ids, predictions, strict=True)
        )
    )
    with transaction(connection):
        return tuple(
            _record_estimator_row(connection, payload, on_duplicate=on_duplicate)[0]
            for payload in payloads
        )


def record_regression_predictions(
    connection: sqlite3.Connection, result: Any, *, run_id: int,
    measurement_ids: Sequence[int], model_id: int,
    assumption_id: int | None = None, analysis_key: str = "default",
    irf_model_relation: str = "unspecified",
    on_duplicate: DuplicatePolicy = "raise",
) -> tuple[int, ...]:
    """Record ordered RegressionBenchmarkResult predictions, never its metrics."""
    from tcspc_toolkit.ml_evaluation import RegressionBenchmarkResult

    if not isinstance(result, RegressionBenchmarkResult):
        raise TypeError("result must be a RegressionBenchmarkResult")
    estimator_name = _require_result_model(
        connection, model_id, families=("ml", "baseline"),
    )
    if estimator_name != result.estimator_name:
        raise ValueError("model_id estimator_name differs from benchmark result")
    predictions = _prediction_vector(result.y_pred)
    ids = _ordered_measurement_ids(measurement_ids, n_predictions=len(predictions))
    return _record_prediction_batch(
        connection, predictions=predictions, measurement_ids=ids,
        run_id=run_id, model_id=model_id, assumption_id=assumption_id,
        analysis_key=analysis_key,
        source_result_type="tcspc_toolkit.ml_evaluation.RegressionBenchmarkResult",
        irf_model_relation=irf_model_relation, on_duplicate=on_duplicate,
    )


def record_baseline_predictions(
    connection: sqlite3.Connection, predictions: ArrayLike, *,
    baseline_name: Literal["constant_mean", "mean_arrival_time"],
    run_id: int, measurement_ids: Sequence[int], model_id: int,
    assumption_id: int | None = None, analysis_key: str = "default",
    irf_model_relation: str = "unspecified",
    on_duplicate: DuplicatePolicy = "raise",
) -> tuple[int, ...]:
    """Record one ordered array from either current baseline estimator."""
    sources = {
        "constant_mean": "tcspc_toolkit.baselines.predict_constant_mean_baseline",
        "mean_arrival_time": "tcspc_toolkit.baselines.estimate_lifetime_from_mean_arrival",
    }
    if baseline_name not in sources:
        raise ValueError("unsupported baseline_name")
    estimator_name = _require_result_model(
        connection, model_id, families=("baseline",),
    )
    if estimator_name != baseline_name:
        raise ValueError("model_id estimator_name differs from baseline_name")
    values = _prediction_vector(predictions)
    ids = _ordered_measurement_ids(measurement_ids, n_predictions=len(values))
    return _record_prediction_batch(
        connection, predictions=values, measurement_ids=ids,
        run_id=run_id, model_id=model_id, assumption_id=assumption_id,
        analysis_key=analysis_key, source_result_type=sources[baseline_name],
        irf_model_relation=irf_model_relation, on_duplicate=on_duplicate,
    )


def record_scalar_prediction(
    connection: sqlite3.Connection, prediction_ns: Any, *,
    run_id: int, measurement_id: int, model_id: int,
    source_result_type: str, assumption_id: int | None = None,
    analysis_key: str = "default",
    status: Literal["available", "failed", "unavailable"] = "available",
    is_valid: bool = True, failure_reason: str | None = None,
    irf_model_relation: str = "unspecified",
    runtime_seconds: float | None = None, runtime_scope: str | None = None,
    random_seed: int | str | None = None,
    execution: Mapping[str, Any] | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Record an explicitly identified already-fitted ML or baseline output."""
    _require_result_model(connection, model_id, families=("ml", "baseline"))
    payload = _estimator_payload(
        run_id=run_id, measurement_id=measurement_id, model_id=model_id,
        assumption_id=assumption_id, analysis_key=analysis_key,
        point_summary="predicted_lifetime_ns", status=status, is_valid=is_valid,
        source_result_type=source_result_type,
        irf_model_relation=irf_model_relation,
        lifetime_estimate_ns=prediction_ns, failure_reason=failure_reason,
        runtime_seconds=runtime_seconds, runtime_scope=runtime_scope,
        random_seed=random_seed, execution=execution,
    )
    return _record_estimator_row(connection, payload, on_duplicate=on_duplicate)[0]
