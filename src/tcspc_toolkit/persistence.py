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
from dataclasses import dataclass, fields, is_dataclass, replace
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


# Stage-5 Bayesian point results and their posterior/sampler/PPC extension.
def _bayesian_array_summary(
    values: Any, name: str, nonfinite: dict[str, str],
) -> list[Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim not in (1, 2):
        raise ValueError(f"{name} must have one or two dimensions")
    if array.ndim == 1:
        return [_result_number(value, f"{name}[{index}]", nonfinite)
                for index, value in enumerate(array)]
    return [
        [_result_number(value, f"{name}[{row},{column}]", nonfinite)
         for column, value in enumerate(line)]
        for row, line in enumerate(array)
    ]


def _bayesian_probability(
    value: Any, name: str, nonfinite: dict[str, str],
) -> float | None:
    number = _result_number(value, name, nonfinite)
    if number is not None and not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie between zero and one")
    return number


def _bayesian_correlation(
    value: Any, name: str, nonfinite: dict[str, str],
) -> float | None:
    number = _result_number(value, name, nonfinite)
    if number is not None and not -1.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a correlation in [-1, 1]")
    return number


def _bayesian_model_context(
    connection: sqlite3.Connection, *, model_id: int, assumption_id: int,
    measurement_id: int, source: Any, context: Any | None,
    sampling_seed: int | str,
) -> tuple[dict[str, Any], dict[str, Any], sqlite3.Row | tuple]:
    """Validate the reusable prior and the separately persisted physical model."""
    from tcspc_toolkit.bayesian import (
        BayesianPriorConfig, BoundedUniformPrior, GammaPrior, LogNormalPrior,
    )
    from tcspc_toolkit.bayesian_sampling import BayesianReconvolutionResult, BayesianSamplingConfig

    model = connection.execute(
        "SELECT family, estimator_name, configuration_json, prior_policy_id "
        "FROM model_versions WHERE model_id = ?", (model_id,),
    ).fetchone()
    if model is None or model[0] != "bayesian":
        raise ValueError("model_id must identify the current Bayesian Poisson reconvolution model")
    configuration = json.loads(model[2])
    if not isinstance(configuration, dict) or not isinstance(configuration.get("priors"), dict) \
            or not isinstance(configuration.get("sampler"), dict):
        raise ValueError("Bayesian model requires explicit prior and sampler configuration")
    # Compact Issue-4 records omit the policy objects. Require the complete
    # canonical Stage-2 configuration instead of interpreting a prior label.
    try:
        prior_config = configuration["priors"]
        shift_prior = prior_config["temporal_shift_prior"]
        validated_priors = BayesianPriorConfig(
            amplitude=GammaPrior(**prior_config["amplitude"]),
            lifetime_ns=LogNormalPrior(**prior_config["lifetime_ns"]),
            background_per_bin=GammaPrior(**prior_config["background_per_bin"]),
            temporal_shift_prior=None if shift_prior is None else BoundedUniformPrior(**shift_prior),
            fixed_temporal_shift_ns=prior_config["fixed_temporal_shift_ns"],
        )
        validated_sampler = BayesianSamplingConfig(
            random_seed=int(_decimal_seed(sampling_seed, "sampling_seed")),
            **configuration["sampler"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Bayesian model requires the complete current prior/sampler policy") from exc
    reusable = bayesian_model_configuration(validated_priors, validated_sampler)
    if any(configuration[key] != reusable[key]
           for key in ("priors", "sampler")):
        raise ValueError("Bayesian model requires the complete canonical prior/sampler policy")
    if isinstance(source, BayesianReconvolutionResult) and (
        configuration["priors"] != _json_value(source.priors)
        or configuration["sampler"] != _without_bayesian_sampling_seed(source.sampling_config)
    ):
        raise ValueError("Bayesian result prior/sampler policy conflicts with model specification")
    assumption = connection.execute(
        "SELECT assumed_decay_model, observation_model, context_completeness, "
        "prepared_irf_id, fixed_temporal_shift_ns, temporal_shift_lower_ns, "
        "temporal_shift_upper_ns, source_assumption_id "
        "FROM model_assumptions WHERE assumption_id = ?", (assumption_id,),
    ).fetchone()
    if assumption is None or assumption[0] != "monoexponential" \
            or assumption[1] != "poisson_reconvolution":
        raise ValueError("Bayesian result requires a monoexponential Poisson assumption")
    measurement = connection.execute(
        "SELECT data_kind, n_bins, time_grid_sha256, time_start_ns, time_step_ns, "
        "values_sha256, condition_pk, attached_irf_source_id, sample_id, metadata_json, provenance_json "
        "FROM measurements WHERE measurement_id = ?",
        (measurement_id,),
    ).fetchone()
    if measurement is None or measurement[0] != "raw_counts":
        raise ValueError("Bayesian Poisson inference requires a raw-count measurement")
    if assumption[2] == "complete" and assumption[3] is None:
        raise ValueError("complete Bayesian Poisson assumption requires a prepared IRF")
    if assumption[3] is not None:
        prepared = connection.execute(
            "SELECT n_bins, time_grid_sha256, time_start_ns, time_step_ns "
            "FROM prepared_irfs WHERE prepared_irf_id = ?", (assumption[3],),
        ).fetchone()
        if prepared is None or tuple(measurement[1:5]) != tuple(prepared):
            raise ValueError("prepared IRF target grid differs from measurement time grid")
    priors = configuration["priors"]
    fixed_prior = priors.get("fixed_temporal_shift_ns")
    bounded_prior = priors.get("temporal_shift_prior")
    if (fixed_prior is None) == (bounded_prior is None):
        raise ValueError("Bayesian model requires exactly one temporal-shift prior mode")
    if assumption[2] == "complete":
        if assumption[4] is not None:
            if fixed_prior is None:
                raise ValueError("Bayesian shift prior conflicts with fixed model assumption")
            prior_shifts = (fixed_prior,)
        else:
            if bounded_prior is None:
                raise ValueError("Bayesian free-shift assumption requires a bounded shift prior")
            prior_shifts = (bounded_prior["lower_ns"], bounded_prior["upper_ns"])
        for shift in prior_shifts:
            _validate_assumed_temporal_shift(
                shift, context_completeness=assumption[2], fixed_shift=assumption[4],
                lower_shift=assumption[5], upper_shift=assumption[6], label="Bayesian prior",
            )
    if isinstance(source, BayesianReconvolutionResult):
        if source.model_context.sample_id is not None and measurement[8] is not None \
                and source.model_context.sample_id != measurement[8]:
            raise ValueError("Bayesian source sample identity conflicts with stored measurement")
        if not source.model_context.fixed_irf_assumption:
            raise ValueError("current Bayesian source requires a fixed IRF assumption")
        selection = source.model_context.irf_selection
        if selection not in ("explicit_prepared", "attached_sampled"):
            raise ValueError("unknown Bayesian source IRF selection")
        if selection == "explicit_prepared" and assumption[3] is None:
            raise ValueError("explicit Bayesian prepared IRF is missing from model assumption")
        if assumption[3] is not None:
            source_kind = connection.execute(
                "SELECT s.source_kind FROM prepared_irfs AS p LEFT JOIN irf_sources AS s "
                "ON p.irf_source_id = s.irf_source_id WHERE p.prepared_irf_id = ?",
                (assumption[3],),
            ).fetchone()[0]
            if source_kind is not None and source_kind != source.model_context.irf_source_kind.value:
                raise ValueError("Bayesian IRF source kind conflicts with assumed prepared IRF")
        if selection == "attached_sampled" and assumption[3] is not None:
            prepared_source = connection.execute(
                "SELECT irf_source_id FROM prepared_irfs WHERE prepared_irf_id = ?",
                (assumption[3],),
            ).fetchone()[0]
            if measurement[7] is None or prepared_source != measurement[7]:
                raise ValueError("Bayesian attached IRF conflicts with assumed prepared source")
    if context is not None:
        if isinstance(source, BayesianReconvolutionResult) and (
            context.irf_model_relation != source.model_context.irf_model_relation
            or context.irf_selection != source.model_context.irf_selection
            or context.irf_source_kind != source.model_context.irf_source_kind
        ):
            raise ValueError("Bayesian source summary conflicts with full model context")
        if context.prepared_irf is not None:
            if assumption[3] is None:
                raise ValueError("Bayesian context prepared IRF is missing from model assumption")
            _verify_prepared_irf(connection, assumption[3], context.prepared_irf)
        elif assumption[3] is not None:
            _verify_irf_source(connection, measurement[7], context.irf_source)
            kernel_hash = connection.execute(
                "SELECT kernel_sha256 FROM prepared_irfs WHERE prepared_irf_id = ?",
                (assumption[3],),
            ).fetchone()[0]
            if _hash_prepared_kernel(context.irf_kernel) != kernel_hash:
                raise ValueError("Bayesian attached kernel conflicts with assumed prepared IRF")
        _, context_bins, context_start, context_step, context_hash = _uniform_grid(
            context.time_ns,
        )
        if tuple(measurement[1:5]) != (
            context_bins, context_hash, context_start, context_step,
        ):
            raise ValueError("Bayesian context measurement grid differs from stored measurement")
        if _hash_raw_counts(context.counts) != measurement[5]:
            raise ValueError("Bayesian context counts differ from stored measurement")
        if context.measurement.sample_id != measurement[8] or any(
            not _same_payload_value(column, stored, _mapping_json(supplied, column))
            for column, stored, supplied in (
                ("metadata_json", measurement[9], context.measurement.metadata),
                ("provenance_json", measurement[10], context.measurement.provenance),
            )
        ):
            raise ValueError("Bayesian context measurement identity/provenance differs from stored measurement")
    return configuration, {"prior_policy_id": model[3], "assumption": assumption}, measurement


def _bayesian_artifact(
    connection: sqlite3.Connection, artifact_id: int | None, *, run_id: int,
    role: Literal["posterior", "predictive"],
) -> int | None:
    selected = _optional_integer(artifact_id, f"{role}_artifact_id", minimum=1)
    if selected is None:
        return None
    row = connection.execute(
        "SELECT artifact_kind, producing_run_id FROM artifacts WHERE artifact_id = ?",
        (selected,),
    ).fetchone()
    allowed = (
        {"posterior_chain", "posterior_samples"} if role == "posterior"
        else {"posterior_predictive_samples"}
    )
    if row is None or row[0] not in allowed or row[1] not in (None, run_id):
        raise ValueError(f"{role} artifact role or producing run conflicts with Bayesian result")
    return selected


def _bayesian_predictive_payload(
    predictive: Any | None, *, ppc_seed: int | str | None,
    nonfinite: dict[str, str],
) -> dict[str, Any]:
    from tcspc_toolkit.bayesian_mismatch_evaluation import MismatchPredictiveSummary
    from tcspc_toolkit.bayesian_predictive import BayesianPosteriorPredictiveResult

    if predictive is None:
        if ppc_seed is not None:
            raise ValueError("unused posterior-predictive seed must not be invented")
        return {
            "ppc_status": "not_requested", "ppc_n_draws": None,
            "ppc_runtime_seconds": None, "ppc_deviance_tail_probability": None,
            "ppc_discrepancies_json": "{}",
        }
    if isinstance(predictive, BayesianPosteriorPredictiveResult):
        if ppc_seed is not None and _decimal_seed(ppc_seed, "ppc_seed") != str(predictive.random_seed):
            raise ValueError("posterior-predictive seed conflicts with source result")
        ppc_seed = predictive.random_seed
        diagnostics = predictive.diagnostics
        named = [
            ("poisson_deviance", diagnostics.poisson_deviance),
            ("residual_rms", diagnostics.rms_signed_deviance_residual),
            ("maximum_absolute_residual", diagnostics.maximum_absolute_signed_deviance_residual),
            ("total_counts", diagnostics.total_counts),
            ("peak_counts", diagnostics.peak_counts),
            ("peak_time_ns", diagnostics.peak_time_ns),
        ] + [(window.name.removesuffix("_window_ns") + "_window_counts", window.total_counts)
             for window in diagnostics.windows]
        summaries = {}
        window_bounds = {
            window.name.removesuffix("_window_ns") + "_window_counts": [
                window.lower_ns, window.upper_ns,
            ] for window in diagnostics.windows
        }
        for name, discrepancy in named:
            summaries[name] = {
                "observed_mean": _result_number(
                    np.mean(discrepancy.observed), f"ppc.{name}.observed_mean", nonfinite,
                ),
                "replicated_median": _result_number(
                    np.median(discrepancy.replicated), f"ppc.{name}.replicated_median", nonfinite,
                ),
                "tail_probability": _bayesian_probability(
                    discrepancy.posterior_predictive_tail_probability,
                    f"ppc.{name}.tail_probability", nonfinite,
                ),
            }
        status, reason = "success", None
        n_draws = int(predictive.replicated_counts.shape[0])
        runtime = predictive.runtime_seconds
        interval_level = _finite_scalar(predictive.interval_level, "predictive interval_level")
        if not 0 < interval_level < 1:
            raise ValueError("predictive interval_level must lie between zero and one")
        selection = {
            "interval_level": interval_level,
            "selection_with_replacement": bool(_result_flag(
                predictive.selection_with_replacement, "selection_with_replacement",
            )),
            "retained_posterior_sample_count": _integer(
                predictive.retained_posterior_sample_count, "retained_posterior_sample_count",
                minimum=1,
            ),
            "window_bounds_ns": window_bounds,
        }
    elif isinstance(predictive, MismatchPredictiveSummary):
        status, reason = predictive.status, predictive.reason
        n_draws, runtime = predictive.n_draws, predictive.runtime_seconds
        summaries = {}
        selection = {}
        for name, observed, replicated, tail in predictive.discrepancy_summaries:
            key = _required_text(name, "PPC discrepancy name")
            if key in summaries:
                raise ValueError("duplicate PPC discrepancy name")
            summaries[key] = {
                "observed_mean": _result_number(observed, f"ppc.{key}.observed_mean", nonfinite),
                "replicated_median": _result_number(replicated, f"ppc.{key}.replicated_median", nonfinite),
                "tail_probability": _bayesian_probability(tail, f"ppc.{key}.tail_probability", nonfinite),
            }
    else:
        raise TypeError("posterior_predictive must be a current Bayesian PPC result")
    if status not in ("success", "not_available", "failed"):
        raise ValueError("unsupported posterior-predictive status")
    draws = _integer(n_draws, "ppc_n_draws")
    if status == "success" and draws == 0:
        raise ValueError("successful posterior prediction requires draws")
    deviance = summaries.get("poisson_deviance", {}).get("tail_probability")
    return {
        "ppc_status": status, "ppc_n_draws": draws,
        "ppc_runtime_seconds": _result_number(
            runtime, "ppc_runtime_seconds", nonfinite, nonnegative=True,
        ),
        "ppc_deviance_tail_probability": deviance,
        "ppc_discrepancies_json": _canonical_json({
            "discrepancies": summaries,
            "failure_reason": _optional_diagnostic_text(reason, "PPC failure_reason"),
            "random_seed_decimal": _decimal_seed(ppc_seed, "ppc_seed"),
            "selection": selection,
        }),
    }


def _bayesian_payloads(
    source: Any, *, run_id: int, measurement_id: int, model_id: int,
    assumption_id: int, analysis_key: str, configuration: dict[str, Any],
    relation: str, posterior_predictive: Any | None,
    ppc_seed: int | str | None, posterior_artifact_id: int | None,
    predictive_artifact_id: int | None, call_seconds: float | None,
    valid_retained_samples: int | None, execution: Mapping[str, Any] | None,
    sampling_seed: int | str, assumption: sqlite3.Row | tuple,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from tcspc_toolkit.bayesian_evaluation import Issue4Bayesian
    from tcspc_toolkit.bayesian_sampling import BayesianReconvolutionResult, BayesianSamplingStatus

    if not isinstance(source, (BayesianReconvolutionResult, Issue4Bayesian)):
        raise TypeError("Bayesian source must be a full sampler result or Issue4Bayesian")
    status = source.status
    if not isinstance(status, BayesianSamplingStatus):
        raise ValueError("Bayesian sampling status must use the source enum")
    compact = isinstance(source, Issue4Bayesian)
    diagnostic = source if compact else source.diagnostics
    accepted = _result_flag(
        source.diagnostics_accepted if compact else diagnostic.accepted,
        "diagnostics_accepted",
    )
    reasons = tuple(source.diagnostic_failure_reasons if compact else diagnostic.failure_reasons)
    if any(not isinstance(reason, str) for reason in reasons):
        raise ValueError("diagnostic failure reasons must be strings")
    if accepted and (status is not BayesianSamplingStatus.SUCCESS or reasons):
        raise ValueError("accepted diagnostics require successful sampling and no failures")
    nonfinite: dict[str, str] = {}
    credible_probability = _finite_scalar(
        configuration["sampler"].get("credible_interval_level"),
        "credible_interval_level",
    )
    if not 0.0 < credible_probability < 1.0:
        raise ValueError("credible_interval_level must lie between zero and one")
    if compact:
        raw_median = source.lifetime_median_ns
        median = _result_number(source.lifetime_median_ns, "lifetime_median_ns", nonfinite)
        mean = _result_number(source.lifetime_mean_ns, "lifetime_mean_ns", nonfinite)
        parameter_summaries = [{
            "name": "lifetime_ns", "unit": "ns", "mean": mean, "median": median,
            "standard_deviation": _result_number(
                source.posterior_lifetime_std_ns, "lifetime.standard_deviation", nonfinite,
                nonnegative=True,
            ),
            "credible_lower": _result_number(
                source.credible_lower_ns, "lifetime.credible_lower", nonfinite,
            ),
            "credible_upper": _result_number(
                source.credible_upper_ns, "lifetime.credible_upper", nonfinite,
            ),
        }]
        correlations = [
            {"first": "lifetime_ns", "second": "background_per_bin",
             "correlation": _bayesian_correlation(
                 source.lifetime_background_correlation,
                 "correlation.lifetime_background", nonfinite,
             )},
            {"first": "lifetime_ns", "second": "temporal_shift_ns",
             "correlation": _bayesian_correlation(
                 source.lifetime_shift_correlation, "correlation.lifetime_shift", nonfinite,
             )},
        ]
        diagnostic_arrays: dict[str, Any] = {}
        diagnostic_parameter_order = None
        runtime = source.inference_total_seconds
        actual_call_seconds = source.call_seconds
        source_type = "tcspc_toolkit.bayesian_evaluation.Issue4Bayesian"
    else:
        names = tuple(source.inferred_parameter_names)
        if len(set(names)) != len(names) or (
            source.parameter_summaries
            and (names != tuple(item.name for item in source.parameter_summaries)
                 or tuple(source.inferred_parameter_units) != tuple(
                     item.unit for item in source.parameter_summaries
                 ))
        ):
            raise ValueError("posterior parameter summaries conflict with inferred names")
        expected_names = {"amplitude", "lifetime_ns", "background_per_bin"}
        if source.priors.infer_temporal_shift:
            expected_names.add("temporal_shift_ns")
        if set(names) != expected_names:
            raise ValueError("Bayesian inferred parameters conflict with the prior shift mode")
        if not np.allclose(source.correlation_matrix, source.correlation_matrix.T,
                           rtol=1e-7, atol=1e-12, equal_nan=True):
            raise ValueError("Bayesian correlation matrix must be symmetric")
        diagnostic_parameter_order = names
        n_ensembles = configuration["sampler"]["n_ensembles"]
        n_walkers = configuration["sampler"]["n_walkers"]
        for field, shape in (
            ("acceptance_fraction", (n_ensembles, n_walkers)),
            ("autocorrelation_time_steps", (n_ensembles, len(names))),
            ("autocorrelation_relative_change", (n_ensembles, len(names))),
            ("approximate_effective_samples", (len(names),)),
        ):
            if np.shape(getattr(diagnostic, field)) != shape:
                raise ValueError(f"{field} shape conflicts with sampler/parameter axes")
        parameter_summaries = []
        for item in source.parameter_summaries:
            name = _required_text(item.name, "parameter name")
            parameter_summaries.append({
                "name": name, "unit": _required_text(item.unit, "parameter unit"),
                **{field: _result_number(
                    getattr(item, field), f"parameter.{name}.{field}", nonfinite,
                    nonnegative=(field == "standard_deviation"),
                ) for field in (
                    "mean", "median", "standard_deviation", "credible_lower", "credible_upper",
                )},
            })
        lifetime = next((item for item in parameter_summaries if item["name"] == "lifetime_ns"), None)
        raw_median = next((item.median for item in source.parameter_summaries
                           if item.name == "lifetime_ns"), None)
        median = None if lifetime is None else lifetime["median"]
        mean = None if lifetime is None else lifetime["mean"]
        correlations = []
        for left in range(len(names)):
            for right in range(left, len(names)):
                correlations.append({
                    "first": names[left], "second": names[right],
                    "correlation": _bayesian_correlation(
                        source.correlation_matrix[left, right],
                        f"correlation.{names[left]}.{names[right]}", nonfinite,
                    ),
                })
        diagnostic_arrays = {
            "acceptance_fraction_by_ensemble_walker": _bayesian_array_summary(
                diagnostic.acceptance_fraction, "acceptance_fraction", nonfinite,
            ),
            "autocorrelation_time_steps_by_ensemble_parameter": _bayesian_array_summary(
                diagnostic.autocorrelation_time_steps, "autocorrelation_time_steps", nonfinite,
            ),
            "autocorrelation_relative_change_by_ensemble_parameter": _bayesian_array_summary(
                diagnostic.autocorrelation_relative_change, "autocorrelation_relative_change", nonfinite,
            ),
            "approximate_effective_samples_by_parameter": _bayesian_array_summary(
                diagnostic.approximate_effective_samples, "approximate_effective_samples", nonfinite,
            ),
        }
        if any(value is not None and not 0.0 <= value <= 1.0
               for row in diagnostic_arrays["acceptance_fraction_by_ensemble_walker"]
               for value in row):
            raise ValueError("walker acceptance fractions must lie between zero and one")
        runtime = source.runtime.total_seconds
        actual_call_seconds = call_seconds
        source_type = "tcspc_toolkit.bayesian_sampling.BayesianReconvolutionResult"
    if accepted and (median is None or median <= 0):
        raise ValueError("accepted Bayesian diagnostics require a positive finite posterior median")
    if accepted and parameter_summaries:
        lifetime = next(item for item in parameter_summaries if item["name"] == "lifetime_ns")
        low, high = lifetime["credible_lower"], lifetime["credible_upper"]
        if low is None or high is None or low > high:
            raise ValueError("accepted Bayesian summary requires ordered finite credible bounds")
    estimator_status = (
        "available" if median is not None and status in (
            BayesianSamplingStatus.SUCCESS, BayesianSamplingStatus.INSUFFICIENT_SAMPLING,
        ) else "failed"
    )
    point = _estimator_payload(
        run_id=run_id, measurement_id=measurement_id, model_id=model_id,
        assumption_id=assumption_id, analysis_key=analysis_key,
        point_summary="posterior_median", status=estimator_status,
        is_valid=bool(accepted), source_result_type=source_type,
        irf_model_relation=relation, lifetime_estimate_ns=raw_median,
        failure_reason=source.failure_reason if not compact else None,
        runtime_seconds=runtime, runtime_scope="bayesian_inference_total",
        random_seed=sampling_seed, execution=execution,
    )
    if accepted:
        shift = next((item for item in parameter_summaries
                      if item["name"] == "temporal_shift_ns"), None)
        if shift is not None:
            for field in ("mean", "median", "credible_lower", "credible_upper"):
                if shift[field] is None:
                    raise ValueError("accepted Bayesian shift summary must be finite")
                _validate_assumed_temporal_shift(
                    shift[field], context_completeness=assumption[2], fixed_shift=assumption[4],
                    lower_shift=assumption[5], upper_shift=assumption[6],
                    label=f"posterior {field}",
                )
    if compact:
        minimum_ess = source.minimum_effective_samples
        minimum_multiples = source.minimum_autocorrelation_multiples
        maximum_change = source.maximum_autocorrelation_relative_change
    else:
        effective = np.asarray(diagnostic.approximate_effective_samples, dtype=float)
        finite_ess = effective[np.isfinite(effective)]
        minimum_ess = np.min(finite_ess) if finite_ess.size else math.nan
        tau = np.asarray(diagnostic.autocorrelation_time_steps, dtype=float)
        finite_tau = tau[np.isfinite(tau) & (tau > 0)]
        minimum_multiples = (
            diagnostic.production_steps / np.max(finite_tau) if finite_tau.size else math.nan
        )
        change = np.asarray(diagnostic.autocorrelation_relative_change, dtype=float)
        finite_change = change[np.isfinite(change)]
        maximum_change = np.max(finite_change) if finite_change.size else math.nan
    parameter_json = {
        "credible_probability": credible_probability,
        "parameters": parameter_summaries,
        "diagnostic_parameter_order": diagnostic_parameter_order,
        "sampling_counts": {
            "n_walkers": _integer(configuration["sampler"].get("n_walkers"), "n_walkers", minimum=1),
            "n_ensembles": _integer(configuration["sampler"].get("n_ensembles"), "n_ensembles", minimum=1),
            "requested_warmup_steps": _integer(
                configuration["sampler"].get("warmup_steps"), "warmup_steps", minimum=1,
            ),
            "actual_warmup_steps": None,
            "requested_production_steps": _integer(
                configuration["sampler"].get("production_steps"), "requested production_steps", minimum=1,
            ),
            "valid_retained_samples": _optional_integer(
                valid_retained_samples, "valid_retained_samples",
            ),
        },
        "diagnostic_arrays": diagnostic_arrays,
    }
    predictive_payload = _bayesian_predictive_payload(
        posterior_predictive, ppc_seed=ppc_seed, nonfinite=nonfinite,
    )
    summary = {
        "result_id": 0,
        "posterior_artifact_id": posterior_artifact_id,
        "predictive_artifact_id": predictive_artifact_id,
        "sampling_status": status.value,
        "diagnostics_accepted": accepted,
        "lifetime_mean_ns": mean,
        "mean_acceptance_fraction": _bayesian_probability(
            diagnostic.mean_acceptance_fraction, "mean_acceptance_fraction", nonfinite,
        ),
        "minimum_effective_samples": _result_number(
            minimum_ess,
            "minimum_effective_samples", nonfinite, nonnegative=True,
        ),
        "production_steps": _integer(diagnostic.production_steps, "production_steps"),
        "retained_samples": _integer(diagnostic.retained_samples, "retained_samples"),
        "extension_count": _integer(diagnostic.extension_count, "extension_count"),
        "minimum_autocorrelation_multiples": _result_number(
            minimum_multiples, "minimum_autocorrelation_multiples", nonfinite,
            nonnegative=True,
        ),
        "maximum_autocorrelation_relative_change": _result_number(
            maximum_change, "maximum_autocorrelation_relative_change", nonfinite,
            nonnegative=True,
        ),
        "maximum_ensemble_mean_difference_sd": _result_number(
            diagnostic.maximum_ensemble_mean_difference_sd,
            "maximum_ensemble_mean_difference_sd", nonfinite, nonnegative=True,
        ),
        "maximum_ensemble_median_difference_sd": _result_number(
            diagnostic.maximum_ensemble_median_difference_sd,
            "maximum_ensemble_median_difference_sd", nonfinite, nonnegative=True,
        ),
        "lifetime_background_correlation": _bayesian_correlation(
            source.lifetime_background_correlation if compact else
            source.correlation_matrix[source.inferred_parameter_names.index("lifetime_ns"),
                                      source.inferred_parameter_names.index("background_per_bin")],
            "lifetime_background_correlation", nonfinite,
        ),
        "lifetime_shift_correlation": _bayesian_correlation(
            source.lifetime_shift_correlation if compact else (
                source.correlation_matrix[source.inferred_parameter_names.index("lifetime_ns"),
                                          source.inferred_parameter_names.index("temporal_shift_ns")]
                if "temporal_shift_ns" in source.inferred_parameter_names else None
            ), "lifetime_shift_correlation", nonfinite,
        ),
        "initialization_seconds": _result_number(
            source.initialization_seconds if compact else source.runtime.initialization_seconds,
            "initialization_seconds", nonfinite, nonnegative=True,
        ),
        "sampling_seconds": _result_number(
            source.sampling_seconds if compact else source.runtime.sampling_seconds,
            "sampling_seconds", nonfinite, nonnegative=True,
        ),
        "diagnostic_seconds": _result_number(
            source.diagnostic_seconds if compact else source.runtime.diagnostic_seconds,
            "diagnostic_seconds", nonfinite, nonnegative=True,
        ),
        "inference_total_seconds": _result_number(
            runtime, "inference_total_seconds", nonfinite, nonnegative=True,
        ),
        "call_seconds": _result_number(
            actual_call_seconds, "call_seconds", nonfinite, nonnegative=True,
        ),
        "diagnostic_failure_reasons_json": _canonical_json(reasons),
        "parameter_summaries_json": _canonical_json(parameter_json),
        "correlation_json": _canonical_json(correlations),
        **predictive_payload,
        "nonfinite_fields_json": _canonical_json(nonfinite),
    }
    counts = parameter_json["sampling_counts"]
    if summary["retained_samples"] != (
        counts["n_walkers"] * counts["n_ensembles"] * summary["production_steps"]
    ):
        raise ValueError("retained samples conflict with production steps and ensemble/walker counts")
    if valid_retained_samples is not None and valid_retained_samples > summary["retained_samples"]:
        raise ValueError("finite retained samples exceed retained samples")
    return point, summary


def _record_bayesian_composed(
    connection: sqlite3.Connection, source: Any, *, run_id: int,
    measurement_id: int, model_id: int, assumption_id: int,
    analysis_key: str, relation: str, posterior_predictive: Any | None,
    ppc_seed: int | str | None, posterior_artifact_id: int | None,
    predictive_artifact_id: int | None, call_seconds: float | None,
    valid_retained_samples: int | None, execution: Mapping[str, Any] | None,
    context: Any | None, sampling_seed: int | str, on_duplicate: DuplicatePolicy,
) -> int:
    run = _integer(run_id, "run_id", minimum=1)
    measured = _integer(measurement_id, "measurement_id", minimum=1)
    model = _integer(model_id, "model_id", minimum=1)
    assumed = _integer(assumption_id, "assumption_id", minimum=1)
    if connection.execute(
        "SELECT 1 FROM run_measurements WHERE run_id = ? AND measurement_id = ?",
        (run, measured),
    ).fetchone() is None:
        raise ValueError("run_measurements membership is required for a result")
    model_config, model_context, _ = _bayesian_model_context(
        connection, model_id=model, assumption_id=assumed,
        measurement_id=measured, source=source, context=context, sampling_seed=sampling_seed,
    )
    from tcspc_toolkit.bayesian_predictive import BayesianPosteriorPredictiveResult

    if isinstance(posterior_predictive, BayesianPosteriorPredictiveResult):
        _bayesian_model_context(
            connection, model_id=model, assumption_id=assumed,
            measurement_id=measured, source=source,
            context=posterior_predictive.model_context, sampling_seed=sampling_seed,
        )
    posterior_artifact = _bayesian_artifact(
        connection, posterior_artifact_id, run_id=run, role="posterior",
    )
    predictive_artifact = _bayesian_artifact(
        connection, predictive_artifact_id, run_id=run, role="predictive",
    )
    if predictive_artifact is not None and posterior_predictive is None:
        raise ValueError("predictive artifact requires a posterior-predictive result")
    point, summary = _bayesian_payloads(
        source, run_id=run, measurement_id=measured, model_id=model,
        assumption_id=assumed, analysis_key=analysis_key,
        configuration=model_config, relation=relation,
        posterior_predictive=posterior_predictive, ppc_seed=ppc_seed,
        posterior_artifact_id=posterior_artifact,
        predictive_artifact_id=predictive_artifact, call_seconds=call_seconds,
        valid_retained_samples=valid_retained_samples, execution=execution,
        sampling_seed=sampling_seed, assumption=model_context["assumption"],
    )
    if predictive_artifact is not None and not summary["ppc_n_draws"]:
        raise ValueError("predictive-sample artifact requires retained predictive draws")
    with transaction(connection):
        result_id, inserted = _record_estimator_row(
            connection, point, on_duplicate=on_duplicate,
        )
        summary["result_id"] = result_id
        if inserted:
            _record_row(
                connection, table="bayesian_summaries", key_columns=("result_id",),
                payload=summary, return_column=None, on_duplicate="raise",
            )
        else:
            cursor = connection.execute(
                "SELECT * FROM bayesian_summaries WHERE result_id = ?", (result_id,),
            )
            found = cursor.fetchone()
            if found is None:
                _record_row(
                    connection, table="bayesian_summaries", key_columns=("result_id",),
                    payload=summary, return_column=None, on_duplicate="raise",
                )
            else:
                existing = dict(zip((item[0] for item in cursor.description), found))
                if not all(_same_payload_value(column, existing[column], value)
                           for column, value in summary.items()):
                    raise PersistenceConflictError("conflicting Bayesian summary for reused result")
        return result_id


def record_bayesian_result(
    connection: sqlite3.Connection, result: Any, *, run_id: int,
    measurement_id: int, model_id: int, assumption_id: int,
    analysis_key: str = "default", posterior_predictive: Any | None = None,
    posterior_artifact_id: int | None = None,
    predictive_artifact_id: int | None = None,
    call_seconds: float | None = None,
    execution: Mapping[str, Any] | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Record one completed Bayesian sampler result and optional PPC summary."""
    from tcspc_toolkit.bayesian_sampling import BayesianInferenceRun, BayesianReconvolutionResult

    if isinstance(result, BayesianInferenceRun):
        source, context = result.result, result.context
        samples = result.samples
        finite = np.isfinite(samples.log_probability) & np.all(
            np.isfinite(samples.physical), axis=-1,
        ) & np.all(np.isfinite(samples.transformed), axis=-1)
        valid_retained_samples = int(np.count_nonzero(finite))
        if source.diagnostics.retained_samples != samples.physical.shape[0] * \
                samples.physical.shape[1] * samples.physical.shape[2]:
            raise ValueError("retained-sample count conflicts with source sample shape")
    elif isinstance(result, BayesianReconvolutionResult):
        source, context, valid_retained_samples = result, None, None
    else:
        raise TypeError("result must be BayesianReconvolutionResult or BayesianInferenceRun")
    if posterior_predictive is not None:
        from tcspc_toolkit.bayesian_predictive import BayesianPosteriorPredictiveResult
        if not isinstance(posterior_predictive, BayesianPosteriorPredictiveResult):
            raise TypeError("posterior_predictive must be BayesianPosteriorPredictiveResult")
        if posterior_predictive.inference_sampling_status is not source.status or (
            _canonical_json(posterior_predictive.priors) != _canonical_json(source.priors)
            or _canonical_json(posterior_predictive.sampling_config)
            != _canonical_json(source.sampling_config)
            or posterior_predictive.retained_posterior_sample_count != source.diagnostics.retained_samples
        ):
            raise ValueError("posterior prediction conflicts with Bayesian sampler result")
    return _record_bayesian_composed(
        connection, source, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id, analysis_key=analysis_key,
        relation=source.model_context.irf_model_relation.value,
        posterior_predictive=posterior_predictive, ppc_seed=None,
        posterior_artifact_id=posterior_artifact_id,
        predictive_artifact_id=predictive_artifact_id,
        call_seconds=call_seconds, valid_retained_samples=valid_retained_samples,
        execution=execution, context=context, sampling_seed=source.sampling_config.random_seed,
        on_duplicate=on_duplicate,
    )


def record_issue4_bayesian_result(
    connection: sqlite3.Connection, record: Any, *, run_id: int,
    measurement_id: int, model_id: int, assumption_id: int,
    analysis_key: str = "default", pseudo_true_reference_id: int | None = None,
    irf_model_relation: str = "unspecified",
    posterior_artifact_id: int | None = None,
    predictive_artifact_id: int | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Record the compact matched or mismatch Issue-4 Bayesian realization."""
    from tcspc_toolkit.bayesian_evaluation import BayesianClassicalRealizationResult
    from tcspc_toolkit.bayesian_mismatch_evaluation import MismatchInferenceRecord

    if not isinstance(record, (BayesianClassicalRealizationResult, MismatchInferenceRecord)):
        raise TypeError("record must be an Issue-4 Bayesian evaluation realization")
    mismatch = isinstance(record, MismatchInferenceRecord)
    if mismatch != (pseudo_true_reference_id is not None):
        raise ValueError("mismatch records require a selected pseudo-true reference")
    measurement = connection.execute(
        "SELECT m.condition_pk, m.values_sha256, c.condition_id, c.generating_model, "
        "c.mono_lifetime_ns, c.primary_lifetime_ns, c.secondary_lifetime_ns, "
        "c.secondary_detected_fraction, c.signal_photon_count, c.background_per_bin, "
        "c.true_temporal_shift_ns, m.observed_total_counts, m.observation_seed_decimal "
        "FROM measurements AS m LEFT JOIN simulation_conditions AS c "
        "ON c.condition_pk = m.condition_pk WHERE m.measurement_id = ?",
        (_integer(measurement_id, "measurement_id", minimum=1),),
    ).fetchone()
    if measurement is None or measurement[0] is None or measurement[2] != record.condition_id \
            or measurement[1] != record.observed_counts_sha256:
        raise ValueError("Issue-4 record conflicts with generating condition or observation")
    if measurement[8] != record.signal_photon_count or measurement[11] != record.observed_total_counts:
        raise ValueError("Issue-4 photon counts conflict with stored condition/measurement")
    physical_values = [(measurement[9], record.background_per_bin),
                       (measurement[10], record.true_temporal_shift_ns)]
    if mismatch:
        physical_values.extend([
            (measurement[4], record.mono_generating_lifetime_ns),
            (measurement[5] if record.generating_model == "biexponential" else measurement[4],
             record.primary_lifetime_ns),
            (measurement[6], record.secondary_lifetime_ns),
            (measurement[7], record.secondary_detected_fraction),
        ])
    else:
        if measurement[3] != "monoexponential":
            raise ValueError("matched Issue-4 record requires monoexponential generation")
        physical_values.append((measurement[4], record.true_lifetime_ns))
    for stored, supplied in physical_values:
        if (stored is None) != (supplied is None) or stored is not None and not math.isclose(
            stored, _finite_scalar(supplied, "Issue-4 generating parameter"),
            rel_tol=1e-7, abs_tol=1e-12,
        ):
            raise ValueError("Issue-4 generating parameters conflict with stored condition")
    if measurement[12] is not None and measurement[12] != _decimal_seed(
        record.seeds.observation, "observation seed",
    ):
        raise ValueError("Issue-4 observation seed conflicts with stored measurement")
    run = connection.execute("SELECT profile FROM experiment_runs WHERE run_id = ?", (run_id,)).fetchone()
    if run is not None and run[0] is not None and run[0] != record.profile:
        raise ValueError("Issue-4 source profile conflicts with stored run")
    assumption = connection.execute(
        "SELECT source_assumption_id FROM model_assumptions WHERE assumption_id = ?",
        (_integer(assumption_id, "assumption_id", minimum=1),),
    ).fetchone()
    if mismatch:
        if assumption is None or assumption[0] != record.assumption_id \
                or measurement[3] != record.generating_model \
                or record.assumed_decay_model != "monoexponential":
            raise ValueError("mismatch source context conflicts with stored assumption or condition")
        reference = connection.execute(
            "SELECT reference_kind, condition_pk, assumption_id FROM lifetime_references "
            "WHERE reference_id = ?",
            (_integer(pseudo_true_reference_id, "pseudo_true_reference_id", minimum=1),),
        ).fetchone()
        if reference is None or tuple(reference) != (
            "pseudo_true_mono", measurement[0], assumption_id,
        ):
            raise ValueError("pseudo-true reference conflicts with mismatch condition/assumption")
    model = connection.execute(
        "SELECT prior_policy_id FROM model_versions WHERE model_id = ?", (model_id,),
    ).fetchone()
    if not mismatch and (model is None or model[0] != record.prior_policy_id):
        raise ValueError("matched Issue-4 prior policy conflicts with model")
    if mismatch and irf_model_relation == "unspecified":
        raise ValueError("mismatch IRF relation must be caller-declared")
    if not mismatch and irf_model_relation != "matched":
        raise ValueError("matched Issue-4 evaluation requires declared matched IRF relation")
    execution = {
        "source_profile": record.profile,
        "realization_index": record.realization_index,
    }
    if mismatch:
        # Frozen Issue-4 f8 hashes retain signed zero, unlike schema-v1 canonical
        # float hashes. Preserve their role; never compare the two conventions.
        execution["source_issue4_float64_hashes"] = {
            field: _sha256(getattr(record, field), field) for field in (
                "expected_counts_sha256", "time_grid_sha256",
                "generating_irf_sha256", "assumed_irf_sha256",
            )
        }
        execution["mismatch_mechanism"] = _required_text(record.mechanism, "mismatch mechanism")
        execution["source_generating_irf_id"] = _required_text(record.generating_irf_id, "generating IRF label")
        execution["source_assumed_irf_id"] = _required_text(record.assumed_irf_id, "assumed IRF label")
    predictive = record.posterior_predictive if mismatch else None
    ppc_seed = record.seeds.posterior_predictive if mismatch and predictive.status != "not_available" else None
    return _record_bayesian_composed(
        connection, record.bayesian, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id, analysis_key=analysis_key,
        relation=irf_model_relation, posterior_predictive=predictive,
        ppc_seed=ppc_seed, posterior_artifact_id=posterior_artifact_id,
        predictive_artifact_id=predictive_artifact_id,
        call_seconds=None, valid_retained_samples=None, execution=execution,
        context=None, sampling_seed=record.seeds.bayesian, on_duplicate=on_duplicate,
    )


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


# Stage-4 per-observation uncertainty. The scientific result types remain in
# their numerical modules; this boundary records their existing outputs.
def _ordered_result_ids(result_ids: Sequence[int], *, size: int) -> tuple[int, ...]:
    if isinstance(result_ids, np.ndarray):
        if result_ids.ndim != 1:
            raise ValueError("result_ids must be a one-dimensional ordered sequence")
        supplied = tuple(result_ids)
    elif isinstance(result_ids, Sequence) and not isinstance(result_ids, (str, bytes)):
        supplied = tuple(result_ids)
    else:
        raise ValueError("result_ids must be an ordered sequence")
    if len(supplied) != size:
        raise ValueError("result_ids length must match uncertainty output length")
    ids = tuple(_integer(item, "result_id", minimum=1) for item in supplied)
    if len(set(ids)) != len(ids):
        raise ValueError("result_ids must not repeat within one batch")
    return ids


def _require_point_for_uncertainty(
    connection: sqlite3.Connection, result_id: int, *, family: str | None = None,
    prediction: Any = None,
) -> None:
    row = connection.execute(
        "SELECT r.lifetime_estimate_ns, m.family, r.nonfinite_fields_json "
        "FROM estimator_results AS r "
        "JOIN model_versions AS m ON m.model_id = r.model_id WHERE r.result_id = ?",
        (result_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown estimator_results.result_id: {result_id}")
    if family is not None and row[1] != family:
        raise ValueError(f"uncertainty source requires a {family} point result")
    if prediction is not None:
        nonfinite: dict[str, str] = {}
        source_prediction = _result_number(prediction, "prediction", nonfinite)
        stored_nonfinite = json.loads(row[2]).get("lifetime_estimate_ns")
        if row[0] != source_prediction or stored_nonfinite != nonfinite.get("prediction"):
            raise ValueError("uncertainty prediction differs from its point result")


def _require_poisson_classical_point(
    connection: sqlite3.Connection, result_id: int,
) -> None:
    row = connection.execute(
        "SELECT model.configuration_json, measurement.data_kind "
        "FROM estimator_results AS point "
        "JOIN model_versions AS model ON model.model_id = point.model_id "
        "JOIN measurements AS measurement ON measurement.measurement_id = point.measurement_id "
        "WHERE point.result_id = ? AND model.family = 'classical'",
        (result_id,),
    ).fetchone()
    if row is None or json.loads(row[0]).get("objective") != "poisson" or (
        row[1] != "raw_counts"
    ):
        raise ValueError("Poisson classical uncertainty requires a raw-count Poisson fit")


def _uncertainty_payload(
    *, result_id: int, method_id: str, output_kind: str,
    method_configuration: Mapping[str, Any], interpretation: str,
    calibration_scope: str, is_valid: bool,
    interval_kind: str | None = None, nominal_coverage: Any = None,
    lower_ns: Any = None, upper_ns: Any = None,
    uncertainty_score: Any = None, reported_std_ns: Any = None,
    resample_median_ns: Any = None, n_requested: int | None = None,
    n_valid: int | None = None, refit_failure_rate: Any = None,
    runtime_seconds: Any = None, random_seed: int | str | None = None,
    calibration_run_id: int | None = None, samples_artifact_id: int | None = None,
    failure_reason: str | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    extra_nonfinite: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if output_kind not in ("prediction_interval", "uncertainty_score", "covariance_summary"):
        raise ValueError("unsupported Stage-4 uncertainty output_kind")
    config_json, config_hash = _configuration_json(method_configuration)
    nonfinite = dict(extra_nonfinite or {})
    lower = _result_number(lower_ns, "lower_ns", nonfinite)
    upper = _result_number(upper_ns, "upper_ns", nonfinite)
    score = _result_number(uncertainty_score, "uncertainty_score", nonfinite)
    std = _result_number(reported_std_ns, "reported_std_ns", nonfinite)
    median = _result_number(resample_median_ns, "resample_median_ns", nonfinite)
    failure_rate = _result_number(refit_failure_rate, "refit_failure_rate", nonfinite)
    runtime = _result_number(runtime_seconds, "runtime_seconds", nonfinite)
    valid = _result_flag(is_valid, "is_valid")
    if (valid and std is not None and std < 0) or (score is not None and score < 0):
        raise ValueError("valid uncertainty spread and source scores must be nonnegative")
    if failure_rate is not None and not 0 <= failure_rate <= 1:
        raise ValueError("refit_failure_rate must lie in [0, 1]")
    if runtime is not None and runtime < 0:
        raise ValueError("runtime_seconds must be nonnegative")
    requested = _optional_integer(n_requested, "n_requested")
    valid_count = _optional_integer(n_valid, "n_valid")
    if requested is not None and valid_count is not None and valid_count > requested:
        raise ValueError("n_valid cannot exceed n_requested")
    if output_kind == "prediction_interval":
        kind = _required_text(interval_kind, "interval_kind")
        coverage = _finite_scalar(nominal_coverage, "nominal_coverage")
        if not 0 < coverage < 1:
            raise ValueError("nominal_coverage must lie strictly between 0 and 1")
        if score is not None or "uncertainty_score" in nonfinite:
            raise ValueError("interval uncertainty cannot contain a score")
        if valid and (lower is None or upper is None or lower > upper):
            raise ValueError("valid interval requires finite ordered bounds")
    elif output_kind == "uncertainty_score":
        kind = coverage = None
        if interval_kind is not None or nominal_coverage is not None or lower_ns is not None or upper_ns is not None:
            raise ValueError("score-only uncertainty cannot contain interval fields")
        if valid and score is None:
            raise ValueError("valid uncertainty score must be finite")
    else:
        kind = coverage = None
        if any(item is not None for item in (
            interval_kind, nominal_coverage, lower_ns, upper_ns, uncertainty_score,
        )):
            raise ValueError("covariance summary cannot contain interval or score fields")
        if valid and std is None:
            raise ValueError("valid covariance summary requires a finite lifetime std")
    return {
        "result_id": _integer(result_id, "result_id", minimum=1),
        "calibration_run_id": _optional_integer(calibration_run_id, "calibration_run_id", minimum=1),
        "samples_artifact_id": _optional_integer(samples_artifact_id, "samples_artifact_id", minimum=1),
        "method_id": _required_text(method_id, "method_id"),
        "output_kind": output_kind, "interval_kind": kind,
        "nominal_coverage": coverage,
        "method_config_json": config_json, "method_config_sha256": config_hash,
        "interpretation": _required_text(interpretation, "interpretation"),
        "calibration_scope": _required_text(calibration_scope, "calibration_scope"),
        "is_valid": valid, "lower_ns": lower, "upper_ns": upper,
        "uncertainty_score": score, "reported_std_ns": std,
        "resample_median_ns": median, "n_requested": requested,
        "n_valid": valid_count, "refit_failure_rate": failure_rate,
        "runtime_seconds": runtime,
        "random_seed_decimal": _decimal_seed(random_seed, "random_seed"),
        "failure_reason": _optional_diagnostic_text(failure_reason, "failure_reason"),
        "diagnostics_json": _mapping_json(diagnostics, "diagnostics"),
        "nonfinite_fields_json": _canonical_json(nonfinite),
    }


def _record_uncertainty_rows(
    connection: sqlite3.Connection, payloads: Sequence[dict[str, Any]], *,
    on_duplicate: DuplicatePolicy,
) -> tuple[int, ...]:
    if on_duplicate not in ("raise", "reuse_identical"):
        raise ValueError("on_duplicate must be 'raise' or 'reuse_identical'")
    if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise ValueError("foreign keys must be enabled before recording")
    expected = {row[1] for row in connection.execute("PRAGMA table_info(uncertainty_results)")}
    if any(set(payload) != expected - {"uncertainty_id"} for payload in payloads):
        raise RuntimeError("incomplete uncertainty_results adapter payload")
    ids: list[int] = []
    with transaction(connection):
        for payload in payloads:
            _require_point_for_uncertainty(connection, payload["result_id"])
            for table, column, value in (
                ("experiment_runs", "run_id", payload["calibration_run_id"]),
                ("artifacts", "artifact_id", payload["samples_artifact_id"]),
            ):
                if value is not None:
                    _require_row(connection, table, column, value)
            cursor = connection.execute(
                "SELECT * FROM uncertainty_results WHERE result_id = ? AND method_id = ? "
                "AND output_kind = ? AND method_config_sha256 = ? "
                "AND nominal_coverage IS ?",
                (payload["result_id"], payload["method_id"], payload["output_kind"],
                 payload["method_config_sha256"], payload["nominal_coverage"]),
            )
            found = cursor.fetchone()
            if found is not None:
                existing = dict(zip((item[0] for item in cursor.description), found))
                if on_duplicate == "raise":
                    raise PersistenceConflictError("duplicate uncertainty_results identity")
                if not all(_same_payload_value(column, existing[column], value)
                           for column, value in payload.items()):
                    raise PersistenceConflictError("conflicting uncertainty_results identity")
                ids.append(int(existing["uncertainty_id"]))
            else:
                names = tuple(payload)
                inserted = connection.execute(
                    f"INSERT INTO uncertainty_results ({', '.join(names)}) "
                    f"VALUES ({', '.join('?' for _ in names)})",
                    tuple(payload.values()),
                )
                ids.append(int(inserted.lastrowid))
    return tuple(ids)


def record_prediction_intervals(
    connection: sqlite3.Connection, intervals: Any, *,
    result_ids: Sequence[int], method: Any,
    method_configuration: Mapping[str, Any], interval_kind: str,
    calibration_run_id: int | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> tuple[int, ...]:
    """Attach ordered source prediction intervals to existing point results."""
    from tcspc_toolkit.uncertainty_evaluation import (
        PredictionIntervalResult, UncertaintyMethodDefinition,
        UncertaintyOutputKind,
    )
    from tcspc_toolkit.uncertainty_robustness import CONFORMALIZED_QUANTILE_METHOD_ID
    from tcspc_toolkit.ml_uncertainty import ML_UNCERTAINTY_METHODS
    from tcspc_toolkit.classical_uncertainty import CLASSICAL_UNCERTAINTY_METHODS

    if not isinstance(intervals, PredictionIntervalResult):
        raise TypeError("intervals must be a PredictionIntervalResult")
    if not isinstance(method, UncertaintyMethodDefinition):
        raise TypeError("method must be an UncertaintyMethodDefinition")
    if method.method_id != intervals.method_id or method.output_kind != UncertaintyOutputKind.PREDICTION_INTERVAL:
        raise ValueError("interval method definition conflicts with source intervals")
    if method.method_id in ML_UNCERTAINTY_METHODS:
        family = "ml"
        if method != ML_UNCERTAINTY_METHODS[method.method_id]:
            raise ValueError("interval method definition differs from source registry")
    elif method.method_id in CLASSICAL_UNCERTAINTY_METHODS:
        family = "classical"
        if method != CLASSICAL_UNCERTAINTY_METHODS[method.method_id]:
            raise ValueError("interval method definition differs from source registry")
    elif method.method_id == CONFORMALIZED_QUANTILE_METHOD_ID:
        family = "ml"
    else:
        raise ValueError("interval method is not a current repository method")
    calibrated = method.method_id in ("split_conformal", CONFORMALIZED_QUANTILE_METHOD_ID)
    if calibrated != (calibration_run_id is not None):
        raise ValueError("conformal intervals require a calibration run; other intervals do not")
    if not isinstance(method_configuration, Mapping):
        raise TypeError("method_configuration must be a mapping")
    config = dict(method_configuration)
    for key, value in (("nominal_coverage", float(intervals.nominal_coverage)),
                       ("interval_kind", _required_text(interval_kind, "interval_kind"))):
        if key in config and config[key] != value:
            raise ValueError(f"{key} conflicts with source interval")
        config[key] = value
    if calibrated:
        calibrated_run = _integer(calibration_run_id, "calibration_run_id", minimum=1)
        if "calibration_run_id" in config and config["calibration_run_id"] != calibrated_run:
            raise ValueError("calibration_run_id conflicts with method configuration")
        config["calibration_run_id"] = calibrated_run
    ids = _ordered_result_ids(result_ids, size=len(intervals.prediction))
    payloads = []
    for index, result_id in enumerate(ids):
        _require_point_for_uncertainty(
            connection, result_id, family=family,
            prediction=intervals.prediction[index],
        )
        if family == "classical":
            _require_poisson_classical_point(connection, result_id)
        lower = intervals.lower[index]
        central = intervals.prediction[index]
        upper = intervals.upper[index]
        sample_diagnostics = dict(diagnostics or {})
        triplet_crossed = False
        if method.method_id == "quantile_gradient_boosting":
            triplet_crossed = bool(
                np.isfinite(lower) and np.isfinite(central) and np.isfinite(upper)
                and (lower > central or central > upper)
            )
            sample_diagnostics["quantile_triplet_crossed"] = triplet_crossed
        payloads.append(_uncertainty_payload(
            result_id=result_id, method_id=method.method_id,
            output_kind="prediction_interval", method_configuration=config,
            interpretation=method.interpretation, calibration_scope=method.calibration_scope,
            is_valid=bool(intervals.valid_interval_mask[index]),
            interval_kind=interval_kind, nominal_coverage=intervals.nominal_coverage,
            lower_ns=lower, upper_ns=upper, calibration_run_id=calibration_run_id,
            diagnostics=sample_diagnostics,
        ))
    return _record_uncertainty_rows(connection, payloads, on_duplicate=on_duplicate)


def record_quantile_intervals(
    connection: sqlite3.Connection, intervals: Any, *,
    result_ids: Sequence[int], on_duplicate: DuplicatePolicy = "raise",
) -> tuple[int, ...]:
    """Record the current q05/q50/q95 interval without creating a median point."""
    from tcspc_toolkit.ml_uncertainty import (
        DEFAULT_LOWER_QUANTILE, DEFAULT_MEDIAN_QUANTILE, DEFAULT_UPPER_QUANTILE,
        ML_UNCERTAINTY_METHODS, QUANTILE_GRADIENT_BOOSTING_METHOD_ID,
    )
    if not math.isclose(
        intervals.nominal_coverage,
        DEFAULT_UPPER_QUANTILE - DEFAULT_LOWER_QUANTILE,
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError("quantile interval coverage differs from q05/q95 definition")
    return record_prediction_intervals(
        connection, intervals, result_ids=result_ids,
        method=ML_UNCERTAINTY_METHODS[QUANTILE_GRADIENT_BOOSTING_METHOD_ID],
        method_configuration={
            "lower_quantile": DEFAULT_LOWER_QUANTILE,
            "median_quantile": DEFAULT_MEDIAN_QUANTILE,
            "upper_quantile": DEFAULT_UPPER_QUANTILE,
            "interval_rule": "unrepaired_quantile_predictions",
        },
        interval_kind="quantile", on_duplicate=on_duplicate,
    )


def record_conformal_intervals(
    connection: sqlite3.Connection, intervals: Any, *,
    result_ids: Sequence[int], calibration_run_id: int,
    correction_ns: float, n_calibration_scores: int,
    on_duplicate: DuplicatePolicy = "raise",
) -> tuple[int, ...]:
    """Record split-conformalized quantile intervals and calibration identity."""
    from tcspc_toolkit.ml_uncertainty import ML_UNCERTAINTY_METHODS
    from tcspc_toolkit.uncertainty_evaluation import UncertaintyMethodDefinition, UncertaintyOutputKind
    from tcspc_toolkit.uncertainty_robustness import CONFORMALIZED_QUANTILE_METHOD_ID

    correction = _finite_scalar(correction_ns, "correction_ns", nonnegative=True)
    n_scores = _integer(n_calibration_scores, "n_calibration_scores", minimum=1)
    calibrated_run = _integer(calibration_run_id, "calibration_run_id", minimum=1)
    base = ML_UNCERTAINTY_METHODS["split_conformal"]
    method = UncertaintyMethodDefinition(
        method_id=CONFORMALIZED_QUANTILE_METHOD_ID,
        output_kind=UncertaintyOutputKind.PREDICTION_INTERVAL,
        interpretation=base.interpretation, calibration_scope=base.calibration_scope,
    )
    return record_prediction_intervals(
        connection, intervals, result_ids=result_ids, method=method,
        method_configuration={
            "base_method": "quantile_gradient_boosting",
            "calibration_policy": "finite_sample_split_conformal",
            "calibration_run_id": calibrated_run,
            "correction_ns": correction,
            "n_calibration_scores": n_scores,
        },
        interval_kind="conformalized_quantile", calibration_run_id=calibrated_run,
        on_duplicate=on_duplicate,
    )


def record_uncertainty_scores(
    connection: sqlite3.Connection, scores: Any, *,
    result_ids: Sequence[int], method: Any,
    method_configuration: Mapping[str, Any], n_members: int | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> tuple[int, ...]:
    """Attach score-only ensemble or bootstrap spread without interval claims."""
    from tcspc_toolkit.uncertainty_evaluation import (
        UncertaintyMethodDefinition, UncertaintyOutputKind, UncertaintyScoreResult,
    )
    from tcspc_toolkit.ml_uncertainty import (
        ML_TRAINING_BOOTSTRAP_METHOD_ID, ML_UNCERTAINTY_METHODS,
        RANDOM_FOREST_TREE_SPREAD_METHOD_ID,
    )

    if not isinstance(scores, UncertaintyScoreResult):
        raise TypeError("scores must be an UncertaintyScoreResult")
    if not isinstance(method, UncertaintyMethodDefinition):
        raise TypeError("method must be an UncertaintyMethodDefinition")
    if method.method_id != scores.method_id or method.output_kind != UncertaintyOutputKind.UNCERTAINTY_SCORE:
        raise ValueError("score method definition conflicts with source scores")
    if method.method_id not in ML_UNCERTAINTY_METHODS or method != ML_UNCERTAINTY_METHODS[method.method_id]:
        raise ValueError("score method is not a current repository method")
    if not isinstance(method_configuration, Mapping):
        raise TypeError("method_configuration must be a mapping")
    config = dict(method_configuration)
    spread_definition, ddof = {
        RANDOM_FOREST_TREE_SPREAD_METHOD_ID: ("population_std", 0),
        ML_TRAINING_BOOTSTRAP_METHOD_ID: ("sample_std", 1),
    }[method.method_id]
    for key, canonical in (("spread", spread_definition), ("ddof", ddof)):
        if key in config and config[key] != canonical:
            raise ValueError(f"{key} conflicts with the source spread definition")
        config[key] = canonical
    if n_members is not None:
        members = _integer(n_members, "n_members", minimum=1)
        if "n_members" in config and config["n_members"] != members:
            raise ValueError("n_members conflicts with method configuration")
        config["n_members"] = members
    ids = _ordered_result_ids(result_ids, size=len(scores.prediction))
    payloads = []
    for index, result_id in enumerate(ids):
        _require_point_for_uncertainty(
            connection, result_id, family="ml", prediction=scores.prediction[index],
        )
        payloads.append(_uncertainty_payload(
            result_id=result_id, method_id=method.method_id,
            output_kind="uncertainty_score", method_configuration=config,
            interpretation=method.interpretation, calibration_scope=method.calibration_scope,
            is_valid=bool(scores.valid_score_mask[index]),
            uncertainty_score=scores.uncertainty_score[index],
        ))
    return _record_uncertainty_rows(connection, payloads, on_duplicate=on_duplicate)


def record_classical_covariance_uncertainty(
    connection: sqlite3.Connection, covariance: Any, *,
    result_id: int, method_configuration: Mapping[str, Any],
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Store a per-fit Poisson/Fisher local summary, never empirical variability."""
    from tcspc_toolkit.classical_uncertainty import (
        CLASSICAL_UNCERTAINTY_METHODS, POISSON_LOCAL_COVARIANCE_METHOD_ID,
        PoissonLocalCovarianceResult,
    )
    if not isinstance(covariance, PoissonLocalCovarianceResult):
        raise TypeError("covariance must be a PoissonLocalCovarianceResult")
    point_id = _integer(result_id, "result_id", minimum=1)
    _require_point_for_uncertainty(connection, point_id, family="classical")
    _require_poisson_classical_point(connection, point_id)
    matrix = np.asarray(covariance.covariance_matrix)
    if matrix.shape != (4, 4):
        raise ValueError("covariance_matrix must have shape (4, 4)")
    nonfinite: dict[str, str] = {}
    matrix_values = [
        [_result_number(value, f"covariance_matrix[{i}][{j}]", nonfinite)
         for j, value in enumerate(row)]
        for i, row in enumerate(matrix)
    ]
    diagnostics = {
        "covariance_matrix": matrix_values,
        "amplitude_std": _result_number(covariance.amplitude_std, "amplitude_std", nonfinite),
        "background_std": _result_number(covariance.background_std, "background_std", nonfinite),
        "temporal_shift_std": _result_number(
            covariance.temporal_shift_std, "temporal_shift_std", nonfinite,
        ),
        "condition_number": _result_number(covariance.condition_number, "condition_number", nonfinite),
        "information_rank": _integer(covariance.information_rank, "information_rank"),
        "boundary_hit": bool(_result_flag(covariance.boundary_hit, "boundary_hit")),
    }
    definition = CLASSICAL_UNCERTAINTY_METHODS[POISSON_LOCAL_COVARIANCE_METHOD_ID]
    payload = _uncertainty_payload(
        result_id=point_id, method_id=definition.method_id,
        output_kind="covariance_summary", method_configuration=method_configuration,
        interpretation=definition.interpretation,
        calibration_scope=definition.calibration_scope,
        is_valid=covariance.covariance_valid, reported_std_ns=covariance.lifetime_std,
        failure_reason=covariance.failure_reason, diagnostics=diagnostics,
        extra_nonfinite=nonfinite,
    )
    return _record_uncertainty_rows(connection, (payload,), on_duplicate=on_duplicate)[0]


def record_parametric_bootstrap_uncertainty(
    connection: sqlite3.Connection, bootstrap: Any, *,
    result_id: int, method_configuration: Mapping[str, Any],
    bootstrap_seed: int | str | None = None,
    samples_artifact_id: int | None = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> int:
    """Store one Poisson refit-bootstrap interval and std in the same row."""
    from tcspc_toolkit.classical_uncertainty import (
        CLASSICAL_UNCERTAINTY_METHODS, PARAMETRIC_POISSON_BOOTSTRAP_METHOD_ID,
        ParametricPoissonBootstrapResult,
    )
    if not isinstance(bootstrap, ParametricPoissonBootstrapResult):
        raise TypeError("bootstrap must be a ParametricPoissonBootstrapResult")
    point_id = _integer(result_id, "result_id", minimum=1)
    _require_point_for_uncertainty(
        connection, point_id, family="classical", prediction=bootstrap.source_lifetime_ns,
    )
    _require_poisson_classical_point(connection, point_id)
    requested = _integer(bootstrap.n_resamples, "n_resamples", minimum=1)
    successful = _integer(bootstrap.n_successful_fits, "n_successful_fits")
    failed = _integer(bootstrap.n_failed_fits, "n_failed_fits")
    boundary_hits = _integer(bootstrap.n_boundary_hits, "n_boundary_hits")
    if successful + failed != requested or boundary_hits > successful:
        raise ValueError("bootstrap replicate counts are inconsistent")
    if np.asarray(bootstrap.lifetime_samples_ns).shape != (requested,):
        raise ValueError("bootstrap samples length differs from n_resamples")
    if not isinstance(method_configuration, Mapping):
        raise TypeError("method_configuration must be a mapping")
    config = dict(method_configuration)
    for key, value in (
        ("n_resamples", requested),
        ("nominal_coverage", float(bootstrap.nominal_coverage)),
        ("interval_rule", "central_percentile"),
    ):
        if key in config and config[key] != value:
            raise ValueError(f"{key} conflicts with bootstrap method configuration")
        config[key] = value
    nonfinite: dict[str, str] = {}
    diagnostics = {
        "n_failed_fits": failed, "n_boundary_hits": boundary_hits,
        "boundary_hit_rate": _result_number(
            bootstrap.boundary_hit_rate, "boundary_hit_rate", nonfinite,
        ),
    }
    definition = CLASSICAL_UNCERTAINTY_METHODS[PARAMETRIC_POISSON_BOOTSTRAP_METHOD_ID]
    payload = _uncertainty_payload(
        result_id=point_id, method_id=definition.method_id,
        output_kind="prediction_interval", method_configuration=config,
        interpretation=definition.interpretation,
        calibration_scope=definition.calibration_scope,
        is_valid=bootstrap.bootstrap_valid, interval_kind="bootstrap_percentile",
        nominal_coverage=bootstrap.nominal_coverage,
        lower_ns=bootstrap.lower_ns, upper_ns=bootstrap.upper_ns,
        reported_std_ns=bootstrap.bootstrap_std_ns,
        resample_median_ns=bootstrap.bootstrap_median_ns,
        n_requested=requested, n_valid=successful,
        refit_failure_rate=bootstrap.fit_failure_rate,
        random_seed=bootstrap_seed, samples_artifact_id=samples_artifact_id,
        failure_reason=bootstrap.failure_reason, diagnostics=diagnostics,
        extra_nonfinite=nonfinite,
    )
    return _record_uncertainty_rows(connection, (payload,), on_duplicate=on_duplicate)[0]


@dataclass(frozen=True)
class BenchmarkMetricScope:
    """Caller-selected aggregate population and scientific comparison context.

    ``population_key`` identifies the exact selection, not just a regime label.
    Supply stored measurement IDs when available; external datasets retain their
    explicit dataset and selection identities without inventing measurement rows.
    """

    run_id: int
    population_key: str
    dataset_key: str
    reference_semantics: str
    model_id: int | None = None
    assumption_id: int | None = None
    condition_pk: int | None = None
    reference_id: int | None = None
    reference_ids: Sequence[int] | None = None
    source_artifact_id: int | None = None
    test_id: str | None = None
    regime_id: str | None = None
    method_id: str | None = None
    method_configuration: Mapping[str, Any] | None = None
    nominal_coverage: float | None = None
    reference_kind: str | None = None
    reference_version: str | None = None
    signal_photon_count: int | None = None
    background_per_bin: float | None = None
    measurement_ids: Sequence[int] | None = None
    selection: Mapping[str, Any] | None = None


def _metric_scope(
    connection: sqlite3.Connection, scope: BenchmarkMetricScope, *,
    n_attempted: int,
) -> dict[str, Any]:
    if not isinstance(scope, BenchmarkMetricScope):
        raise TypeError("scope must be a BenchmarkMetricScope")
    run_id = _integer(scope.run_id, "run_id", minimum=1)
    run = connection.execute(
        "SELECT protocol_id, protocol_version, profile FROM experiment_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if run is None:
        raise ValueError("scope run_id must identify an experiment run")
    population_key = _required_text(scope.population_key, "population_key")
    dataset_key = _required_text(scope.dataset_key, "dataset_key")
    reference_semantics = _required_text(scope.reference_semantics, "reference_semantics")
    selection = json.loads(_mapping_json(scope.selection, "selection"))
    ids: tuple[int, ...] | None = None
    selected_conditions: set[int] = set()
    has_measurement_without_condition = False
    if scope.measurement_ids is not None:
        ids = _ordered_measurement_ids(scope.measurement_ids, n_predictions=n_attempted)
        for measurement_id in ids:
            row = connection.execute(
                "SELECT rm.test_id, rm.regime_id, rm.dataset_key, m.condition_pk "
                "FROM run_measurements AS rm JOIN measurements AS m "
                "ON m.measurement_id = rm.measurement_id "
                "WHERE rm.run_id = ? AND rm.measurement_id = ?",
                (run_id, measurement_id),
            ).fetchone()
            if row is None:
                raise ValueError("scope measurement is not a member of its run")
            if row[3] is None:
                has_measurement_without_condition = True
            else:
                selected_conditions.add(int(row[3]))
            for actual, selected, label in (
                (row[0], scope.test_id, "test_id"),
                (row[1], scope.regime_id, "regime_id"),
                (row[2], dataset_key, "dataset_key"),
                (row[3], scope.condition_pk, "condition_pk"),
            ):
                if selected is not None and actual != selected:
                    raise ValueError(f"scope {label} conflicts with measurement membership")
    model_id = _optional_integer(scope.model_id, "model_id", minimum=1)
    assumption_id = _optional_integer(scope.assumption_id, "assumption_id", minimum=1)
    condition_pk = _optional_integer(scope.condition_pk, "condition_pk", minimum=1)
    reference_id = _optional_integer(scope.reference_id, "reference_id", minimum=1)
    if scope.reference_ids is not None:
        if reference_id is not None:
            raise ValueError("use either one reference_id or a reference_ids set")
        if not isinstance(scope.reference_ids, Sequence) or isinstance(
            scope.reference_ids, (str, bytes),
        ) or not scope.reference_ids:
            raise ValueError("reference_ids must be a nonempty ordered sequence")
        reference_ids = tuple(
            _integer(item, "reference_id", minimum=1) for item in scope.reference_ids
        )
        if len(set(reference_ids)) != len(reference_ids):
            raise ValueError("reference_ids must not repeat")
    else:
        reference_ids = None
    source_artifact_id = _optional_integer(scope.source_artifact_id, "source_artifact_id", minimum=1)
    for table, column, value in (
        ("model_versions", "model_id", model_id),
        ("model_assumptions", "assumption_id", assumption_id),
        ("artifacts", "artifact_id", source_artifact_id),
    ):
        if value is not None:
            _require_row(connection, table, column, value)
    condition = None
    if condition_pk is not None:
        condition = connection.execute(
            "SELECT generating_model, signal_photon_count, background_per_bin "
            "FROM simulation_conditions WHERE condition_pk = ?", (condition_pk,),
        ).fetchone()
        if condition is None:
            raise ValueError("unknown simulation condition in metric scope")
    reference_kind = _optional_text(scope.reference_kind, "reference_kind")
    reference_version = _optional_text(scope.reference_version, "reference_version")
    if reference_kind not in (None, "generating_mono", "primary_component",
                               "trusted_experimental", "pseudo_true_mono"):
        raise ValueError("unsupported benchmark reference_kind")
    pseudo_reference_conditions: set[int] = set()
    for linked_reference_id in (
        (reference_id,) if reference_id is not None else (reference_ids or ())
    ):
        reference = connection.execute(
            "SELECT reference_kind, reference_version, condition_pk, assumption_id, "
            "measurement_id FROM lifetime_references WHERE reference_id = ?",
            (linked_reference_id,),
        ).fetchone()
        if reference is None:
            raise ValueError("unknown lifetime reference in metric scope")
        if reference_kind != reference[0] or reference_version != reference[1]:
            raise ValueError("metric reference kind/version conflicts with linked reference")
        if reference[0] == "pseudo_true_mono" and (
            condition_pk is not None and condition_pk != reference[2]
            or assumption_id is not None and assumption_id != reference[3]
        ):
            raise ValueError("pseudo-true metric scope conflicts with reference context")
        if reference[0] == "pseudo_true_mono":
            pseudo_reference_conditions.add(int(reference[2]))
        if reference[0] == "trusted_experimental" and ids is not None and (
            reference[4] not in ids
        ):
            raise ValueError("trusted reference is outside metric measurement population")
    if reference_kind in ("trusted_experimental", "pseudo_true_mono") and (
        reference_id is None and reference_ids is None
    ):
        raise ValueError("trusted and pseudo-true scopes require reference ID(s)")
    if reference_ids is not None and reference_kind not in (
        "trusted_experimental", "pseudo_true_mono",
    ):
        raise ValueError("reference_ids only apply to stored trusted or pseudo-true references")
    if reference_kind == "pseudo_true_mono" and ids is not None and (
        has_measurement_without_condition
        or pseudo_reference_conditions != selected_conditions
    ):
        raise ValueError("pseudo-true references must match selected measurement conditions")
    if reference_kind in ("generating_mono", "primary_component"):
        expected_model = "monoexponential" if reference_kind == "generating_mono" else "biexponential"
        if condition is not None and condition[0] != expected_model:
            raise ValueError("generating reference must match its simulation condition")
        if ids is not None and condition is None:
            generating_models = connection.execute(
                "SELECT c.generating_model FROM simulation_conditions AS c "
                "JOIN measurements AS m ON m.condition_pk = c.condition_pk "
                "WHERE m.measurement_id IN (" + ",".join("?" for _ in ids) + ")",
                ids,
            ).fetchall() if ids else []
            if len(generating_models) != len(ids) or any(
                item[0] != expected_model for item in generating_models
            ):
                raise ValueError("population generating models conflict with reference kind")
    if reference_kind is not None and reference_kind not in reference_semantics:
        raise ValueError("reference_semantics must identify the selected reference kind")
    if reference_kind is None and any(kind in reference_semantics for kind in (
        "generating_mono", "primary_component", "trusted_experimental", "pseudo_true_mono",
    )):
        raise ValueError("reference_semantics names a reference kind missing from scope")
    photon_count = _optional_integer(scope.signal_photon_count, "signal_photon_count", minimum=1)
    background = _optional_scalar(scope.background_per_bin, "background_per_bin", nonnegative=True)
    if condition is not None and (photon_count is not None and photon_count != condition[1]
                                  or background is not None and background != condition[2]):
        raise ValueError("metric regime values conflict with simulation condition")
    if condition is not None:
        photon_count = condition[1]
        background = condition[2]
    method_id = _optional_text(scope.method_id, "method_id")
    if (method_id is None) != (scope.method_configuration is None):
        raise ValueError("method_id and method_configuration must be supplied together")
    method_config_hash = None
    method_config = None
    if scope.method_configuration is not None:
        method_config_json, method_config_hash = _configuration_json(scope.method_configuration)
        method_config = json.loads(method_config_json)
    coverage = None
    if scope.nominal_coverage is not None:
        coverage = _finite_scalar(scope.nominal_coverage, "nominal_coverage")
        if not 0 < coverage < 1:
            raise ValueError("nominal_coverage must lie strictly between 0 and 1")
        if method_id is None:
            raise ValueError("nominal coverage requires an uncertainty method")
    context = {
        "run_id": run_id, "protocol_id": run[0], "protocol_version": run[1],
        "profile": run[2], "population_key": population_key,
        "dataset_key": dataset_key, "selection": selection,
        "measurement_ids": sorted(ids) if ids is not None else None,
        "model_id": model_id, "assumption_id": assumption_id,
        "condition_pk": condition_pk, "reference_id": reference_id,
        "reference_ids": sorted(reference_ids) if reference_ids is not None else None,
        "source_artifact_id": source_artifact_id,
        "test_id": _optional_text(scope.test_id, "test_id"),
        "regime_id": _optional_text(scope.regime_id, "regime_id"),
        "method_id": method_id, "method_configuration": method_config,
        "method_config_sha256": method_config_hash,
        "nominal_coverage": coverage, "reference_kind": reference_kind,
        "reference_version": reference_version,
        "reference_semantics": reference_semantics,
        "signal_photon_count": photon_count, "background_per_bin": background,
    }
    serialized = _canonical_json(context)
    context["scope_json"] = serialized
    context["scope_sha256"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return context


def _metric_fact(
    context: Mapping[str, Any], *, name: str, unit: str, value: Any,
    n_attempted: int, n_valid: int | None, n_contributing: int | None,
    denominator_kind: str,
) -> dict[str, Any]:
    attempted = _integer(n_attempted, "n_attempted")
    valid = _optional_integer(n_valid, "n_valid")
    contributing = _optional_integer(n_contributing, "n_contributing")
    if valid is not None and valid > attempted or contributing is not None and contributing > attempted:
        raise ValueError("metric counts cannot exceed attempted population")
    if valid is not None and contributing is not None and contributing > valid and (
        denominator_kind not in (
            "attempted_observations", "attempted_fits", "attempted_realizations",
        )
    ):
        raise ValueError("contributing count cannot exceed valid count")
    nonfinite: dict[str, str] = {}
    number = _result_number(value, "metric_value", nonfinite)
    return {
        "run_id": context["run_id"], "model_id": context["model_id"],
        "assumption_id": context["assumption_id"],
        "condition_pk": context["condition_pk"],
        "reference_id": context["reference_id"],
        "source_artifact_id": context["source_artifact_id"],
        "metric_name": _required_text(name, "metric_name"),
        "metric_unit": _required_text(unit, "metric_unit"),
        "scope_sha256": context["scope_sha256"],
        "scope_json": context["scope_json"],
        "value_status": "finite" if number is not None else "undefined",
        "metric_value": number, "n_attempted": attempted,
        "n_valid": valid, "n_contributing": contributing,
        "denominator_kind": _required_text(denominator_kind, "denominator_kind"),
        "test_id": context["test_id"], "regime_id": context["regime_id"],
        "method_id": context["method_id"],
        "nominal_coverage": context["nominal_coverage"],
        "reference_kind": context["reference_kind"],
        "reference_version": context["reference_version"],
        "signal_photon_count": context["signal_photon_count"],
        "background_per_bin": context["background_per_bin"],
        "nonfinite_fields_json": _canonical_json(nonfinite),
    }


def _record_metric_facts(
    connection: sqlite3.Connection, scope: BenchmarkMetricScope, *,
    n_attempted: int,
    facts: Sequence[tuple[str, str, Any, int | None, int | None, str]],
    on_duplicate: DuplicatePolicy,
) -> tuple[int, ...]:
    if not facts:
        raise ValueError("at least one benchmark metric fact is required")
    with transaction(connection):
        context = _metric_scope(connection, scope, n_attempted=n_attempted)
        payloads = [
            _metric_fact(context, name=name, unit=unit, value=value,
                         n_attempted=n_attempted, n_valid=valid,
                         n_contributing=contributing, denominator_kind=denominator)
            for name, unit, value, valid, contributing, denominator in facts
        ]
        return tuple(_record_row(
            connection, table="benchmark_metrics",
            key_columns=("run_id", "scope_sha256", "metric_name"),
            payload=payload, return_column="metric_id", on_duplicate=on_duplicate,
        ) for payload in payloads)


def _validate_attempt_failure_rate(
    value: Any, *, attempted: int, valid: int, name: str,
) -> None:
    rate = _finite_scalar(value, name)
    if not 0 <= rate <= 1 or not math.isclose(
        rate, 1.0 - valid / attempted, rel_tol=1e-9, abs_tol=1e-12,
    ):
        raise ValueError(f"{name} conflicts with attempted/valid counts")


def _validate_interval_coverage_consistency(
    empirical_coverage: Any, coverage_error: Any, *,
    nominal_coverage: Any, n_valid_intervals: int,
) -> None:
    """Source coverage_error is empirical minus nominal, not an absolute gap."""
    nonfinite: dict[str, str] = {}
    empirical = _result_number(empirical_coverage, "empirical_coverage", nonfinite)
    error = _result_number(coverage_error, "coverage_error", nonfinite)
    if n_valid_intervals == 0:
        if nonfinite != {"empirical_coverage": "nan", "coverage_error": "nan"}:
            raise ValueError("interval coverage is undefined with no valid intervals")
    elif empirical is None or error is None or not math.isclose(
        empirical - error, _finite_scalar(nominal_coverage, "nominal_coverage"),
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError("interval coverage/error conflicts with nominal coverage")


def _require_classical_aggregate_model(
    connection: sqlite3.Connection, scope: BenchmarkMetricScope, *,
    required_objective: str | None = None,
) -> str:
    """Validate the reusable reconvolution model selected for a classical fact."""
    model_id = _optional_integer(scope.model_id, "model_id", minimum=1)
    if model_id is None:
        raise ValueError("classical aggregate metrics require a model_id")
    row = connection.execute(
        "SELECT family, configuration_json FROM model_versions WHERE model_id = ?",
        (model_id,),
    ).fetchone()
    if row is None or row[0] != "classical":
        raise ValueError("classical aggregate metrics require a classical model")
    configuration = json.loads(row[1])
    objective = configuration.get("objective") if isinstance(configuration, dict) else None
    if not isinstance(objective, str) or objective not in ("poisson", "least_squares"):
        raise ValueError("classical aggregate model requires a recognized objective")
    if required_objective is not None and objective != required_objective:
        raise ValueError(f"classical aggregate model requires objective={required_objective}")
    if scope.assumption_id is not None:
        assumption = connection.execute(
            "SELECT assumed_decay_model, observation_model FROM model_assumptions "
            "WHERE assumption_id = ?", (scope.assumption_id,),
        ).fetchone()
        if assumption is None or assumption[0] != "monoexponential":
            raise ValueError("classical aggregate requires a monoexponential assumption")
        assumption_objective = {
            "poisson_reconvolution": "poisson",
            "least_squares_reconvolution": "least_squares",
        }.get(assumption[1])
        if assumption_objective is not None and objective != assumption_objective:
            raise ValueError("classical aggregate objective conflicts with assumption")
    return objective


def record_interval_metrics(
    connection: sqlite3.Connection, metrics: Any, *,
    scope: BenchmarkMetricScope, on_duplicate: DuplicatePolicy = "raise",
) -> tuple[int, ...]:
    """Record aggregate interval calibration/sharpness with source denominators."""
    from tcspc_toolkit.uncertainty_evaluation import IntervalEvaluationMetrics

    if not isinstance(metrics, IntervalEvaluationMetrics):
        raise TypeError("metrics must be IntervalEvaluationMetrics")
    if scope.method_id is None or scope.nominal_coverage is None:
        raise ValueError("interval metrics require method and nominal coverage in scope")
    attempted = _integer(metrics.n_samples, "n_samples", minimum=1)
    valid = _integer(metrics.n_valid_intervals, "n_valid_intervals")
    _validate_attempt_failure_rate(
        metrics.interval_failure_rate, attempted=attempted,
        valid=valid, name="interval_failure_rate",
    )
    _validate_interval_coverage_consistency(
        metrics.empirical_coverage, metrics.coverage_error,
        nominal_coverage=scope.nominal_coverage, n_valid_intervals=valid,
    )
    facts = [
        ("empirical_coverage", "fraction", metrics.empirical_coverage, valid, valid, "valid_intervals"),
        ("coverage_error", "fraction", metrics.coverage_error, valid, valid, "valid_intervals"),
        ("mean_interval_width_ns", "ns", metrics.mean_interval_width, valid, valid, "valid_intervals"),
        ("median_interval_width_ns", "ns", metrics.median_interval_width, valid, valid, "valid_intervals"),
        ("mean_interval_score_ns", "ns", metrics.mean_interval_score, valid, valid, "valid_intervals"),
        ("interval_failure_rate", "fraction", metrics.interval_failure_rate, valid, attempted, "attempted_observations"),
    ]
    return _record_metric_facts(
        connection, scope, n_attempted=attempted, facts=facts, on_duplicate=on_duplicate,
    )


def record_quantile_interval_metrics(
    connection: sqlite3.Connection, metrics: Any, *,
    scope: BenchmarkMetricScope, on_duplicate: DuplicatePolicy = "raise",
) -> tuple[int, ...]:
    """Record quantile pinball losses and crossing rate, not point intervals."""
    from tcspc_toolkit.uncertainty_evaluation import QuantileIntervalEvaluationMetrics

    if not isinstance(metrics, QuantileIntervalEvaluationMetrics):
        raise TypeError("metrics must be QuantileIntervalEvaluationMetrics")
    if scope.method_id is None or scope.nominal_coverage is None:
        raise ValueError("quantile metrics require method and nominal coverage")
    attempted = _integer(metrics.n_samples, "n_samples", minimum=1)
    valid = _integer(metrics.n_finite_quantile_triplets, "n_finite_quantile_triplets")
    facts = [
        ("lower_pinball_loss_ns", "ns", metrics.lower_pinball_loss, valid, valid, "finite_quantile_triplets"),
        ("median_pinball_loss_ns", "ns", metrics.median_pinball_loss, valid, valid, "finite_quantile_triplets"),
        ("upper_pinball_loss_ns", "ns", metrics.upper_pinball_loss, valid, valid, "finite_quantile_triplets"),
        ("mean_pinball_loss_ns", "ns", metrics.mean_pinball_loss, valid, valid, "finite_quantile_triplets"),
        ("quantile_crossing_rate", "fraction", metrics.quantile_crossing_rate, valid, valid, "finite_quantile_triplets"),
    ]
    return _record_metric_facts(
        connection, scope, n_attempted=attempted, facts=facts, on_duplicate=on_duplicate,
    )


def record_uncertainty_score_metrics(
    connection: sqlite3.Connection, metrics: Any, *,
    scope: BenchmarkMetricScope, scores: Any = None,
    on_duplicate: DuplicatePolicy = "raise",
) -> tuple[int, ...]:
    """Record aggregate score/error ranking; tail subset sizes remain unknown."""
    from tcspc_toolkit.uncertainty_evaluation import UncertaintyScoreMetrics

    if not isinstance(metrics, UncertaintyScoreMetrics):
        raise TypeError("metrics must be UncertaintyScoreMetrics")
    if scope.method_id is None or scope.nominal_coverage is not None:
        raise ValueError("score metrics require a method and no nominal coverage")
    attempted = _integer(metrics.n_samples, "n_samples", minimum=1)
    valid = _integer(metrics.n_valid_scores, "n_valid_scores")
    _validate_attempt_failure_rate(
        metrics.score_failure_rate, attempted=attempted,
        valid=valid, name="score_failure_rate",
    )
    fraction = _finite_scalar(metrics.tail_fraction, "tail_fraction")
    if not 0 < fraction <= 0.5:
        raise ValueError("tail_fraction must lie in (0, 0.5]")
    if not isinstance(scope.selection, Mapping) or scope.selection.get("tail_fraction") != fraction:
        raise ValueError("score metric selection must identify its tail_fraction")
    facts = [
        ("score_failure_rate", "fraction", metrics.score_failure_rate, valid, attempted, "attempted_observations"),
        ("mean_absolute_error_ns", "ns", metrics.mean_absolute_error_ns, valid, valid, "valid_scores"),
        ("spearman_error_correlation", "unitless", metrics.spearman_error_correlation, valid, valid, "valid_scores"),
        ("low_uncertainty_mae_ns", "ns", metrics.low_uncertainty_mae_ns, valid, None, "low_uncertainty_subset"),
        ("high_uncertainty_mae_ns", "ns", metrics.high_uncertainty_mae_ns, valid, None, "high_uncertainty_subset"),
        ("tail_fraction", "fraction", fraction, valid, valid, "valid_scores"),
    ]
    if scores is not None:
        from tcspc_toolkit.uncertainty_evaluation import UncertaintyScoreResult
        if not isinstance(scores, UncertaintyScoreResult):
            raise TypeError("scores must be UncertaintyScoreResult")
        if scores.method_id != scope.method_id or len(scores.prediction) != attempted or (
            int(np.count_nonzero(scores.valid_score_mask)) != valid
        ):
            raise ValueError("score output differs from aggregate metric population")
        mean_score = float(np.mean(scores.uncertainty_score[scores.valid_score_mask])) if valid else float("nan")
        facts.append(("mean_uncertainty_score", "ns", mean_score, valid, valid, "valid_scores"))
    return _record_metric_facts(
        connection, scope, n_attempted=attempted, facts=facts, on_duplicate=on_duplicate,
    )


def record_selective_prediction_metrics(
    connection: sqlite3.Connection, metrics: Any, *,
    scope: BenchmarkMetricScope, n_attempted: int,
    on_duplicate: DuplicatePolicy = "raise",
) -> tuple[int, ...]:
    """Record the existing selective-rejection evaluation at a stated population."""
    from tcspc_toolkit.uncertainty_evaluation import SelectivePredictionMetrics

    if not isinstance(metrics, SelectivePredictionMetrics):
        raise TypeError("metrics must be SelectivePredictionMetrics")
    if scope.method_id is None or scope.nominal_coverage is not None:
        raise ValueError("selective score metrics require a method and no nominal coverage")
    attempted = _integer(n_attempted, "n_attempted", minimum=1)
    valid = _integer(metrics.n_valid_scores, "n_valid_scores")
    retained = _integer(metrics.n_retained, "n_retained")
    if valid == 0 or retained > valid:
        raise ValueError("selective metrics require valid scores and retained <= valid")
    rejected = _finite_scalar(metrics.rejection_fraction, "rejection_fraction")
    if not 0 <= rejected < 1:
        raise ValueError("rejection_fraction must lie in [0, 1)")
    if not isinstance(scope.selection, Mapping) or scope.selection.get("rejection_fraction") != rejected:
        raise ValueError("selective metric scope must identify rejection_fraction")
    if not math.isclose(
        _finite_scalar(metrics.retained_fraction, "retained_fraction"),
        retained / valid, rel_tol=1e-9, abs_tol=1e-12,
    ):
        raise ValueError("retained_fraction conflicts with retained/valid counts")
    facts = [
        ("retained_fraction", "fraction", metrics.retained_fraction, valid, retained, "valid_scores"),
        ("mae_all_ns", "ns", metrics.mae_all_ns, valid, valid, "valid_scores"),
        ("mae_retained_ns", "ns", metrics.mae_retained_ns, valid, retained, "retained_predictions"),
        ("mae_improvement_ns", "ns", metrics.mae_improvement_ns, valid, retained, "retained_predictions"),
    ]
    return _record_metric_facts(
        connection, scope, n_attempted=attempted, facts=facts, on_duplicate=on_duplicate,
    )


def record_repeated_poisson_metrics(
    connection: sqlite3.Connection, result: Any, *,
    scope: BenchmarkMetricScope, on_duplicate: DuplicatePolicy = "raise",
) -> tuple[int, ...]:
    """Record empirical sampling variability and method comparisons as facts."""
    from tcspc_toolkit.classical_uncertainty import (
        REPEATED_POISSON_REFERENCE_METHOD_ID, RepeatedPoissonUncertaintyResult,
    )

    if not isinstance(result, RepeatedPoissonUncertaintyResult):
        raise TypeError("result must be RepeatedPoissonUncertaintyResult")
    _require_classical_aggregate_model(connection, scope, required_objective="poisson")
    if scope.method_id != REPEATED_POISSON_REFERENCE_METHOD_ID:
        raise ValueError("repeated-Poisson scope requires its source method_id")
    if scope.reference_kind != "generating_mono" or scope.condition_pk is None:
        raise ValueError("repeated-Poisson metrics require a mono generating condition")
    if scope.nominal_coverage is None or not math.isclose(
        scope.nominal_coverage, result.covariance_intervals.nominal_coverage,
        rel_tol=0.0, abs_tol=1e-12,
    ) or not math.isclose(
        scope.nominal_coverage, result.bootstrap_intervals.nominal_coverage,
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError("repeated-Poisson nominal coverage conflicts with source intervals")
    attempted = len(result.lifetime_estimates_ns)
    if attempted == 0:
        raise ValueError("repeated-Poisson result has no realizations")
    valid = int(np.count_nonzero(np.isfinite(result.lifetime_estimates_ns)))
    _validate_attempt_failure_rate(
        result.fit_failure_rate, attempted=attempted,
        valid=valid, name="fit_failure_rate",
    )
    if result.covariance_metrics.n_samples != attempted or result.bootstrap_metrics.n_samples != attempted:
        raise ValueError("repeated-Poisson interval counts conflict with realization count")
    for interval_metrics in (result.covariance_metrics, result.bootstrap_metrics):
        _validate_interval_coverage_consistency(
            interval_metrics.empirical_coverage, interval_metrics.coverage_error,
            nominal_coverage=scope.nominal_coverage,
            n_valid_intervals=interval_metrics.n_valid_intervals,
        )
    condition = connection.execute(
        "SELECT mono_lifetime_ns, signal_photon_count, background_per_bin "
        "FROM simulation_conditions WHERE condition_pk = ?", (scope.condition_pk,),
    ).fetchone()
    if condition is None or not math.isclose(result.true_lifetime_ns, condition[0], rel_tol=1e-12) or (
        result.signal_photon_count != condition[1] or result.background_per_bin != condition[2]
    ):
        raise ValueError("repeated-Poisson source conflicts with generating condition")
    selection = json.loads(_mapping_json(scope.selection, "selection"))
    for key, value in (
        ("irf_fwhm_ns", result.irf_fwhm_ns),
        ("irf_shift_ns", result.irf_shift_ns),
    ):
        finite_value = _finite_scalar(value, key, positive=(key == "irf_fwhm_ns"))
        if key in selection and selection[key] != finite_value:
            raise ValueError(f"repeated-Poisson {key} conflicts with metric scope")
        selection[key] = finite_value
    scope = replace(scope, selection=selection)
    covariance_valid = result.covariance_metrics.n_valid_intervals
    bootstrap_valid = result.bootstrap_metrics.n_valid_intervals
    facts = [
        ("empirical_lifetime_bias_ns", "ns", result.empirical_bias_ns, valid, valid, "successful_fits"),
        ("empirical_lifetime_std_ns", "ns", result.empirical_std_ns, valid, valid, "successful_fits"),
        ("empirical_lifetime_rmse_ns", "ns", result.empirical_rmse_ns, valid, valid, "successful_fits"),
        ("fit_failure_rate", "fraction", result.fit_failure_rate, valid, attempted, "attempted_realizations"),
        ("boundary_hit_rate", "fraction", result.boundary_hit_rate, valid, valid, "successful_fits"),
        ("mean_covariance_std_ns", "ns", result.mean_covariance_std_ns, valid, int(np.count_nonzero(np.isfinite(result.covariance_std_ns))), "finite_covariance_summaries"),
        ("covariance_to_empirical_std_ratio", "ratio", result.covariance_to_empirical_std_ratio, valid, None, "paired_empirical_comparison"),
        ("covariance_empirical_coverage", "fraction", result.covariance_metrics.empirical_coverage, covariance_valid, covariance_valid, "valid_intervals"),
        ("covariance_coverage_error", "fraction", result.covariance_metrics.coverage_error, covariance_valid, covariance_valid, "valid_intervals"),
        ("covariance_mean_interval_width_ns", "ns", result.covariance_metrics.mean_interval_width, covariance_valid, covariance_valid, "valid_intervals"),
        ("covariance_median_interval_width_ns", "ns", result.covariance_metrics.median_interval_width, covariance_valid, covariance_valid, "valid_intervals"),
        ("covariance_mean_interval_score_ns", "ns", result.covariance_metrics.mean_interval_score, covariance_valid, covariance_valid, "valid_intervals"),
        ("covariance_interval_failure_rate", "fraction", result.covariance_metrics.interval_failure_rate, covariance_valid, attempted, "attempted_realizations"),
        ("mean_bootstrap_std_ns", "ns", result.mean_bootstrap_std_ns, valid, int(np.count_nonzero(np.isfinite(result.bootstrap_std_ns))), "finite_bootstrap_summaries"),
        ("bootstrap_to_empirical_std_ratio", "ratio", result.bootstrap_to_empirical_std_ratio, valid, None, "paired_empirical_comparison"),
        ("bootstrap_empirical_coverage", "fraction", result.bootstrap_metrics.empirical_coverage, bootstrap_valid, bootstrap_valid, "valid_intervals"),
        ("bootstrap_coverage_error", "fraction", result.bootstrap_metrics.coverage_error, bootstrap_valid, bootstrap_valid, "valid_intervals"),
        ("bootstrap_mean_interval_width_ns", "ns", result.bootstrap_metrics.mean_interval_width, bootstrap_valid, bootstrap_valid, "valid_intervals"),
        ("bootstrap_median_interval_width_ns", "ns", result.bootstrap_metrics.median_interval_width, bootstrap_valid, bootstrap_valid, "valid_intervals"),
        ("bootstrap_mean_interval_score_ns", "ns", result.bootstrap_metrics.mean_interval_score, bootstrap_valid, bootstrap_valid, "valid_intervals"),
        ("bootstrap_interval_failure_rate", "fraction", result.bootstrap_metrics.interval_failure_rate, bootstrap_valid, attempted, "attempted_realizations"),
        ("mean_bootstrap_fit_failure_rate", "fraction", result.mean_bootstrap_fit_failure_rate, valid, int(np.count_nonzero(np.isfinite(result.bootstrap_fit_failure_rates))), "finite_bootstrap_failure_rates"),
    ]
    return _record_metric_facts(
        connection, scope, n_attempted=attempted, facts=facts, on_duplicate=on_duplicate,
    )


def record_regression_metrics(
    connection: sqlite3.Connection, metrics: Any, *,
    scope: BenchmarkMetricScope, n_attempted: int,
    on_duplicate: DuplicatePolicy = "raise",
) -> tuple[int, ...]:
    """Record an existing ML regression summary over an explicit test population."""
    from tcspc_toolkit.ml_evaluation import RegressionMetrics

    if not isinstance(metrics, RegressionMetrics):
        raise TypeError("metrics must be RegressionMetrics")
    attempted = _integer(n_attempted, "n_attempted", minimum=1)
    facts = [
        ("mae_ns", "ns", metrics.mae_ns, attempted, attempted, "evaluated_predictions"),
        ("median_absolute_error_ns", "ns", metrics.median_absolute_error_ns, attempted, attempted, "evaluated_predictions"),
        ("rmse_ns", "ns", metrics.rmse_ns, attempted, attempted, "evaluated_predictions"),
        ("mean_relative_error", "ratio", metrics.mean_relative_error, attempted, attempted, "evaluated_predictions"),
        ("median_relative_error", "ratio", metrics.median_relative_error, attempted, attempted, "evaluated_predictions"),
        ("r2", "unitless", metrics.r2, attempted, attempted, "evaluated_predictions"),
    ]
    return _record_metric_facts(
        connection, scope, n_attempted=attempted, facts=facts, on_duplicate=on_duplicate,
    )


def record_robustness_metrics(
    connection: sqlite3.Connection, metrics: Any, *,
    scope: BenchmarkMetricScope, on_duplicate: DuplicatePolicy = "raise",
) -> tuple[int, ...]:
    """Record the existing generalization-summary fields for one selected test."""
    from tcspc_toolkit.generalization_evaluation import RobustnessMetrics

    if not isinstance(metrics, RobustnessMetrics):
        raise TypeError("metrics must be RobustnessMetrics")
    attempted = _integer(metrics.n_samples, "n_samples", minimum=1)
    facts = [
        ("mae_ns", "ns", metrics.mae_ns, attempted, attempted, "evaluated_predictions"),
        ("median_absolute_error_ns", "ns", metrics.median_absolute_error_ns, attempted, attempted, "evaluated_predictions"),
        ("rmse_ns", "ns", metrics.rmse_ns, attempted, attempted, "evaluated_predictions"),
        ("bias_ns", "ns", metrics.bias_ns, attempted, attempted, "evaluated_predictions"),
        ("p90_absolute_error_ns", "ns", metrics.p90_absolute_error_ns, attempted, attempted, "evaluated_predictions"),
        ("p95_absolute_error_ns", "ns", metrics.p95_absolute_error_ns, attempted, attempted, "evaluated_predictions"),
    ]
    return _record_metric_facts(
        connection, scope, n_attempted=attempted, facts=facts, on_duplicate=on_duplicate,
    )


def record_reconvolution_benchmark_metrics(
    connection: sqlite3.Connection, summary: Any, *,
    scope: BenchmarkMetricScope, on_duplicate: DuplicatePolicy = "raise",
) -> tuple[int, ...]:
    """Record classical fit success, error, and runtime summary fields."""
    from tcspc_toolkit.classical_evaluation import ReconvolutionBenchmarkSummary

    if not isinstance(summary, ReconvolutionBenchmarkSummary):
        raise TypeError("summary must be ReconvolutionBenchmarkSummary")
    _require_classical_aggregate_model(connection, scope)
    attempted = _integer(summary.n_samples, "n_samples", minimum=1)
    successful = _integer(summary.n_successful_fits, "n_successful_fits")
    failed = _integer(summary.n_failed_fits, "n_failed_fits")
    if successful + failed != attempted:
        raise ValueError("classical benchmark fit counts are inconsistent")
    _validate_attempt_failure_rate(
        summary.failure_rate, attempted=attempted,
        valid=successful, name="failure_rate",
    )
    facts = [
        ("success_rate", "fraction", summary.success_rate, successful, attempted, "attempted_fits"),
        ("failure_rate", "fraction", summary.failure_rate, successful, attempted, "attempted_fits"),
        ("mae_valid_ns", "ns", summary.mae_valid_ns, successful, successful, "successful_fits"),
        ("median_absolute_error_valid_ns", "ns", summary.median_absolute_error_valid_ns, successful, successful, "successful_fits"),
        ("rmse_valid_ns", "ns", summary.rmse_valid_ns, successful, successful, "successful_fits"),
        ("mean_runtime_ms", "ms", summary.mean_runtime_ms, None, None, "finite_runtime_records"),
        ("median_runtime_ms", "ms", summary.median_runtime_ms, None, None, "finite_runtime_records"),
    ]
    return _record_metric_facts(
        connection, scope, n_attempted=attempted, facts=facts, on_duplicate=on_duplicate,
    )


def _week9_scorecard_row(row: Any, *, required: set[str]) -> Mapping[str, Any]:
    import pandas as pd

    if isinstance(row, pd.Series):
        row = row.to_dict()
    if not isinstance(row, Mapping) or not required.issubset(row):
        raise TypeError("row must be one current Week-9 scorecard row")
    return row


def _validate_week9_scorecard_scope(
    row: Mapping[str, Any], scope: BenchmarkMetricScope, *, interval: bool,
) -> None:
    if row["method"] != scope.method_id or row["test_id"] != scope.test_id:
        raise ValueError("Week-9 scorecard method/test differs from metric scope")
    reference_kind = {
        "monoexponential_lifetime": "generating_mono",
        "dominant_component_tau_1": "primary_component",
    }.get(row["target_reference"])
    if reference_kind is None or scope.reference_kind != reference_kind:
        raise ValueError("Week-9 scorecard reference differs from metric scope")
    if interval:
        if row["method"] not in (
            "quantile_gradient_boosting", "conformalized_quantile_gradient_boosting",
        ):
            raise ValueError("interval scorecard adapter requires a current ML interval method")
        if scope.nominal_coverage is None or not math.isclose(
            _finite_scalar(scope.nominal_coverage, "nominal_coverage"),
            _finite_scalar(row["nominal_coverage"], "nominal_coverage"),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("Week-9 scorecard coverage differs from metric scope")
    elif scope.nominal_coverage is not None:
        raise ValueError("score-only Week-9 metrics cannot claim nominal coverage")


def record_ml_interval_scorecard_row(
    connection: sqlite3.Connection, row: Any, *,
    scope: BenchmarkMetricScope, n_attempted: int, n_valid_intervals: int,
    n_valid_predictions: int,
    on_duplicate: DuplicatePolicy = "raise",
) -> tuple[int, ...]:
    """Record one ML interval scorecard row with explicit attempted/valid counts."""
    source = _week9_scorecard_row(row, required={
        "method", "test_id", "target_reference", "mae_ns", "nominal_coverage",
        "empirical_coverage", "coverage_gap", "mean_width_ns",
        "median_width_ns", "interval_score", "interval_failure_rate",
    })
    _validate_week9_scorecard_scope(source, scope, interval=True)
    attempted = _integer(n_attempted, "n_attempted", minimum=1)
    valid_intervals = _integer(n_valid_intervals, "n_valid_intervals")
    valid_predictions = _integer(n_valid_predictions, "n_valid_predictions")
    if valid_predictions > attempted or valid_intervals > valid_predictions:
        raise ValueError("interval scorecard prediction/interval counts are inconsistent")
    _validate_attempt_failure_rate(
        source["interval_failure_rate"], attempted=attempted,
        valid=valid_intervals, name="interval_failure_rate",
    )
    _validate_interval_coverage_consistency(
        source["empirical_coverage"], source["coverage_gap"],
        nominal_coverage=scope.nominal_coverage, n_valid_intervals=valid_intervals,
    )
    mae = _result_number(source["mae_ns"], "mae_ns", {})
    if mae is not None and valid_predictions != attempted:
        raise ValueError("finite ML scorecard MAE requires all attempted predictions to be finite")
    facts = [
        ("mae_ns", "ns", source["mae_ns"], valid_predictions, attempted, "attempted_observations"),
        ("empirical_coverage", "fraction", source["empirical_coverage"], valid_intervals, valid_intervals, "valid_intervals"),
        ("coverage_error", "fraction", source["coverage_gap"], valid_intervals, valid_intervals, "valid_intervals"),
        ("mean_interval_width_ns", "ns", source["mean_width_ns"], valid_intervals, valid_intervals, "valid_intervals"),
        ("median_interval_width_ns", "ns", source["median_width_ns"], valid_intervals, valid_intervals, "valid_intervals"),
        ("mean_interval_score_ns", "ns", source["interval_score"], valid_intervals, valid_intervals, "valid_intervals"),
        ("interval_failure_rate", "fraction", source["interval_failure_rate"], valid_intervals, attempted, "attempted_observations"),
    ]
    return _record_metric_facts(
        connection, scope, n_attempted=attempted, facts=facts, on_duplicate=on_duplicate,
    )


def record_ml_uncertainty_scorecard_row(
    connection: sqlite3.Connection, row: Any, *,
    scope: BenchmarkMetricScope, n_attempted: int, n_valid_scores: int,
    on_duplicate: DuplicatePolicy = "raise",
) -> tuple[int, ...]:
    """Record one ML uncertainty-score row without guessing absent tail counts."""
    source = _week9_scorecard_row(row, required={
        "method", "test_id", "target_reference", "mae_ns",
        "mean_uncertainty_score", "error_score_spearman",
        "low_uncertainty_mae_ns", "high_uncertainty_mae_ns",
        "score_failure_rate",
    })
    _validate_week9_scorecard_scope(source, scope, interval=False)
    if not isinstance(scope.selection, Mapping) or "tail_fraction" not in scope.selection:
        raise ValueError("scorecard scope must identify the tail_fraction")
    tail_fraction = _finite_scalar(scope.selection["tail_fraction"], "tail_fraction")
    if not 0 < tail_fraction <= 0.5:
        raise ValueError("tail_fraction must lie in (0, 0.5]")
    attempted = _integer(n_attempted, "n_attempted", minimum=1)
    valid = _integer(n_valid_scores, "n_valid_scores")
    _validate_attempt_failure_rate(
        source["score_failure_rate"], attempted=attempted,
        valid=valid, name="score_failure_rate",
    )
    facts = [
        ("mean_absolute_error_ns", "ns", source["mae_ns"], valid, valid, "valid_scores"),
        ("mean_uncertainty_score", "ns", source["mean_uncertainty_score"], valid, valid, "valid_scores"),
        ("spearman_error_correlation", "unitless", source["error_score_spearman"], valid, valid, "valid_scores"),
        ("low_uncertainty_mae_ns", "ns", source["low_uncertainty_mae_ns"], valid, None, "low_uncertainty_subset"),
        ("high_uncertainty_mae_ns", "ns", source["high_uncertainty_mae_ns"], valid, None, "high_uncertainty_subset"),
        ("score_failure_rate", "fraction", source["score_failure_rate"], valid, attempted, "attempted_observations"),
    ]
    return _record_metric_facts(
        connection, scope, n_attempted=attempted, facts=facts, on_duplicate=on_duplicate,
    )


# Stage-6 retrieval. SQL identifiers/expressions below are toolkit-owned; callers
# supply only allowlisted filter/sort names and bound scalar values.
@dataclass(frozen=True)
class QueryResult:
    """Column names and ordered tuple rows, including columns for an empty query.

    SQL NULL stays None. JSON is text unless decoding was explicitly requested.
    No scientific values, validity decisions or non-finite values are reconstructed.
    """

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]


def _query_equalities(
    filters: Mapping[str, Any] | None, allowed: Mapping[str, str],
) -> tuple[list[str], list[Any]]:
    if filters is None:
        return [], []
    if not isinstance(filters, Mapping):
        raise TypeError("filters must be a mapping of supported names to scalar values")
    clauses, values = [], []
    for name, value in filters.items():
        if name not in allowed:
            raise ValueError(f"unsupported query filter: {name!r}; choose from {tuple(allowed)}")
        if value is None:
            clauses.append(f"{allowed[name]} IS NULL")
        else:
            if not isinstance(value, (str, int, float)) or (
                isinstance(value, float) and not math.isfinite(value)
            ):
                raise TypeError("query filter values must be strings, integers, finite floats or None")
            clauses.append(f"{allowed[name]} = ?")
            values.append(value)
    return clauses, values


def _query_read(
    connection: sqlite3.Connection | str | Path, *, select: str, from_sql: str,
    filters: Mapping[str, Any] | None, allowed_filters: Mapping[str, str],
    order_by: str, allowed_order: Mapping[str, str], grain_order: tuple[str, ...],
    descending: bool, decode_json: bool,
    extra_where: Sequence[str] = (), extra_values: Sequence[Any] = (),
) -> QueryResult:
    if order_by not in allowed_order:
        raise ValueError(f"unsupported query order: {order_by!r}; choose from {tuple(allowed_order)}")
    if not isinstance(descending, bool) or not isinstance(decode_json, bool):
        raise TypeError("descending and decode_json must be bool")
    clauses, values = _query_equalities(filters, allowed_filters)
    clauses.extend(extra_where)
    values.extend(extra_values)
    selected_order = allowed_order[order_by]
    ordering = [selected_order + (" DESC" if descending else " ASC")]
    ordering.extend(column + " ASC" for column in grain_order if column != selected_order)
    sql = f"SELECT {select} FROM {from_sql}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY " + ", ".join(ordering)
    owned = not isinstance(connection, sqlite3.Connection)
    db = connect_database(connection, readonly=True) if owned else connection
    try:
        cursor = db.cursor()
        try:
            # Do not alter the caller's connection-wide factory or transaction.
            cursor.row_factory = None
            cursor.execute(sql, values)
            columns = tuple(item[0] for item in cursor.description)
            if len(set(columns)) != len(columns):
                raise PersistenceSchemaError("query has ambiguous duplicate column names")
            rows = tuple(cursor.fetchall())
        finally:
            cursor.close()
    finally:
        if owned:
            db.close()
    if decode_json:
        rows = tuple(tuple(
            json.loads(value) if column.endswith("_json") and value is not None else value
            for column, value in zip(columns, row)
        ) for row in rows)
    return QueryResult(columns, rows)


def query_to_dataframe(result: QueryResult) -> Any:
    """Present a QueryResult without dtype inference or scientific transformations.

    Object columns preserve exact Python integers (including nullable IDs above
    2**53), None, strings and JSON. Callers may explicitly choose analytical dtypes.
    Pandas is a package dependency, but is imported only for this conversion.
    """
    if not isinstance(result, QueryResult):
        raise TypeError("result must be a QueryResult")
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("query_to_dataframe requires the toolkit's pandas dependency") from exc
    return pd.DataFrame(result.rows, columns=result.columns, dtype=object)


def query_runs(
    connection: sqlite3.Connection | str | Path, *, filters: Mapping[str, Any] | None = None,
    order_by: str = "run_id", descending: bool = False, decode_json: bool = False,
) -> QueryResult:
    """One row per experiment run; exact filters, no joined child populations."""
    allowed = {name: f"r.{name}" for name in (
        "run_id", "run_key", "run_type", "status", "origin", "protocol_id",
        "protocol_version", "profile", "source_run_id",
    )}
    return _query_read(
        connection, select="r.*", from_sql="experiment_runs AS r",
        filters=filters, allowed_filters=allowed, order_by=order_by,
        allowed_order={name: f"r.{name}" for name in ("run_id", "run_key", "recorded_at_utc")},
        grain_order=("r.run_id",), descending=descending, decode_json=decode_json,
    )


def query_artifacts(
    connection: sqlite3.Connection | str | Path, *, filters: Mapping[str, Any] | None = None,
    order_by: str = "artifact_id", descending: bool = False, decode_json: bool = False,
) -> QueryResult:
    """One row per artifact registration; never opens or hashes the external file."""
    return _query_read(
        connection, select="a.*", from_sql="artifacts AS a", filters=filters,
        allowed_filters={name: f"a.{name}" for name in (
            "artifact_id", "artifact_key", "producing_run_id", "artifact_kind", "format", "sha256",
        )}, order_by=order_by,
        allowed_order={name: f"a.{name}" for name in ("artifact_id", "artifact_key")},
        grain_order=("a.artifact_id",), descending=descending, decode_json=decode_json,
    )


_MEMBERSHIP_QUERY_FILTERS = {name: f"rm.{name}" for name in (
    "run_id", "test_id", "regime_id", "dataset_key", "data_role", "pair_id", "realization_index",
)}
_GENERATING_QUERY_COLUMNS = """
    c.condition_key, c.condition_id, c.generating_model, c.generating_irf_id,
    c.mono_lifetime_ns AS generating_mono_lifetime_ns,
    c.primary_lifetime_ns AS generating_primary_lifetime_ns,
    c.secondary_lifetime_ns AS generating_secondary_lifetime_ns,
    c.secondary_detected_fraction AS generating_secondary_detected_fraction,
    c.signal_photon_count AS generating_signal_photon_count,
    c.background_per_bin AS generating_background_per_bin,
    c.true_temporal_shift_ns AS generating_shift_ns,
    c.true_reconvolution_amplitude AS generating_reconvolution_amplitude,
    c.expected_counts_sha256 AS generating_expected_counts_sha256,
    c.expected_counts_artifact_id AS generating_expected_counts_artifact_id,
    c.generating_parameters_json
"""


def query_measurements(
    connection: sqlite3.Connection | str | Path, *, filters: Mapping[str, Any] | None = None,
    include_generating: bool = False, include_membership: bool = False,
    order_by: str = "measurement_id", descending: bool = False, decode_json: bool = False,
) -> QueryResult:
    """One row per measurement, or explicitly one row per measurement/membership.

    Membership filters use one EXISTS row unless include_membership=True. That
    option expands all matching memberships; unlinked measurements have a NULL
    membership row when no membership filter excludes them. Generating context
    is an optional LEFT JOIN, never a reference or inferred truth selection.
    """
    allowed = {name: f"m.{name}" for name in (
        "measurement_id", "measurement_key", "sample_id", "source_type", "data_kind",
        "origin_run_id", "condition_pk", "attached_irf_source_id", "histogram_artifact_id",
    )}
    _query_equalities(filters, {**allowed, **_MEMBERSHIP_QUERY_FILTERS})
    selected = {} if filters is None else dict(filters)
    membership = {key: selected.pop(key) for key in tuple(selected) if key in _MEMBERSHIP_QUERY_FILTERS}
    select, from_sql = "m.*", "measurements AS m"
    if include_generating:
        select += ", " + _GENERATING_QUERY_COLUMNS
        from_sql += " LEFT JOIN simulation_conditions AS c ON c.condition_pk = m.condition_pk"
    extra_where, extra_values = [], []
    grain_order = ("m.measurement_id",)
    ordering = {name: f"m.{name}" for name in ("measurement_id", "measurement_key", "sample_id")}
    if include_membership:
        select += """, rm.run_id AS membership_run_id, r.run_key AS membership_run_key,
            rm.data_role, rm.test_id, rm.regime_id, rm.dataset_key, rm.pair_id,
            rm.realization_index, rm.membership_json"""
        from_sql += " LEFT JOIN run_measurements AS rm ON rm.measurement_id = m.measurement_id"
        from_sql += " LEFT JOIN experiment_runs AS r ON r.run_id = rm.run_id"
        extra_where, extra_values = _query_equalities(membership, _MEMBERSHIP_QUERY_FILTERS)
        if membership:
            extra_where.append("rm.run_id IS NOT NULL")
        grain_order += ("rm.run_id",)
        ordering["membership_run_id"] = "rm.run_id"
    elif membership:
        clauses, extra_values = _query_equalities(membership, _MEMBERSHIP_QUERY_FILTERS)
        extra_where = ["EXISTS (SELECT 1 FROM run_measurements AS rm "
                       "WHERE rm.measurement_id = m.measurement_id AND " + " AND ".join(clauses) + ")"]
    return _query_read(
        connection, select=select, from_sql=from_sql, filters=selected, allowed_filters=allowed,
        order_by=order_by, allowed_order=ordering, grain_order=grain_order,
        descending=descending, decode_json=decode_json,
        extra_where=extra_where, extra_values=extra_values,
    )


def query_irf_sources(
    connection: sqlite3.Connection | str | Path, *, filters: Mapping[str, Any] | None = None,
    order_by: str = "irf_source_id", descending: bool = False, decode_json: bool = False,
) -> QueryResult:
    """One row per original IRF source, including sources with no preparation."""
    return _query_read(
        connection, select="s.*", from_sql="irf_sources AS s", filters=filters,
        allowed_filters={name: f"s.{name}" for name in (
            "irf_source_id", "source_key", "source_kind", "source_representation",
            "derived_from_measurement_id", "source_artifact_id",
        )}, order_by=order_by,
        allowed_order={name: f"s.{name}" for name in ("irf_source_id", "source_key")},
        grain_order=("s.irf_source_id",), descending=descending, decode_json=decode_json,
    )


def query_prepared_irfs(
    connection: sqlite3.Connection | str | Path, *, filters: Mapping[str, Any] | None = None,
    include_source: bool = False, order_by: str = "prepared_irf_id",
    descending: bool = False, decode_json: bool = False,
) -> QueryResult:
    """One row per prepared IRF; optional source context never changes this grain."""
    select = "p.*"
    if include_source:
        select += """, s.source_key, s.source_kind, s.source_representation,
            s.n_bins AS source_n_bins, s.source_grid_sha256, s.source_values_sha256,
            s.source_artifact_id, s.derived_from_measurement_id, s.source_parameters_json,
            s.metadata_json AS source_metadata_json, s.provenance_json AS source_provenance_json"""
    return _query_read(
        connection, select=select,
        from_sql="prepared_irfs AS p LEFT JOIN irf_sources AS s ON s.irf_source_id = p.irf_source_id",
        filters=filters, allowed_filters={
            **{name: f"p.{name}" for name in (
                "prepared_irf_id", "preparation_key", "irf_source_id", "preparation_kind",
                "time_grid_sha256", "kernel_sha256",
            )}, "source_kind": "s.source_kind", "source_key": "s.source_key",
        }, order_by=order_by,
        allowed_order={name: f"p.{name}" for name in ("prepared_irf_id", "preparation_key", "irf_source_id")},
        grain_order=("p.prepared_irf_id",), descending=descending, decode_json=decode_json,
    )


_RESULT_QUERY_JOINS = """
    JOIN measurements AS m ON m.measurement_id = e.measurement_id
    JOIN experiment_runs AS r ON r.run_id = e.run_id
    JOIN model_versions AS v ON v.model_id = e.model_id
    LEFT JOIN model_assumptions AS a ON a.assumption_id = e.assumption_id
    JOIN run_measurements AS rm ON rm.run_id = e.run_id AND rm.measurement_id = e.measurement_id
"""
_RESULT_QUERY_FILTERS = {
    **{name: f"e.{name}" for name in (
        "result_id", "run_id", "measurement_id", "model_id", "assumption_id", "analysis_key",
        "status", "is_valid", "point_summary", "irf_model_relation",
    )},
    **{name: f"rm.{name}" for name in ("test_id", "regime_id", "dataset_key", "data_role", "pair_id")},
    "run_key": "r.run_key", "measurement_key": "m.measurement_key", "sample_id": "m.sample_id",
    "condition_pk": "m.condition_pk", "model_key": "v.model_key", "model_family": "v.family",
    "estimator_name": "v.estimator_name", "assumption_key": "a.assumption_key",
}
_RESULT_CONTEXT_COLUMNS = """
    r.run_key, m.measurement_key, m.sample_id, m.data_kind, m.source_type, m.condition_pk,
    v.model_key, v.estimator_name, v.family AS model_family, v.representation_id, v.prior_policy_id,
    v.configuration_json AS model_configuration_json, v.configuration_sha256 AS model_configuration_sha256,
    a.assumption_key, a.source_assumption_id, a.assumed_decay_model, a.observation_model,
    a.background_convention, a.prepared_irf_id AS assumed_prepared_irf_id, a.context_completeness,
    a.fixed_temporal_shift_ns AS assumed_fixed_temporal_shift_ns,
    a.temporal_shift_lower_ns AS assumed_temporal_shift_lower_ns,
    a.temporal_shift_upper_ns AS assumed_temporal_shift_upper_ns,
    a.configuration_json AS assumption_configuration_json,
    rm.data_role, rm.test_id, rm.regime_id, rm.dataset_key, rm.pair_id,
    rm.realization_index, rm.membership_json
"""
_OWNING_RESULT_COLUMNS = """
    e.run_id, e.measurement_id, e.model_id, e.assumption_id, e.analysis_key,
    e.point_summary, e.lifetime_estimate_ns, e.status AS result_status, e.is_valid AS result_is_valid,
    e.irf_model_relation, e.source_result_type,
    e.failure_reason AS result_failure_reason, e.runtime_seconds AS result_runtime_seconds,
    e.runtime_scope AS result_runtime_scope, e.random_seed_decimal AS result_random_seed_decimal,
    e.execution_json AS result_execution_json, e.nonfinite_fields_json AS result_nonfinite_fields_json
"""
_FIT_QUERY_FIELDS = (
    "result_id", "source_success", "optimizer_reported_success", "valid_fit",
    "numerical_validation_passed", "recovery_attempted", "boundary_hit", "fitted_amplitude",
    "fitted_background_per_bin", "fitted_temporal_shift_ns", "initial_amplitude",
    "initial_lifetime_ns", "initial_background_per_bin", "initial_temporal_shift_ns",
    "poisson_nll", "poisson_deviance", "max_coordinate_descent_nll", "optimizer_status",
    "optimizer_message", "optimizer_nfev", "optimizer_njev", "optimizer_seconds", "call_seconds",
    "exception_message", "diagnostics_json", "nonfinite_fields_json",
)


def query_results(
    connection: sqlite3.Connection | str | Path, *, filters: Mapping[str, Any] | None = None,
    include_context: bool = False, include_fit_details: bool = False,
    order_by: str = "result_id", descending: bool = False, decode_json: bool = False,
) -> QueryResult:
    """One row per estimator result; optional context and fit joins are to-one only.

    No uncertainty/reference expansion or error calculation. Fit columns have a
    fit_ prefix, including fit_result_id (NULL when no fit extension exists).
    """
    select, from_sql = "e.*", "estimator_results AS e " + _RESULT_QUERY_JOINS
    if include_context:
        select += ", " + _RESULT_CONTEXT_COLUMNS
    if include_fit_details:
        select += ", " + ", ".join(f"f.{name} AS fit_{name}" for name in _FIT_QUERY_FIELDS)
        from_sql += " LEFT JOIN fit_details AS f ON f.result_id = e.result_id"
    return _query_read(
        connection, select=select, from_sql=from_sql, filters=filters,
        allowed_filters=_RESULT_QUERY_FILTERS, order_by=order_by,
        allowed_order={name: f"e.{name}" for name in (
            "result_id", "run_id", "measurement_id", "model_id", "assumption_id", "lifetime_estimate_ns",
        )}, grain_order=("e.result_id",), descending=descending, decode_json=decode_json,
    )


def _query_extension_filters(alias: str, fields: Sequence[str]) -> dict[str, str]:
    # Extension is_valid/runtime/etc. retain their own meaning; point validity
    # and status are separate, explicitly named filters.
    allowed = {key: value for key, value in _RESULT_QUERY_FILTERS.items()
               if key not in ("status", "is_valid")}
    allowed.update(result_status="e.status", result_is_valid="e.is_valid")
    allowed.update({name: f"{alias}.{name}" for name in fields})
    return allowed


def query_uncertainty(
    connection: sqlite3.Connection | str | Path, *, filters: Mapping[str, Any] | None = None,
    include_context: bool = False, order_by: str = "uncertainty_id",
    descending: bool = False, decode_json: bool = False,
) -> QueryResult:
    """One uncertainty row with its owning point fields; never evaluate coverage."""
    select = "u.*, " + _OWNING_RESULT_COLUMNS
    if include_context:
        select += ", " + _RESULT_CONTEXT_COLUMNS
    return _query_read(
        connection, select=select,
        from_sql="uncertainty_results AS u JOIN estimator_results AS e ON e.result_id = u.result_id "
                 + _RESULT_QUERY_JOINS,
        filters=filters, allowed_filters=_query_extension_filters("u", (
            "uncertainty_id", "method_id", "output_kind", "nominal_coverage", "is_valid",
            "calibration_run_id", "calibration_scope", "method_config_sha256", "samples_artifact_id",
        )), order_by=order_by,
        allowed_order={name: f"u.{name}" for name in ("uncertainty_id", "result_id", "method_id", "nominal_coverage")},
        grain_order=("u.uncertainty_id",), descending=descending, decode_json=decode_json,
    )


def query_bayesian(
    connection: sqlite3.Connection | str | Path, *, filters: Mapping[str, Any] | None = None,
    include_context: bool = False, order_by: str = "result_id",
    descending: bool = False, decode_json: bool = False,
) -> QueryResult:
    """One Bayesian summary joined to its point; no chain loading or re-diagnosis.

    Parameter-specific diagnostics, credible bounds, configured counts and PPC
    seed stay in their existing JSON structures; unavailable facts remain NULL.
    """
    select = "b.*, " + _OWNING_RESULT_COLUMNS
    if include_context:
        select += ", " + _RESULT_CONTEXT_COLUMNS
    return _query_read(
        connection, select=select,
        from_sql="bayesian_summaries AS b JOIN estimator_results AS e ON e.result_id = b.result_id "
                 + _RESULT_QUERY_JOINS,
        filters=filters, allowed_filters=_query_extension_filters("b", (
            "sampling_status", "diagnostics_accepted", "ppc_status", "posterior_artifact_id",
            "predictive_artifact_id",
        )), order_by=order_by,
        allowed_order={"result_id": "b.result_id", "run_id": "e.run_id", "measurement_id": "e.measurement_id"},
        grain_order=("b.result_id",), descending=descending, decode_json=decode_json,
    )


def query_references(
    connection: sqlite3.Connection | str | Path, *, filters: Mapping[str, Any] | None = None,
    order_by: str = "reference_id", descending: bool = False, decode_json: bool = False,
) -> QueryResult:
    """One explicitly scoped lifetime-reference row; return all matching versions.

    measurement_id matches only the stored measurement FK, not a condition-derived
    lookup. No supersession traversal or implicit choice of an evaluation target.
    """
    return _query_read(
        connection, select="l.*", from_sql="lifetime_references AS l", filters=filters,
        allowed_filters={name: f"l.{name}" for name in (
            "reference_id", "reference_key", "reference_kind", "reference_version",
            "measurement_id", "condition_pk", "assumption_id", "supersedes_reference_id",
            "source_artifact_id",
        )}, order_by=order_by,
        allowed_order={name: f"l.{name}" for name in ("reference_id", "reference_key", "reference_version")},
        grain_order=("l.reference_id",), descending=descending, decode_json=decode_json,
    )


def query_metrics(
    connection: sqlite3.Connection | str | Path, *, filters: Mapping[str, Any] | None = None,
    order_by: str = "metric_id", descending: bool = False, decode_json: bool = False,
) -> QueryResult:
    """One stored benchmark fact with its original scope, counts and value status.

    Dataset/population/protocol/method-config filters read named scope JSON fields;
    they never rebuild the scope/hash. reference_id filters the scalar FK only,
    not membership in the explicit reference_ids set inside scope_json.
    """
    allowed = {name: f"b.{name}" for name in (
        "metric_id", "run_id", "model_id", "assumption_id", "condition_pk", "reference_id",
        "source_artifact_id", "metric_name", "metric_unit", "scope_sha256", "value_status",
        "denominator_kind", "test_id", "regime_id", "method_id", "nominal_coverage",
        "reference_kind", "reference_version", "signal_photon_count", "background_per_bin",
    )}
    allowed.update({name: f"json_extract(b.scope_json, '$.{name}')" for name in (
        "dataset_key", "population_key", "reference_semantics", "method_config_sha256",
        "protocol_id", "protocol_version", "profile",
    )})
    return _query_read(
        connection, select="b.*", from_sql="benchmark_metrics AS b", filters=filters,
        allowed_filters=allowed, order_by=order_by,
        allowed_order={name: f"b.{name}" for name in ("metric_id", "run_id", "metric_name", "test_id", "regime_id")},
        grain_order=("b.metric_id",), descending=descending, decode_json=decode_json,
    )


# Legacy public names: direct aliases retained at least through Issue #12.
record_week9_interval_scorecard_row = record_ml_interval_scorecard_row
record_week9_score_only_scorecard_row = record_ml_uncertainty_scorecard_row
