"""Compatibility exports; use :mod:`tcspc_toolkit.benchmarks.generalization_datasets`."""

from tcspc_toolkit.benchmarks.generalization_datasets import (
    FloatArray,
    IntArray,
    PAIR_VARIANTS_PER_CONDITION,
    GeneralizationTestMeasurements,
    GeneralizationTestSuite,
    build_generalization_time_axis,
    build_paired_parameter_design,
    build_test_parameter_table,
    build_test_f_parameter_table,
    generate_generalization_test,
    generate_generalization_test_suite,
    build_generalization_suite_manifest,
)

__all__ = (
    "FloatArray",
    "IntArray",
    "PAIR_VARIANTS_PER_CONDITION",
    "GeneralizationTestMeasurements",
    "GeneralizationTestSuite",
    "build_generalization_time_axis",
    "build_paired_parameter_design",
    "build_test_parameter_table",
    "build_test_f_parameter_table",
    "generate_generalization_test",
    "generate_generalization_test_suite",
    "build_generalization_suite_manifest",
)
