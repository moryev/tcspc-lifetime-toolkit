"""Behavioral contracts for neutral, reference-aware evaluation facts."""

import numpy as np
import pandas as pd
import pytest

from tcspc_toolkit.evaluation_core import build_point_evaluation, combine_point_evaluations
from tcspc_toolkit.evaluation_results import (
    EvaluationBatch, LifetimeReference, MethodDescriptor, PointEvaluationResult,
)


@pytest.mark.parametrize("family", ["baseline", "ml", "classical", "bayesian"])
def test_method_family_is_explicit_and_representation_is_optional(family):
    method = MethodDescriptor("classical_reconvolution_custom", family)
    assert method.family == family
    assert method.representation_id is None


@pytest.mark.parametrize("args", [
    (" ", "ml"), ("x", "unknown"), ("x", "ml", ""),
])
def test_ambiguous_method_identifiers_are_rejected(args):
    with pytest.raises(ValueError):
        MethodDescriptor(*args)


def test_batch_passes_matrices_through_without_conversion_or_index_alignment():
    class BackendMatrix:
        shape = (2, 4)

        def __array__(self, *args, **kwargs):
            raise AssertionError("matrix must not be converted")

    matrix = BackendMatrix()
    frame = pd.DataFrame({"feature": [1, 2]}, index=[90, 80])
    metadata = pd.DataFrame({"sample_id": ["b", "a"]}, index=[8, 7])
    sample_ids = ["b", "a"]
    representations = {"backend/custom": matrix, "frame": frame}
    reference = LifetimeReference("ref", "trusted_experimental", [1, 2], [True, True])
    references = {"ref": reference}
    batch = EvaluationBatch("experiment/run 7", sample_ids, representations, metadata, references)
    assert batch.sample_ids == ("b", "a")
    assert batch.representations["backend/custom"] is matrix
    assert batch.representations["frame"] is frame
    metadata.iloc[0, 0] = "changed"
    sample_ids.reverse()
    representations.clear()
    references.clear()
    returned_metadata = batch.metadata
    returned_metadata.iloc[0, 0] = "changed"
    returned_metadata.drop(index=7, inplace=True)
    assert batch.sample_ids == ("b", "a")
    assert batch.metadata.sample_id.tolist() == ["b", "a"]
    assert len(batch.representations) == 2
    assert batch.references["ref"] is reference
    for mapping in (batch.representations, batch.references):
        with pytest.raises(TypeError):
            mapping["new"] = None
    frame.iloc[0, 0] = 99
    assert batch.representations["frame"].iloc[0, 0] == 99


@pytest.mark.parametrize("kwargs", [
    {"sample_ids": [1, 1]},
    {"sample_ids": [True, 2]},
    {"sample_ids": []},
    {"representations": {"x": np.ones((1, 2))}},
    {"metadata": pd.DataFrame({"x": [1]})},
    {"metadata": pd.DataFrame({"sample_id": [2, 1]})},
    {"metadata": pd.DataFrame({"evaluation_id": ["other", "other"]})},
    {"references": {"ref": LifetimeReference("ref", "generating_mono", [1], [True])}},
    {"references": {"wrong": LifetimeReference("ref", "generating_mono", [1, 2], [True, True])}},
])
def test_batch_rejects_ambiguous_or_misaligned_inputs(kwargs):
    arguments = {"evaluation_id": "a new evaluation", "sample_ids": [1, 2], **kwargs}
    with pytest.raises(ValueError):
        EvaluationBatch(**arguments)


def test_references_preserve_kind_scope_and_copy_input_arrays():
    values = np.array([2.0, 5.0])
    available = np.array([True, False])
    primary = LifetimeReference(
        "primary", "primary_component", values, available, assumption_label="component 1",
    )
    pseudo = LifetimeReference(
        "projection", "pseudo_true_mono", values, available,
        reference_version="corrected-v1", assumption_label="mono/fixed-irf",
    )
    values[:] = 9
    available[:] = False
    np.testing.assert_array_equal(primary.values_ns, [2, 5])
    np.testing.assert_array_equal(primary.available, [True, False])
    for array in (primary.values_ns, primary.available):
        with pytest.raises(ValueError):
            array[0] = 0
    assert primary.kind != pseudo.kind
    assert primary.assumption_label == "component 1"
    assert pseudo.assumption_label == "mono/fixed-irf"
    assert pseudo.reference_version == "corrected-v1"


@pytest.mark.parametrize("kind, values, available, extras", [
    ("weighted_mixture", [2], [True], {}),
    ("trusted_experimental", [0], [True], {}),
    ("generating_mono", [np.nan], [True], {}),
    ("generating_mono", [1, 2], [True], {}),
    ("primary_component", [1], [1], {}),
    ("pseudo_true_mono", [2], [True], {}),
    ("primary_component", [2], [True], {"assumption_label": " "}),
])
def test_reference_validation(kind, values, available, extras):
    with pytest.raises(ValueError):
        LifetimeReference("ref", kind, values, available, **extras)


def _result(*, evaluation_id="experiment", method=None, reference=None):
    if reference is None:
        reference = LifetimeReference("calibration", "trusted_experimental", [2, 4], [True, True])
    return build_point_evaluation(
        batch=EvaluationBatch(evaluation_id, [9, 3], references={reference.reference_id: reference}),
        method=MethodDescriptor("opaque", "classical") if method is None else method,
        lifetime_estimates_ns=[3, 5], is_valid=[True, False],
        failure_reasons=[None, "optimizer rejected"],
    )


def test_combining_preserves_order_and_rejects_duplicate_nullable_identity():
    first = _result()
    second = build_point_evaluation(
        batch=EvaluationBatch("other", [3]), method=MethodDescriptor("opaque", "classical"),
        lifetime_estimates_ns=[2], is_valid=[True],
    )
    combined = combine_point_evaluations([first, second])
    assert combined.points.evaluation_id.tolist() == ["experiment", "experiment", "other"]
    assert combined.points.sample_id.tolist() == [9, 3, 3]
    assert len(combined.reference_comparisons) == 2
    with pytest.raises(ValueError, match="Duplicate point identity"):
        combine_point_evaluations([first, first])


@pytest.mark.parametrize("corruption", ["orphan", "sample_identity", "duplicate", "error", "validity"])
def test_result_constructor_validates_linked_facts(corruption):
    result = _result()
    points = result.points
    comparisons = result.reference_comparisons
    if corruption == "orphan":
        comparisons.loc[0, "sample_id"] = 100
    elif corruption == "sample_identity":
        # Numeric equality must not make float IDs masquerade as integer IDs.
        comparisons["sample_id"] = comparisons.sample_id.astype(float)
    elif corruption == "duplicate":
        comparisons = pd.concat([comparisons, comparisons.iloc[:1]])
    elif corruption == "error":
        comparisons.loc[0, "error_ns"] = 100.0
    else:
        comparisons.loc[1, "is_valid_comparison"] = True
    with pytest.raises(ValueError):
        PointEvaluationResult(points, comparisons)


def test_combining_rejects_conflicting_reference_facts_across_methods():
    first = _result()
    second = _result(
        method=MethodDescriptor("another", "ml"),
        reference=LifetimeReference("calibration", "trusted_experimental", [3, 4], [True, True]),
    )
    with pytest.raises(ValueError, match="Conflicting reference values"):
        combine_point_evaluations([first, second])


def test_method_family_cannot_change_across_representations_or_batches():
    first = _result()
    second = _result(evaluation_id="other", method=MethodDescriptor("opaque", "ml", "custom"))
    with pytest.raises(ValueError, match="Conflicting method family"):
        combine_point_evaluations([first, second])
    compatible = _result(evaluation_id="other", method=MethodDescriptor("opaque", "classical", "custom"))
    assert len(combine_point_evaluations([first, compatible]).points) == 4


@pytest.mark.parametrize("scope", [
    {"kind": "primary_component"}, {"reference_version": "different"},
    {"assumption_label": "different context"},
])
def test_reference_definition_cannot_change_across_batches(scope):
    arguments = {"kind": "trusted_experimental", **scope}
    other = _result(
        evaluation_id="other",
        reference=LifetimeReference("calibration", values_ns=[2, 4], available=[True, True], **arguments),
    )
    with pytest.raises(ValueError, match="Conflicting reference definition"):
        combine_point_evaluations([_result(), other])


def test_result_is_a_snapshot_of_inputs_and_returns_defensive_table_copies():
    original = _result()
    points, comparisons = original.points, original.reference_comparisons
    snapshot = PointEvaluationResult(points, comparisons)
    points.loc[0, "is_valid"] = False
    comparisons.loc[0, "error_ns"] = 100
    returned_points, returned_comparisons = snapshot.points, snapshot.reference_comparisons
    returned_points.loc[0, "is_valid"] = False
    returned_comparisons.loc[0, "reference_value_ns"] = 100
    pd.testing.assert_frame_equal(snapshot.points, original.points)
    pd.testing.assert_frame_equal(snapshot.reference_comparisons, original.reference_comparisons)
    # Edited copies can only become a new canonical result by passing validation.
    with pytest.raises(ValueError, match="is_valid_comparison"):
        PointEvaluationResult(returned_points, snapshot.reference_comparisons)
