"""Reference-independent uncertainty attachments for factual point results.

These are small reporting projections, not uncertainty algorithms or replacements
for method-specific samples, covariance matrices, or diagnostics. Tables are
validated snapshots exposed through defensive-copy properties.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tcspc_toolkit.evaluation_results import (
    PointEvaluationResult, _POINT_KEY, _float_vector, _identifier, _mask, _sample_ids,
)


__all__ = ["IntervalAttachment", "ScoreAttachment"]

_INTERVAL_COLUMNS = (
    *_POINT_KEY, "uncertainty_method_id", "interval_kind", "nominal_level",
    "lower_ns", "upper_ns", "is_valid_interval",
)
_SCORE_COLUMNS = (
    *_POINT_KEY, "uncertainty_method_id", "score", "is_valid_score",
)


def _attachment_table(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or set(frame) != set(columns):
        raise ValueError(f"Attachment table must have exactly these columns: {columns}.")
    table = frame.loc[:, list(columns)].copy(deep=True).reset_index(drop=True)
    # None and pandas missing values denote the same absent representation/level.
    representation = table["representation_id"].astype(object)
    table["representation_id"] = representation.where(representation.notna(), None)
    for name in ("lower_ns", "upper_ns", "score"):
        if name in table:
            table[name] = _float_vector(table[name], name)
    if "nominal_level" in table:
        levels = table["nominal_level"].astype(object)
        table["nominal_level"] = pd.Series(
            [None if pd.isna(value) else float(value) for value in levels],
            dtype=object,
        )
    return table


def _validate_point_links(points: PointEvaluationResult, table: pd.DataFrame) -> None:
    if not isinstance(points, PointEvaluationResult):
        raise TypeError("points must be a PointEvaluationResult.")
    known = {
        tuple(getattr(row, name) for name in _POINT_KEY)
        for row in points.points.itertuples(index=False)
    }
    for row in table.itertuples(index=False):
        _identifier(row.evaluation_id, "evaluation_id")
        _sample_ids((row.sample_id,))
        _identifier(row.method_id, "method_id")
        if row.representation_id is not None:
            _identifier(row.representation_id, "representation_id")
        _identifier(row.uncertainty_method_id, "uncertainty_method_id")
        if tuple(getattr(row, name) for name in _POINT_KEY) not in known:
            raise ValueError("Attachment has no matching point identity.")


@dataclass(frozen=True, init=False)
class IntervalAttachment:
    """Intervals linked to point identities, with separate method validity.

    Identity adds uncertainty_method_id, interval_kind and nominal_level to
    the point key. A method ID has one interval kind within this attachment;
    distinct levels of that kind may coexist. A level is the method-declared
    probability/confidence/credible level, interpreted by interval kind; it is
    not observed coverage. An absent level makes no nominal-level claim. Valid
    intervals require finite, ordered endpoints. Invalid intervals retain raw
    finite, crossed or nonfinite endpoints without repair.
    Point validity and reference availability are independent of interval validity.
    """

    _intervals: pd.DataFrame = field(repr=False)

    def __init__(self, points: PointEvaluationResult, intervals: pd.DataFrame) -> None:
        table = _attachment_table(intervals, _INTERVAL_COLUMNS)
        identity = [*_POINT_KEY, "uncertainty_method_id", "interval_kind", "nominal_level"]
        if table.duplicated(identity).any():
            raise ValueError("Duplicate interval attachment identity.")
        _validate_point_links(points, table)
        valid = _mask(table["is_valid_interval"], len(table), "is_valid_interval")
        lower = _float_vector(table["lower_ns"], "lower_ns")
        upper = _float_vector(table["upper_ns"], "upper_ns")
        if np.any(valid & (~np.isfinite(lower) | ~np.isfinite(upper) | (lower > upper))):
            raise ValueError("Valid intervals must have finite, ordered endpoints.")
        kinds: dict[str, str] = {}
        for row in table.itertuples(index=False):
            _identifier(row.interval_kind, "interval_kind")
            if kinds.setdefault(row.uncertainty_method_id, row.interval_kind) != row.interval_kind:
                raise ValueError("One uncertainty method cannot change interval kind.")
            if row.nominal_level is not None and not (
                np.isfinite(row.nominal_level) and 0.0 < row.nominal_level < 1.0
            ):
                raise ValueError("nominal_level must be strictly between zero and one.")
        object.__setattr__(self, "_intervals", table)

    @property
    def intervals(self) -> pd.DataFrame:
        """Return an editable copy of the validated interval rows."""
        return self._intervals.copy(deep=True)


@dataclass(frozen=True, init=False)
class ScoreAttachment:
    """Opaque method-specific scores linked to point identities.

    Identity adds uncertainty_method_id to the point key. Valid scores must be
    finite; invalid scores retain raw values. Signed scores are permitted because
    interpretation, scale and direction belong to uncertainty_method_id.
    Existing RF/bootstrap spread sources additionally require nonnegative scores
    in their own contract.
    Scores make no nominal coverage or calibration claim.
    """

    _scores: pd.DataFrame = field(repr=False)

    def __init__(self, points: PointEvaluationResult, scores: pd.DataFrame) -> None:
        table = _attachment_table(scores, _SCORE_COLUMNS)
        if table.duplicated([*_POINT_KEY, "uncertainty_method_id"]).any():
            raise ValueError("Duplicate score attachment identity.")
        _validate_point_links(points, table)
        valid = _mask(table["is_valid_score"], len(table), "is_valid_score")
        score = _float_vector(table["score"], "score")
        if np.any(valid & ~np.isfinite(score)):
            raise ValueError("Valid scores must be finite.")
        object.__setattr__(self, "_scores", table)

    @property
    def scores(self) -> pd.DataFrame:
        """Return an editable copy of the validated score rows."""
        return self._scores.copy(deep=True)
