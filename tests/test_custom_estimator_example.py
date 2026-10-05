"""Runnable extension example through the curated root API."""

from pathlib import Path
import runpy

import numpy as np


def test_custom_estimator_example_returns_expected_prediction_array():
    path = Path(__file__).resolve().parents[1] / "examples" / "custom_estimator.py"
    example = runpy.run_path(str(path))
    prediction = example["run_example"]()
    assert isinstance(prediction, np.ndarray)
    assert prediction.dtype == np.float64
    assert prediction.shape == (2,)
    np.testing.assert_allclose(prediction, [1.5, 2.5])
