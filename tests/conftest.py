import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def time_axis() -> NDArray[np.float64]:
    return np.linspace(0.0, 10.0, 1001, dtype=np.float64)
