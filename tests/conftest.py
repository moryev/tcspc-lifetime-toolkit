import numpy as np
import pytest
from numpy.typing import NDArray

from tcspc_toolkit.irf import (
    generate_gaussian_irf,
    normalize_irf,
)


@pytest.fixture
def time_axis() -> NDArray[np.float64]:
    return np.linspace(0.0, 10.0, 1001, dtype=np.float64)

@pytest.fixture
def irf(
    time_axis: NDArray[np.float64],
) -> NDArray[np.float64]:
    irf = generate_gaussian_irf(
        time=time_axis,
        centre=1.0,
        fwhm=0.2,
        amplitude=1.0,
    )

    irf = normalize_irf(
        time=time_axis,
        irf=irf,
    )

    return irf
