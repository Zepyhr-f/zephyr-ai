import math

import pytest

from app.core.vector_utils import normalize_vector


def test_normalize_vector_returns_unit_vector() -> None:
    vector = normalize_vector([3.0, 4.0])

    assert vector == [0.6, 0.8]
    assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0)


def test_normalize_vector_rejects_empty_vector() -> None:
    with pytest.raises(ValueError, match="Vector must not be empty"):
        normalize_vector([])


def test_normalize_vector_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="Vector norm must not be zero"):
        normalize_vector([0.0, 0.0])
