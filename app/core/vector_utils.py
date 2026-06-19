import math


def normalize_vector(vec: list[float]) -> list[float]:
    if not vec:
        raise ValueError("Vector must not be empty")

    norm = math.sqrt(sum(value * value for value in vec))
    if norm == 0:
        raise ValueError("Vector norm must not be zero")

    return [value / norm for value in vec]
