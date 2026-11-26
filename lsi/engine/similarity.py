import numpy as np


def euclidean_norm(vector: np.ndarray) -> float:
    return float(np.sqrt(np.sum(vector * vector)))


def row_norms(matrix: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(matrix * matrix, axis=1))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = euclidean_norm(a) * euclidean_norm(b)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def cosine_to_rows(query: np.ndarray, rows: np.ndarray) -> np.ndarray:
    denominators = row_norms(rows) * euclidean_norm(query)
    dots = rows @ query
    scores = np.zeros(rows.shape[0])
    valid = denominators > 0
    scores[valid] = dots[valid] / denominators[valid]
    return scores
