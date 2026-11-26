import numpy as np
import pytest

from lsi.engine.svd_model import LSIModel
from lsi.engine.similarity import cosine, cosine_to_rows, euclidean_norm


@pytest.fixture(scope="module")
def matrix():
    rng = np.random.default_rng(42)
    return rng.random((30, 8))


@pytest.fixture(scope="module")
def model(matrix):
    return LSIModel.fit(matrix)


def test_orthogonality(model):
    left, right = model.orthogonality_error()
    assert left < 1e-10
    assert right < 1e-10


def test_singular_values_sorted_nonnegative(model):
    s = model.singular_values
    assert np.all(s >= 0)
    assert np.all(np.diff(s) <= 1e-12)


def test_full_rank_reconstruction(matrix, model):
    assert model.reconstruction_error(matrix, model.rank) < 1e-10


@pytest.mark.parametrize("k", [1, 2, 4, 7])
def test_truncation_error_matches_theory(matrix, model, k):
    assert model.reconstruction_error(matrix, k) == pytest.approx(model.theoretical_error(k))


def test_truncation_shapes(model):
    U_k, Sigma_k, Vt_k = model.truncate(3)
    assert U_k.shape == (30, 3)
    assert Sigma_k.shape == (3, 3)
    assert Vt_k.shape == (3, 8)


def test_energy_monotonic_and_complete(model):
    energies = [model.energy(k) for k in range(1, model.rank + 1)]
    assert all(b >= a for a, b in zip(energies, energies[1:]))
    assert energies[-1] == pytest.approx(100.0)


def test_invalid_k_rejected(model):
    with pytest.raises(ValueError):
        model.truncate(0)
    with pytest.raises(ValueError):
        model.truncate(model.rank + 1)


def test_projecting_a_document_column_gives_its_row_of_v(matrix, model):
    k = 3
    projected = model.project_query(matrix[:, 2], k)
    assert np.allclose(projected, model.document_vectors(k)[2])


def test_cosine_helpers():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 2.0])
    assert euclidean_norm(np.array([3.0, 4.0])) == pytest.approx(5.0)
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, b) == pytest.approx(0.0)
    assert cosine(np.zeros(2), a) == 0.0
    rows = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    assert np.allclose(cosine_to_rows(a, rows), [1.0, 0.0, 1 / np.sqrt(2)])
