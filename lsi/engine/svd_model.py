import numpy as np

from lsi import config
from lsi.schemas import SvdFactors


class LSIModel:
    def __init__(self, factors: SvdFactors):
        self.factors = factors

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "LSIModel":
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        return cls(SvdFactors(U=U, S=S, Vt=Vt))

    @property
    def singular_values(self) -> np.ndarray:
        return self.factors.S

    @property
    def shape(self) -> tuple[int, int]:
        return self.factors.U.shape[0], self.factors.Vt.shape[1]

    @property
    def rank(self) -> int:
        return int((self.factors.S > config.SVD_RANK_TOLERANCE).sum())

    def validate_k(self, k: int) -> int:
        if not 1 <= k <= self.rank:
            raise ValueError(f"k harus berada di antara 1 dan {self.rank}, diterima {k}")
        return k

    def truncate(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.validate_k(k)
        U_k = self.factors.U[:, :k]
        Sigma_k = np.diag(self.factors.S[:k])
        Vt_k = self.factors.Vt[:k, :]
        return U_k, Sigma_k, Vt_k

    def document_vectors(self, k: int) -> np.ndarray:
        return self.truncate(k)[2].T

    def project_query(self, q: np.ndarray, k: int) -> np.ndarray:
        U_k, Sigma_k, _ = self.truncate(k)
        return q @ U_k @ np.linalg.inv(Sigma_k)

    def reconstruct(self, k: int) -> np.ndarray:
        U_k, Sigma_k, Vt_k = self.truncate(k)
        return U_k @ Sigma_k @ Vt_k

    def reconstruction_error(self, matrix: np.ndarray, k: int) -> float:
        return float(np.linalg.norm(matrix - self.reconstruct(k), ord="fro"))

    def theoretical_error(self, k: int) -> float:
        return float(np.sqrt(np.sum(self.factors.S[k:] ** 2)))

    def orthogonality_error(self) -> tuple[float, float]:
        U, Vt = self.factors.U, self.factors.Vt
        eye = np.eye(U.shape[1])
        left = float(np.linalg.norm(U.T @ U - eye))
        right = float(np.linalg.norm(Vt @ Vt.T - eye))
        return left, right

    def energy(self, k: int) -> float:
        squared = self.factors.S ** 2
        return float(squared[:k].sum() / squared.sum() * 100.0)

    def cumulative_energy(self) -> np.ndarray:
        squared = self.factors.S ** 2
        return np.cumsum(squared) / squared.sum() * 100.0

    def dimension_report(self, k: int) -> dict[str, object]:
        m, n = self.shape
        self.validate_k(k)
        return {
            "A": (m, n),
            "U_k": (m, k),
            "Sigma_k": (k, k),
            "V_k^T": (k, n),
            "rank_r": self.rank,
            "energi_persen": round(self.energy(k), 2),
            "galat_frobenius": round(self.theoretical_error(k), 6),
        }
