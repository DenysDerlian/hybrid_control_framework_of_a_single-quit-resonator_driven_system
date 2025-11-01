"""
Utility helpers for IQ (two-quadrature) parameter vectors.

Provides small ergonomic functions to build/split 2N vectors used by
IQ simulators and optimizers.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, List


def build_iq_vector(coeffs_I: np.ndarray, coeffs_Q: np.ndarray) -> np.ndarray:
    """Concatenate I and Q coefficient arrays into a single 2N vector.

    Arrays must have the same length.
    """
    if len(coeffs_I) != len(coeffs_Q):
        raise ValueError("I and Q arrays must have the same length")
    return np.concatenate([np.asarray(coeffs_I), np.asarray(coeffs_Q)])


essential_types = (list, tuple, np.ndarray)


def split_iq_vector(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a 2N vector into (I, Q) halves.

    If length is odd, raises ValueError.
    """
    vec = np.asarray(vec)
    if vec.ndim != 1:
        raise ValueError("vec must be 1D")
    if len(vec) % 2 != 0:
        raise ValueError("vec length must be even (2N)")
    half = len(vec) // 2
    return vec[:half], vec[half:]


# -----------------------------------------------------------------------------
# Multi-qubit IQ helpers (forward-compatible, non-breaking additions)
# -----------------------------------------------------------------------------

def build_iq_vector_nd(cI_list: List[np.ndarray], cQ_list: List[np.ndarray]) -> np.ndarray:
    """Pack per-qubit I/Q vectors into a single 1D vector of length 2N*n_qubits.

    Each entry in cI_list and cQ_list must be 1D arrays of equal length N.
    """
    if len(cI_list) != len(cQ_list):
        raise ValueError("cI_list and cQ_list must have same length (n_qubits)")
    if len(cI_list) == 0:
        return np.array([], dtype=float)
    N = len(cI_list[0])
    for idx, (ci, cq) in enumerate(zip(cI_list, cQ_list)):
        if len(ci) != len(cq):
            raise ValueError(f"I/Q length mismatch for qubit {idx}")
        if len(ci) != N:
            raise ValueError("All I/Q arrays must have the same length N across qubits")
    mats = [np.concatenate([np.asarray(ci), np.asarray(cq)]) for ci, cq in zip(cI_list, cQ_list)]
    return np.concatenate(mats, axis=0)


def split_iq_vector_nd(vec: np.ndarray, n_qubits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Unpack a concatenated 1D vector into per-qubit (I, Q) arrays.

    The input shape must be 2N*n_qubits. Returns a list of pairs (I, Q).
    If n_qubits==1, returns a singleton list consistent with split_iq_vector.
    """
    v = np.asarray(vec)
    if v.ndim != 1:
        raise ValueError("vec must be 1D")
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    if n_qubits == 1:
        I, Q = split_iq_vector(v)
        return [(I, Q)]
    if v.size % n_qubits != 0:
        raise ValueError("vec length must be divisible by n_qubits")
    twoN = v.size // n_qubits
    if twoN % 2 != 0:
        raise ValueError("per-qubit segment length must be even (2N)")
    N = twoN // 2
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for j in range(n_qubits):
        seg = v[j * twoN:(j + 1) * twoN]
        out.append((seg[:N], seg[N:]))
    return out
