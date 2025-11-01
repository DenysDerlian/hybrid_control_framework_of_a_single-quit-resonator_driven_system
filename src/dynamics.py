"""
Quantum dynamics simulation module (lab-frame I/Q only).

This module handles time evolution of the quantum system under two-quadrature
(I/Q) control fields in the lab frame. The driven Hamiltonian is:

    H(t) = H0 + Re{ Ω̃(t) e^{i ω_d t} } σx

with complex envelope Ω̃(t) = ΩI(t) - i ΩQ(t). This is equivalent to the
sin/cos decomposition but is more convenient for analysis and RWA.

where ΩI(t), ΩQ(t) are real-valued Fourier envelopes.
"""

from qutip import mesolve, Qobj
import numpy as np
from numba import njit
from typing import Optional, List, Dict, Any, Sequence, Tuple
from .config import USE_NUMBA_JIT, time_array, QUBIT_FREQUENCY, CAVITY_FREQUENCY, COUPLING_STRENGTH, NUM_QUBITS
from .operators import create_operators, construct_system_hamiltonian
from .states import get_initial_state
from .utils import split_iq_vector_nd

# Backward-compatibility wrappers (single-quadrature style)
def compute_fourier_control_field(time_points: np.ndarray,
                                  fourier_coefficients: np.ndarray,
                                  period: float,
                                  drive_frequency: float = QUBIT_FREQUENCY) -> np.ndarray:
    """Compose a real lab-frame control field from a Fourier parameter vector.

    This function reconstructs the control field including the carrier term ω_d:
    - Complex spectrum (symmetric bins):
        Ω̃(t) = Σ_k c_k e^{i 2π k t / T},  V(t) = Re{ Ω̃(t) e^{i ω_d t} }
    - Real cosine series (I-only envelope):
        e(t) = Σ_{k=0}^{N-1} c_k cos(2π k t / T), V(t) = e(t) cos(ω_d t)

    Here ω_d is expected in rad/s, and t in seconds.
    """
    t = np.asarray(time_points, dtype=float)
    c = np.asarray(fourier_coefficients)
    T = float(period)
    if T <= 0:
        raise ValueError("period must be > 0")

    if np.iscomplexobj(c):
        # Centered complex-exponential spectrum: k = -K..+K (length N)
        N = c.size
        center = N // 2
        k = np.arange(-center, N - center, dtype=float)
        phases = 2.0 * np.pi * np.outer(k, t / T)  # shape (N, len(t))
        envelope = np.sum(c[:, None] * np.exp(1j * phases), axis=0)  # Ω̃(t)
        carrier = np.exp(1j * (float(drive_frequency) * t))
        field = envelope * carrier
        return np.real(field).astype(float)

    # For backward compatibility ensure identical path to IQ helper when Q=0
    # Import locally to avoid circular import
    try:
        from .dynamics import compute_iq_control_fields  # self-import safe (already in module)
        fields = compute_iq_control_fields(t, np.real(c), np.zeros_like(c), T, float(drive_frequency))
        return fields['composite'].astype(float)
    except Exception:
        # Fallback simple cosine reconstruction
        k = np.arange(c.size, dtype=float)
        phases = 2.0 * np.pi * np.outer(k, t / T)
        envelope = np.sum(np.real(c)[:, None] * np.cos(phases), axis=0)
        field = envelope * np.cos(float(drive_frequency) * t)
        return field.astype(float)

def simulate_controlled_evolution(time_points: np.ndarray,
                                  fourier_coefficients: np.ndarray,
                                  drive_period: float,
                                  system_params: Dict[str, float],
                                  collapse_ops: List[Qobj],
                                  initial_state: Optional[Qobj] = None,
                                  measurement_ops: Optional[List[Qobj]] = None,
                                  drive_frequency: Optional[float] = None) -> Dict[str, Any]:
    """Compatibility wrapper that treats legacy coefficients as I-only (Q=zeros) and runs IQ sim."""
    if drive_frequency is None:
        drive_frequency = float(QUBIT_FREQUENCY)

    half = len(fourier_coefficients)//2
    cI, cQ = fourier_coefficients[:half], fourier_coefficients[half:]
    
    return simulate_controlled_evolution_iq(
        time_points,
        cI,
        cQ,
        drive_period,
        drive_frequency,
        system_params=system_params,
        initial_state=initial_state,
        measurement_ops=measurement_ops,
        collapse_ops=collapse_ops
    )

@njit
def _complex_exponential_sum(time_point: float,
                             coefficients: np.ndarray,
                             period: float) -> complex:
    """Compute Σ_k c_k exp(i 2π k t / T) with symmetric k indices."""
    coeff_sum = 0.0 + 0.0j
    num = len(coefficients)
    k = -num // 2
    twopi_over_T = 2.0 * np.pi / period
    for c in coefficients:
        coeff_sum += c * np.exp(1j * twopi_over_T * k * time_point)
        k += 1
    return coeff_sum

@njit
def _fourier_envelope_real(t: float, coefficients: np.ndarray, period: float) -> float:
    """Real-valued envelope from complex Fourier sum: Re[Σ_k c_k e^{i2πkt/T}]."""
    s = _complex_exponential_sum(t, coefficients, period)
    # Use .real manually to stay in nopython
    return float(np.real(s))

@njit
def _sigma_x_coeff_labframe(t: float,
                            coeffs_I: np.ndarray,
                            coeffs_Q: np.ndarray,
                            period: float,
                            drive_frequency: float) -> float:
    """Composite drive coefficient via equivalent exponential-sum form (no explicit cos/sin).

    Uses: 1/2 Re Σ_k (cI_k - i cQ_k)[e^{i(ω_d+ω_k)t} + e^{i(ω_d-ω_k)t}],
    where ω_k = 2π k / period for symmetric integer k indices.
    """
    twopi_over_T = 2.0 * np.pi / period
    accum = 0.0 + 0.0j
    num = len(coeffs_I)
    k = -num // 2
    for i in range(num):
        d = coeffs_I[i] - 1j * coeffs_Q[i]
        wk = twopi_over_T * k
        e_pos = np.exp(1j * (drive_frequency + wk) * t)
        e_neg = np.exp(1j * (drive_frequency - wk) * t)
        accum += d * (e_pos + e_neg)
        k += 1
    return float(np.real(0.5 * accum))

@njit
def _compute_iq_fields_arrays(time_points: np.ndarray,
                              coeffs_I: np.ndarray,
                              coeffs_Q: np.ndarray,
                              period: float,
                              drive_frequency: float) -> tuple:
    """Return (I_env, Q_env, composite) arrays.

    composite is computed as Re{ Ω̃(t) e^{i ω_d t} } in nopython mode.
    I_env and Q_env are provided for visualization as ΩI(t)cos and ΩQ(t)sin.
    """
    I_env = np.zeros_like(time_points, dtype=np.float64)
    Q_env = np.zeros_like(time_points, dtype=np.float64)
    comp = np.zeros_like(time_points, dtype=np.float64)
    for i in range(time_points.shape[0]):
        t = float(time_points[i])
        OmI = _fourier_envelope_real(t, coeffs_I, period)
        OmQ = _fourier_envelope_real(t, coeffs_Q, period)
        c = np.cos(drive_frequency * t)
        s = np.sin(drive_frequency * t)
        # Visualization components
        I_env[i] = OmI * c
        Q_env[i] = OmQ * s
        # Composite via exponential-sum form (equivalent to Re{ Ω̃ e^{i ω_d t} })
        comp[i] = _sigma_x_coeff_labframe(t, coeffs_I, coeffs_Q, period, drive_frequency)
    return I_env, Q_env, comp

# ------------------------------------------------------------------
# Lab-frame two-quadrature (IQ) control utilities
# ------------------------------------------------------------------

def compute_iq_control_fields(time_points: np.ndarray,
                              coeffs_I: np.ndarray,
                              coeffs_Q: np.ndarray,
                              period: float,
                              drive_frequency: float) -> Dict[str, np.ndarray]:
    """Compute lab-frame IQ control: ΩI(t)cos(ωd t) and ΩQ(t)sin(ωd t), and composite.

    Returns dict with keys 'I', 'Q', and 'composite'.
    """
    I_env, Q_env, comp = _compute_iq_fields_arrays(time_points.astype(np.float64),
                                                   coeffs_I.astype(np.float64),
                                                   coeffs_Q.astype(np.float64),
                                                   float(period), float(drive_frequency))
    return {'I': I_env, 'Q': Q_env, 'composite': comp}

def simulate_controlled_evolution_iq(time_points: np.ndarray,
                                     coeffs_I: np.ndarray,
                                     coeffs_Q: np.ndarray,
                                     drive_period: float,
                                     drive_frequency: float,
                                     system_params: Dict[str, float],
                                     collapse_ops: List[Qobj],
                                     initial_state: Optional[Qobj] = None,
                                     measurement_ops: Optional[List[Qobj]] = None) -> Dict[str, Any]:
    """Simulate lab-frame evolution with two-quadrature drive on σx.

    H(t) = H0 + Re{ Ω̃(t) e^{i ω_d t} } σx, with Ω̃(t) = ΩI(t) - i ΩQ(t).
    Envelopes ΩI, ΩQ are real-valued Fourier series with period T=drive_period.
    """
    if initial_state is None:
        initial_state = get_initial_state()
    if system_params is None:
        system_params = {
            'omega_r': CAVITY_FREQUENCY,
            'omega_t': QUBIT_FREQUENCY,
            'coupling_g': COUPLING_STRENGTH
        }
    ops = create_operators()
    if measurement_ops is None:
        measurement_ops = [
            ops['qubit_sigma_z'],
            ops['qubit_sigma_x'],
            ops['qubit_sigma_y'],
            ops['qubit_lowering'].dag() * ops['qubit_lowering'],
        ]
    # Bare system Hamiltonian (lab-frame JC already in construct_system_hamiltonian)
    H0 = construct_system_hamiltonian(system_params['omega_r'], system_params['omega_t'], system_params['coupling_g'])
    sx = ops['qubit_sigma_x']

    # Build callable coefficient for σx
    def coeff_sigma_x(t, _args=None):
        return _sigma_x_coeff_labframe(float(t), coeffs_I, coeffs_Q, float(drive_period), float(drive_frequency))

    H = [H0, [sx, coeff_sigma_x]]

    opts = {'store_states': True}
    if collapse_ops:
        res = mesolve(H, initial_state, time_points, c_ops=collapse_ops, e_ops=measurement_ops, options=opts)
    else:
        res = mesolve(H, initial_state, time_points, e_ops=measurement_ops, options=opts)

    final_state = None
    try:
        if getattr(res, 'states', None) and len(res.states) > 0:
            final_state = res.states[-1]
        elif hasattr(res, 'final_state') and res.final_state is not None:
            final_state = res.final_state
    except Exception:
        final_state = None

    # Also return control fields for analysis
    controls = compute_iq_control_fields(time_points, coeffs_I, coeffs_Q, drive_period, drive_frequency)
    return {
        'states': getattr(res, 'states', []),
        'expect': getattr(res, 'expect', []),
        'times': time_points,
        'final_state': final_state,
        'control_fields': controls,
    }

# ------------------------------------------------------------------
# Multi-qubit helpers (additive, backward-compatible additions)
# ------------------------------------------------------------------

def compute_iq_control_fields_multi(time_points: np.ndarray,
                                    coeffs_list: List[Tuple[np.ndarray, np.ndarray]],
                                    period: float,
                                    drive_frequencies: Sequence[float]) -> List[Dict[str, np.ndarray]]:
    """Compute per-qubit lab-frame IQ control fields.

    Returns a list of dicts [{ 'I': ..., 'Q': ..., 'composite': ... }, ...] per qubit.
    """
    out: List[Dict[str, np.ndarray]] = []
    for (cI, cQ), w in zip(coeffs_list, drive_frequencies):
        fields = compute_iq_control_fields(time_points, np.asarray(cI), np.asarray(cQ), float(period), float(w))
        out.append(fields)
    return out


def simulate_controlled_evolution_iq_multi(time_points: np.ndarray,
                                           coeff_vector: np.ndarray,
                                           drive_period: float,
                                           system_params: Dict[str, Any],
                                           collapse_ops: List[Qobj],
                                           drive_frequencies: Optional[Sequence[float]] = None,
                                           initial_state: Optional[Qobj] = None,
                                           measurement_ops: Optional[List[Qobj]] = None,
                                           num_qubits: int = NUM_QUBITS) -> Dict[str, Any]:
    """Multi-qubit lab-frame evolution under independent I/Q drives on each qubit's σx.

    coeff_vector packs per-qubit I/Q coefficients as [I0..., Q0..., I1..., Q1..., ...].
    """
    nq = int(num_qubits)
    if nq <= 0:
        raise ValueError("num_qubits must be positive")

    # System params and defaults
    sys_p = system_params or {}
    omega_r = float(sys_p.get('omega_r', CAVITY_FREQUENCY))
    omega_t = sys_p.get('omega_t', QUBIT_FREQUENCY)
    coupling_g = sys_p.get('coupling_g', COUPLING_STRENGTH)

    ops = create_operators(num_qubits=nq)
    H0 = construct_system_hamiltonian(omega_r, omega_t, coupling_g, num_qubits=nq)

    # Default drive frequencies to qubit ωt if not provided
    if drive_frequencies is None:
        if isinstance(omega_t, (list, tuple, np.ndarray)):
            drive_frequencies = list(map(float, omega_t))
        else:
            drive_frequencies = [float(omega_t)] * nq

    # Split coefficients per qubit
    coeffs_per_qubit = split_iq_vector_nd(np.asarray(coeff_vector), nq)

    # Build time-dependent Hamiltonian: sum_j f_j(t) σx_j
    H_list: List[Any] = [H0]
    for j in range(nq):
        sx_j = ops.get(f'qubit_sigma_x_{j}')
        if sx_j is None:
            raise KeyError(f"Missing operator qubit_sigma_x_{j}")
        cI_j, cQ_j = coeffs_per_qubit[j]
        wj = float(drive_frequencies[j])

        def coeff_j(t, _args=None, cI=cI_j, cQ=cQ_j, w=wj):
            return _sigma_x_coeff_labframe(float(t), np.asarray(cI), np.asarray(cQ), float(drive_period), float(w))

        H_list.append([sx_j, coeff_j])

    # Initial state
    psi0 = initial_state or get_initial_state(num_qubits=nq)

    # Measurement operators: default to None for nq>1 to save time; user can pass their own
    if measurement_ops is None and nq == 1:
        measurement_ops = [
            ops['qubit_sigma_z'],
            ops['qubit_sigma_x'],
            ops['qubit_sigma_y'],
            ops['qubit_lowering'].dag() * ops['qubit_lowering'],
        ]

    # Evolve
    if collapse_ops:
        res = mesolve(H_list, psi0, time_points, c_ops=collapse_ops, e_ops=measurement_ops, options={'store_states': True})
    else:
        res = mesolve(H_list, psi0, time_points, e_ops=measurement_ops, options={'store_states': True})

    # Controls per qubit
    fields_per_qubit = compute_iq_control_fields_multi(time_points, coeffs_per_qubit, float(drive_period), list(map(float, drive_frequencies)))

    final_state = None
    try:
        if getattr(res, 'states', None) and len(res.states) > 0:
            final_state = res.states[-1]
        elif hasattr(res, 'final_state') and res.final_state is not None:
            final_state = res.final_state
    except Exception:
        final_state = None

    return {
        'states': getattr(res, 'states', []),
        'expect': getattr(res, 'expect', []),
        'times': time_points,
        'final_state': final_state,
        'control_fields_per_qubit': fields_per_qubit,
    }
