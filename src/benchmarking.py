"""Randomized benchmarking style helpers (simplified) for single-qubit gate pulses.

This module provides utilities to empirically assess the average performance of an
optimized single-qubit gate pulse by examining the decay of sequence fidelities
for repeated application of the same gate implementation.

The simplified model here (NOT full RB) measures: for sequence length m, the
average state fidelity over a probe set S = {|0>, |1>, |+>, |+i>} between the
ideal U^m |psi> and the actual state produced by m sequential applications of
the optimized control pulse. We then fit F(m) ≈ A p^m + B to extract p which
heuristically captures average depolarizing parameter (smaller infidelity if p
close to 1).
"""
from __future__ import annotations
from typing import List, Sequence, Dict, Any, Optional, Tuple, Iterable
import numpy as np
from qutip import Qobj, basis, tensor, fidelity, qeye, sigmax, sigmay
from .optimization import (
    _single_qubit_gate_targets,
    _direct_final_state_simulation,
    _timestamp,
    _save_optimization_results,
)
from .config import NUM_FOCK_STATES, time_array, QUBIT_FREQUENCY
from .dynamics import simulate_controlled_evolution_iq
from scipy.optimize import least_squares, minimize

ProbeStateSet = List[Qobj]


def default_probe_states() -> ProbeStateSet:
    """Return the default set of probe states: |0>, |1>, |+>, |+i> (cavity vacuum)."""
    cav0 = basis(NUM_FOCK_STATES, 0)
    q0 = basis(2, 0)
    q1 = basis(2, 1)
    psi_0 = tensor(cav0, q0)
    psi_1 = tensor(cav0, q1)
    psi_plus = (tensor(cav0, (q0 + q1)) / np.sqrt(2)).unit()
    psi_plus_i = (tensor(cav0, (q0 + 1j * q1)) / np.sqrt(2)).unit()
    return [psi_0, psi_1, psi_plus, psi_plus_i]


def _apply_control_once(fourier_coeffs: np.ndarray,
                        drive_period: float,
                        initial_state: Qobj,
                        system_params: Dict[str, float],
                        collapse_ops: List[Qobj],
                        drive_frequency: Optional[float] = None) -> Qobj:
    """Apply the optimized control once; supports single or IQ coefficients."""
    # Merge provided system params with defaults to ensure required keys exist
    from .config import CAVITY_FREQUENCY, QUBIT_FREQUENCY as _WF, COUPLING_STRENGTH
    sys_params: Dict[str, float] = {
        'omega_r': float(CAVITY_FREQUENCY),
        'omega_t': float(_WF),
        'coupling_g': float(COUPLING_STRENGTH),
    }
    if system_params:
        sys_params.update({k: float(v) for k, v in system_params.items() if k in sys_params})

    # Split 2N -> (I, Q)
    half = len(fourier_coeffs) // 2
    cI, cQ = fourier_coeffs[:half], fourier_coeffs[half:]
    if drive_frequency is None:
        drive_frequency = float(QUBIT_FREQUENCY)
    result = simulate_controlled_evolution_iq(
        time_array, cI, cQ, drive_period, float(drive_frequency),
        system_params=sys_params, collapse_ops=collapse_ops, initial_state=initial_state
    )

    return result['final_state']


def simulate_repeated_gate(fourier_coeffs: np.ndarray,
                           drive_period: float,
                           gate_name: str,
                           repetitions: int,
                           initial_state: Qobj,
                           system_params: Dict[str, float],
                           collapse_ops: List[Qobj],
                           drive_frequency: Optional[float] = None) -> Tuple[Qobj, Qobj]:
    """Simulate m sequential applications of a single gate control.

    Returns (final_state, ideal_target_state) where ideal_target_state = U^m |psi0>.
    """
    # Accept shorthand aliases 's' -> 's_gate', 't' -> 't_gate'
    if gate_name == 's':
        gate_name = 's_gate'
    elif gate_name == 't':
        gate_name = 't_gate'
    gates = _single_qubit_gate_targets()
    if gate_name not in gates:
        raise ValueError(f"Unknown gate '{gate_name}'")
    U = gates[gate_name]
    state = initial_state
    for _ in range(repetitions):
        state = _apply_control_once(fourier_coeffs, drive_period, state, system_params, 
                                    collapse_ops, drive_frequency=drive_frequency)
    ideal = (U ** repetitions) * initial_state
    return state, ideal


def sequence_fidelity_scan(system_params: Dict[str, float],
                           collapse_ops: List[Qobj],
                           optimized_payload: Dict[str, Any],
                           gate_name: str,
                           sequence_lengths: Sequence[int],
                           probe_states: Optional[ProbeStateSet] = None) -> Dict[str, Any]:
    """Compute average probe fidelity for multiple sequence lengths.

    optimized_payload: Result dict from optimize_* (expects 'optimization_result' & 'control_period').
    gate_name: Name of gate ('hadamard', 's_gate', 't_gate').
    sequence_lengths: Iterable of m values (positive ints).
    probe_states: Optional custom probe state list.
    Returns dict with arrays 'm', 'avg_fidelity', 'per_probe_fidelity'.
    """
    if probe_states is None:
        probe_states = default_probe_states()
    coeffs = optimized_payload['optimization_result'].x
    period = optimized_payload.get('control_period')
    # Detect IQ mode
    is_iq = bool(optimized_payload.get('process_cost') == 'ptm_iq') or (
        'drive_frequency' in optimized_payload and len(coeffs) % 2 == 0
    )
    drive_freq = float(optimized_payload.get('drive_frequency', 0.0)) if is_iq else None
    if period is None:
        raise ValueError("optimized_payload missing 'control_period'")
    m_vals = []
    avg_fids = []
    per_probe = []  # shape (len(sequence_lengths), num_probes)
    for m in sequence_lengths:
        fids_this_m = []
        for psi in probe_states:
            final_state, ideal_state = simulate_repeated_gate(
                coeffs, period, gate_name, m, psi, system_params, 
                collapse_ops, drive_frequency=drive_freq
            )
            if final_state is None:
                fids_this_m.append(0.0)
            else:
                fids_this_m.append(float(fidelity(ideal_state, final_state)))
        m_vals.append(m)
        per_probe.append(fids_this_m)
        avg_fids.append(np.mean(fids_this_m))
    return {
        'm': np.array(m_vals, dtype=int),
        'avg_fidelity': np.array(avg_fids, dtype=float),
        'per_probe_fidelity': np.array(per_probe, dtype=float),
        'num_probes': len(probe_states),
    'gate_name': gate_name,
    'is_iq': is_iq,
    'drive_frequency': drive_freq
    }


def fit_exponential_decay(m: np.ndarray, Fm: np.ndarray) -> Dict[str, float]:
    """Fit F(m) ≈ A * p^m + B using robust bounded least squares with A + B ≤ 1.

    Reparameterization: A = (1 - B) * a_hat, with a_hat ∈ [0,1], B ∈ [0,1], p ∈ [0,1].
    Returns a dict with keys 'A', 'p', 'B'.
    """
    m = np.asarray(m, dtype=float)
    Fm = np.asarray(Fm, dtype=float)

    if m.shape != Fm.shape:
        raise ValueError("m and Fm length mismatch")
    n = m.size
    if n == 0:
        raise ValueError("empty inputs")

    # Clamp to [0,1] for stability
    Fm = np.clip(Fm, 0.0, 1.0)

    # Initial B: median of tail points (robust)
    k_tail = max(3, n // 4)
    B0 = float(np.median(Fm[-k_tail:]))

    # Log-linear initialization where positive
    F_adj = Fm - B0
    pos_mask = F_adj > 1e-10
    if np.count_nonzero(pos_mask) >= 3 and (np.ptp(m[pos_mask]) > 0):
        y = np.log(F_adj[pos_mask])
        x = m[pos_mask]
        A1 = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A1, y, rcond=None)[0]
        p0 = float(np.clip(np.exp(slope), 1e-6, 1.0))
        A0 = float(np.exp(intercept))
    else:
        # Two-point heuristic fallback
        B0 = float(np.min(Fm))
        A0 = float(max(Fm[0] - B0, 1e-6))
        dm = max(1.0, float(m[-1] - m[0]))
        ratio = float(max(Fm[-1] - B0, 1e-12) / A0)
        p0 = float(np.clip(ratio ** (1.0 / dm), 1e-6, 1.0))

    # Map to a_hat with A = (1 - B) * a_hat
    one_minus_B0 = max(1e-6, 1.0 - B0)
    a_hat0 = float(np.clip(A0 / one_minus_B0, 1e-6, 1.0))

    def model_vars(v: np.ndarray) -> np.ndarray:
        a_hat, p, B = v
        return (1.0 - B) * a_hat * (p ** m) + B

    def residuals(v: np.ndarray) -> np.ndarray:
        return model_vars(v) - Fm

    try:
        res = least_squares(
            residuals,
            x0=np.array([a_hat0, p0, np.clip(B0, 0.0, 1.0)], dtype=float),
            bounds=(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
            loss='soft_l1',
            f_scale=0.02,
        )
        a_hat_fit, p_fit, B_fit = res.x
    except Exception:
        # Fallback to SSE minimize
        try:
            def sse(v):
                r = residuals(v)
                return float(np.sum(r * r))
            res = minimize(sse, x0=[a_hat0, p0, B0], bounds=[(0, 1), (0, 1), (0, 1)])
            a_hat_fit, p_fit, B_fit = res.x
        except Exception:
            a_hat_fit, p_fit, B_fit = a_hat0, p0, B0

    # Recover A, enforce bounds
    B_fit = float(np.clip(B_fit, 0.0, 1.0))
    A_fit = float(np.clip((1.0 - B_fit) * a_hat_fit, 0.0, 1.0 - B_fit))
    p_fit = float(np.clip(p_fit, 0.0, 1.0))

    return {'A': A_fit, 'p': p_fit, 'B': B_fit}


def benchmarking_summary(scan_result: Dict[str, Any], fit_params: Dict[str, float]) -> str:
    """Return human-readable summary string for a sequence fidelity scan & fit."""
    lines = [
        f"Gate: {scan_result['gate_name']}",
        f"Probe states: {scan_result['num_probes']}",
        f"Sequence lengths tested: {scan_result['m'].tolist()}",
        f"Average fidelities: {[f'{v:.4f}' for v in scan_result['avg_fidelity']]}",
        f"Fit: F(m) ≈ {fit_params['A']:.4f} * {fit_params['p']:.4f}^m + {fit_params['B']:.4f}",
        f"Approx per-gate infidelity ~ 1 - p = {1.0 - fit_params['p']:.4e}"
    ]
    return "\n".join(lines)


def _normalize_gate_name_from_target_type(target_type: str) -> str:
    """Infer canonical gate name from optimization payload target_type.

    Examples:
      'hadamard_gate' -> 'hadamard'
      's_gate' -> 's_gate'
      't_gate' -> 't_gate'
      'hadamard_process' -> 'hadamard'
    """
    if target_type.endswith('_gate'):
        base = target_type.replace('_gate', '')
        # Preserve phase gate names and 90-degree variants
        if base in ('s', 't', 'x90', 'y90'):
            return f"{base}_gate" if base in ('s', 't') else f"{base}_gate"
        # For x/y return canonical with suffix as used elsewhere
        if base in ('x', 'y'):
            return f"{base}_gate"
        return base
    if target_type.endswith('_process'):
        return target_type.replace('_process', '')
    return target_type


def _canonical_gate_name(name: str) -> str:
    """Return a canonical gate identifier accepted by sequence_fidelity_scan.

    Maps common aliases like 'x' -> 'x_gate', 'x90' -> 'x90_gate', etc.
    """
    if not name:
        return name
    n = name.lower()
    alias_map = {
        'x': 'x_gate', 'x_gate': 'x_gate',
        'y': 'y_gate', 'y_gate': 'y_gate',
        'x90': 'x90_gate', 'x90_gate': 'x90_gate',
        'y90': 'y90_gate', 'y90_gate': 'y90_gate',
        's': 's_gate', 's_gate': 's_gate',
        't': 't_gate', 't_gate': 't_gate',
        'hadamard': 'hadamard',
    }
    return alias_map.get(n, n)


def run_sequence_benchmark(system_params: Dict[str, float],
                           collapse_ops: List[Qobj],
                           optimized_payload: Dict[str, Any],
                           gate_name: Optional[str] = None,
                           sequence_lengths: Iterable[int] = (1, 2, 3, 4, 5, 6),
                           probe_states: Optional[ProbeStateSet] = None,
                           fit: bool = True,
                           save: bool = True,
                           results_dir: str = 'results') -> Dict[str, Any]:
    """Full workflow helper: take an optimization payload, compute sequence fidelities, fit decay, and save.

    Parameters
    ----------
    optimized_payload : dict
        Result dict returned by optimize_* functions (must contain 'optimization_result').
    gate_name : str, optional
        Gate name; if None will be inferred from optimized_payload['target_type'].
    sequence_lengths : iterable[int]
        Sequence lengths m to evaluate (positive integers).
    probe_states : list[Qobj], optional
        Custom probe states; defaults to default_probe_states().
    fit : bool
        Whether to perform exponential decay fit.
    save : bool
        If True, persist benchmark metadata and arrays (JSON + NPZ) under results/.
    results_dir : str
        Directory to store results.
    system_params : dict, optional
        System parameter overrides for evolution.

    Returns
    -------
    dict
        Benchmark payload containing scan, (optional) fit, and pulse metadata.
    """
    if gate_name is None:
        gate_name = _normalize_gate_name_from_target_type(optimized_payload.get('target_type', ''))
    gate_name = _canonical_gate_name(gate_name)
    if not gate_name:
        raise ValueError("Unable to infer gate_name; please provide it explicitly.")
    # Ensure sequence lengths are sorted unique ints
    seq = sorted({int(m) for m in sequence_lengths if int(m) > 0})
    if len(seq) == 0:
        raise ValueError("sequence_lengths must contain at least one positive integer")

    # Route to specialized helpers for x/y and 90-degree gates to compare against correct unitaries
    if gate_name in ('x_gate', 'y_gate', 'x90_gate', 'y90_gate'):
        if gate_name == 'x_gate':
            scan_result = sequence_fidelity_scan_x(system_params, collapse_ops, optimized_payload, seq, probe_states)
        elif gate_name == 'y_gate':
            scan_result = sequence_fidelity_scan_y(system_params, collapse_ops, optimized_payload, seq, probe_states)
        elif gate_name == 'x90_gate':
            scan_result = sequence_fidelity_scan_x90(system_params, collapse_ops, optimized_payload, seq, probe_states)
        else:
            scan_result = sequence_fidelity_scan_y90(system_params, collapse_ops, optimized_payload, seq, probe_states)
    else:
        scan_result = sequence_fidelity_scan(system_params, collapse_ops, optimized_payload, gate_name, seq, probe_states)
    fit_params = fit_exponential_decay(scan_result['m'], scan_result['avg_fidelity']) if fit else None

    # Pulse / optimization metadata extraction
    opt_res = optimized_payload['optimization_result']
    pulse_metadata = {
        'fourier_modes': optimized_payload.get('fourier_modes'),
        'control_period': optimized_payload.get('control_period'),
        'max_iterations': optimized_payload.get('max_iterations'),
        'coefficient_bounds': optimized_payload.get('coefficient_bounds'),
        'leakage_weight': optimized_payload.get('leakage_weight'),
        'l2_regularization': optimized_payload.get('l2_regularization'),
        'optimization_cost': float(getattr(opt_res, 'fun', np.nan)),
        'optimization_success': bool(getattr(opt_res, 'success', False)),
        'optimization_nfev': int(getattr(opt_res, 'nfev', -1)),
        'optimization_message': str(getattr(opt_res, 'message', '')),
        'optimization_save_base_name': optimized_payload.get('save_base_name'),
    }

    payload: Dict[str, Any] = {
        'benchmark_type': 'sequence_fidelity_scan',
        'gate_name': gate_name,
        'scan_result': scan_result,
        'fit_params': fit_params,
        'pulse_metadata': pulse_metadata,
        'timestamp': _timestamp(),
    }

    if save:
        base_opt_name = optimized_payload.get('save_base_name', 'unsaved_opt')
        base_name = f"rb_{gate_name}_from={base_opt_name}_mmin={seq[0]}_mmax={seq[-1]}_{_timestamp()}"
        # Build meta JSON (exclude large arrays)
        meta = {
            'benchmark_type': 'sequence_fidelity_scan',
            'gate_name': gate_name,
            'sequence_lengths': scan_result['m'].tolist(),
            'avg_fidelity': [float(x) for x in scan_result['avg_fidelity']],
            'num_probes': scan_result['num_probes'],
            'fit_params': fit_params if fit_params else None,
            'pulse_metadata': pulse_metadata,
            'timestamp': payload['timestamp']
        }
        arrays = {
            'm': scan_result['m'],
            'avg_fidelity': scan_result['avg_fidelity'],
            'per_probe_fidelity': scan_result['per_probe_fidelity'],
            'optimal_coefficients': opt_res.x,  # snapshot of pulse used
        }
        saved_paths = _save_optimization_results(base_name, meta, arrays, results_dir=results_dir)
        payload['saved_paths'] = saved_paths
        payload['save_base_name'] = base_name

    return payload


# =============================
# X and Y gate benchmarking
# =============================

def _simulate_repeated_gate_with_qubit_unitary(fourier_coeffs: np.ndarray,
                                               drive_period: float,
                                               U_qubit: Qobj,
                                               repetitions: int,
                                               initial_state: Qobj,
                                               system_params: Dict[str, float],
                                               collapse_ops: List[Qobj],
                                               drive_frequency: Optional[float] = None) -> Tuple[Qobj, Qobj]:
    """Simulate m sequential applications of a gate using a provided 2x2 qubit unitary."""
    U = tensor(qeye(NUM_FOCK_STATES), U_qubit)
    state = initial_state
    for _ in range(repetitions):
        state = _apply_control_once(fourier_coeffs, drive_period, state, 
                                    system_params, collapse_ops, drive_frequency=drive_frequency)
    ideal = (U ** repetitions) * initial_state
    return state, ideal


def sequence_fidelity_scan_x(system_params: Dict[str, float],
                             collapse_ops: List[Qobj],
                             optimized_payload: Dict[str, Any],
                             sequence_lengths: Sequence[int],
                             probe_states: Optional[ProbeStateSet] = None) -> Dict[str, Any]:
    """Compute average probe fidelity for multiple sequence lengths for X gate (π around X)."""
    if probe_states is None:
        probe_states = default_probe_states()
    coeffs = optimized_payload['optimization_result'].x
    period = optimized_payload.get('control_period')
    is_iq = bool(optimized_payload.get('process_cost') == 'ptm_iq') or (
        'drive_frequency' in optimized_payload and len(coeffs) % 2 == 0
    )
    drive_freq = float(optimized_payload.get('drive_frequency', 0.0)) if is_iq else None
    if period is None:
        raise ValueError("optimized_payload missing 'control_period'")
    Uq = sigmax()
    m_vals, avg_fids, per_probe = [], [], []
    for m in sequence_lengths:
        fids_this_m = []
        for psi in probe_states:
            final_state, ideal_state = _simulate_repeated_gate_with_qubit_unitary(
                coeffs, period, Uq, m, psi, system_params, collapse_ops, drive_frequency=drive_freq
            )
            if final_state is None:
                fids_this_m.append(0.0)
            else:
                fids_this_m.append(float(fidelity(ideal_state, final_state)))
        m_vals.append(m)
        per_probe.append(fids_this_m)
        avg_fids.append(np.mean(fids_this_m))
    return {
        'm': np.array(m_vals, dtype=int),
        'avg_fidelity': np.array(avg_fids, dtype=float),
        'per_probe_fidelity': np.array(per_probe, dtype=float),
        'num_probes': len(probe_states),
        'gate_name': 'x_gate',
        'is_iq': is_iq,
        'drive_frequency': drive_freq,
    }


def sequence_fidelity_scan_y(system_params: Dict[str, float],
                             collapse_ops: List[Qobj],
                             optimized_payload: Dict[str, Any],
                             sequence_lengths: Sequence[int],
                             probe_states: Optional[ProbeStateSet] = None) -> Dict[str, Any]:
    """Compute average probe fidelity for multiple sequence lengths for Y gate (π around Y)."""
    if probe_states is None:
        probe_states = default_probe_states()
    coeffs = optimized_payload['optimization_result'].x
    period = optimized_payload.get('control_period')
    is_iq = bool(optimized_payload.get('process_cost') == 'ptm_iq') or (
        'drive_frequency' in optimized_payload and len(coeffs) % 2 == 0
    )
    drive_freq = float(optimized_payload.get('drive_frequency', 0.0)) if is_iq else None
    if period is None:
        raise ValueError("optimized_payload missing 'control_period'")
    Uq = sigmay()
    m_vals, avg_fids, per_probe = [], [], []
    for m in sequence_lengths:
        fids_this_m = []
        for psi in probe_states:
            final_state, ideal_state = _simulate_repeated_gate_with_qubit_unitary(
                coeffs, period, Uq, m, psi, system_params, collapse_ops, drive_frequency=drive_freq
            )
            if final_state is None:
                fids_this_m.append(0.0)
            else:
                fids_this_m.append(float(fidelity(ideal_state, final_state)))
        m_vals.append(m)
        per_probe.append(fids_this_m)
        avg_fids.append(np.mean(fids_this_m))
    return {
        'm': np.array(m_vals, dtype=int),
        'avg_fidelity': np.array(avg_fids, dtype=float),
        'per_probe_fidelity': np.array(per_probe, dtype=float),
        'num_probes': len(probe_states),
        'gate_name': 'y_gate',
        'is_iq': is_iq,
        'drive_frequency': drive_freq,
    }


def run_sequence_benchmark_x(system_params: Dict[str, float],
                             collapse_ops: List[Qobj],
                             optimized_payload: Dict[str, Any],
                             sequence_lengths: Iterable[int] = (1, 2, 3, 4, 5, 6),
                             probe_states: Optional[ProbeStateSet] = None,
                             fit: bool = True,
                             save: bool = True,
                             results_dir: str = 'results') -> Dict[str, Any]:
    """Run sequence fidelity benchmark for X gate using optimized payload."""
    # Ensure sequence lengths are sorted unique ints
    seq = sorted({int(m) for m in sequence_lengths if int(m) > 0})
    if len(seq) == 0:
        raise ValueError("sequence_lengths must contain at least one positive integer")

    scan_result = sequence_fidelity_scan_x(system_params, collapse_ops, optimized_payload, seq, probe_states)
    fit_params = fit_exponential_decay(scan_result['m'], scan_result['avg_fidelity']) if fit else None

    opt_res = optimized_payload['optimization_result']
    pulse_metadata = {
        'fourier_modes': optimized_payload.get('fourier_modes'),
        'control_period': optimized_payload.get('control_period'),
        'max_iterations': optimized_payload.get('max_iterations'),
        'coefficient_bounds': optimized_payload.get('coefficient_bounds'),
        'leakage_weight': optimized_payload.get('leakage_weight'),
        'l2_regularization': optimized_payload.get('l2_regularization'),
        'optimization_cost': float(getattr(opt_res, 'fun', np.nan)),
        'optimization_success': bool(getattr(opt_res, 'success', False)),
        'optimization_nfev': int(getattr(opt_res, 'nfev', -1)),
        'optimization_message': str(getattr(opt_res, 'message', '')),
        'optimization_save_base_name': optimized_payload.get('save_base_name'),
    }

    payload: Dict[str, Any] = {
        'benchmark_type': 'sequence_fidelity_scan',
        'gate_name': 'x_gate',
        'scan_result': scan_result,
        'fit_params': fit_params,
        'pulse_metadata': pulse_metadata,
        'timestamp': _timestamp(),
    }

    if save:
        base_opt_name = optimized_payload.get('save_base_name', 'unsaved_opt')
        base_name = f"rb_x_gate_from={base_opt_name}_mmin={seq[0]}_mmax={seq[-1]}_{_timestamp()}"
        meta = {
            'benchmark_type': 'sequence_fidelity_scan',
            'gate_name': 'x_gate',
            'sequence_lengths': scan_result['m'].tolist(),
            'avg_fidelity': [float(x) for x in scan_result['avg_fidelity']],
            'num_probes': scan_result['num_probes'],
            'fit_params': fit_params if fit_params else None,
            'pulse_metadata': pulse_metadata,
            'timestamp': payload['timestamp']
        }
        arrays = {
            'm': scan_result['m'],
            'avg_fidelity': scan_result['avg_fidelity'],
            'per_probe_fidelity': scan_result['per_probe_fidelity'],
            'optimal_coefficients': opt_res.x,
        }
        saved_paths = _save_optimization_results(base_name, meta, arrays, results_dir=results_dir)
        payload['saved_paths'] = saved_paths
        payload['save_base_name'] = base_name

    return payload


def run_sequence_benchmark_y(system_params: Dict[str, float],
                             collapse_ops: List[Qobj],
                             optimized_payload: Dict[str, Any],
                             sequence_lengths: Iterable[int] = (1, 2, 3, 4, 5, 6),
                             probe_states: Optional[ProbeStateSet] = None,
                             fit: bool = True,
                             save: bool = True,
                             results_dir: str = 'results') -> Dict[str, Any]:
    """Run sequence fidelity benchmark for Y gate using optimized payload."""
    seq = sorted({int(m) for m in sequence_lengths if int(m) > 0})
    if len(seq) == 0:
        raise ValueError("sequence_lengths must contain at least one positive integer")

    scan_result = sequence_fidelity_scan_y(system_params, collapse_ops, optimized_payload, seq, probe_states)
    fit_params = fit_exponential_decay(scan_result['m'], scan_result['avg_fidelity']) if fit else None

    opt_res = optimized_payload['optimization_result']
    pulse_metadata = {
        'fourier_modes': optimized_payload.get('fourier_modes'),
        'control_period': optimized_payload.get('control_period'),
        'max_iterations': optimized_payload.get('max_iterations'),
        'coefficient_bounds': optimized_payload.get('coefficient_bounds'),
        'leakage_weight': optimized_payload.get('leakage_weight'),
        'l2_regularization': optimized_payload.get('l2_regularization'),
        'optimization_cost': float(getattr(opt_res, 'fun', np.nan)),
        'optimization_success': bool(getattr(opt_res, 'success', False)),
        'optimization_nfev': int(getattr(opt_res, 'nfev', -1)),
        'optimization_message': str(getattr(opt_res, 'message', '')),
        'optimization_save_base_name': optimized_payload.get('save_base_name'),
    }

    payload: Dict[str, Any] = {
        'benchmark_type': 'sequence_fidelity_scan',
        'gate_name': 'y_gate',
        'scan_result': scan_result,
        'fit_params': fit_params,
        'pulse_metadata': pulse_metadata,
        'timestamp': _timestamp(),
    }

    if save:
        base_opt_name = optimized_payload.get('save_base_name', 'unsaved_opt')
        base_name = f"rb_y_gate_from={base_opt_name}_mmin={seq[0]}_mmax={seq[-1]}_{_timestamp()}"
        meta = {
            'benchmark_type': 'sequence_fidelity_scan',
            'gate_name': 'y_gate',
            'sequence_lengths': scan_result['m'].tolist(),
            'avg_fidelity': [float(x) for x in scan_result['avg_fidelity']],
            'num_probes': scan_result['num_probes'],
            'fit_params': fit_params if fit_params else None,
            'pulse_metadata': pulse_metadata,
            'timestamp': payload['timestamp']
        }
        arrays = {
            'm': scan_result['m'],
            'avg_fidelity': scan_result['avg_fidelity'],
            'per_probe_fidelity': scan_result['per_probe_fidelity'],
            'optimal_coefficients': opt_res.x,
        }
        saved_paths = _save_optimization_results(base_name, meta, arrays, results_dir=results_dir)
        payload['saved_paths'] = saved_paths
        payload['save_base_name'] = base_name

    return payload


# =============================
# √X and √Y gate benchmarking (π/2 rotations)
# =============================

def _rotation_unitary(axis: str, theta: float) -> Qobj:
    """Return 2x2 unitary for rotation R_axis(theta) = exp(-i theta σ_axis / 2)."""
    from qutip import qeye as _qeye, sigmax as _sx, sigmay as _sy, sigmaz as _sz
    import numpy as _np
    c = _np.cos(theta / 2.0)
    s = _np.sin(theta / 2.0)
    if axis.lower() == 'x':
        return c * _qeye(2) - 1j * s * _sx()
    if axis.lower() == 'y':
        return c * _qeye(2) - 1j * s * _sy()
    if axis.lower() == 'z':
        return c * _qeye(2) - 1j * s * _sz()
    raise ValueError("axis must be one of {'x','y','z'}")


def sequence_fidelity_scan_x90(system_params: Dict[str, float],
                               collapse_ops: List[Qobj],
                               optimized_payload: Dict[str, Any],
                               sequence_lengths: Sequence[int],
                               probe_states: Optional[ProbeStateSet] = None) -> Dict[str, Any]:
    """Average probe fidelity for √X gate (RX(π/2)) across sequence lengths."""
    if probe_states is None:
        probe_states = default_probe_states()
    coeffs = optimized_payload['optimization_result'].x
    period = optimized_payload.get('control_period')
    is_iq = bool(optimized_payload.get('process_cost') == 'ptm_iq') or (
        'drive_frequency' in optimized_payload and len(coeffs) % 2 == 0
    )
    drive_freq = float(optimized_payload.get('drive_frequency', 0.0)) if is_iq else None
    if period is None:
        raise ValueError("optimized_payload missing 'control_period'")
    Uq = _rotation_unitary('x', np.pi/2.0)
    U = tensor(qeye(NUM_FOCK_STATES), Uq)
    m_vals, avg_fids, per_probe = [], [], []
    for m in sequence_lengths:
        fids_this_m = []
        for psi in probe_states:
            state = psi
            for _ in range(m):
                state = _apply_control_once(coeffs, period, state, 
                                            system_params, collapse_ops, drive_frequency=drive_freq)
            ideal = (U ** m) * psi
            fids_this_m.append(0.0 if state is None else float(fidelity(ideal, state)))
        m_vals.append(m)
        per_probe.append(fids_this_m)
        avg_fids.append(np.mean(fids_this_m))
    return {
        'm': np.array(m_vals, dtype=int),
        'avg_fidelity': np.array(avg_fids, dtype=float),
        'per_probe_fidelity': np.array(per_probe, dtype=float),
        'num_probes': len(probe_states),
        'gate_name': 'x90_gate',
        'is_iq': is_iq,
        'drive_frequency': drive_freq,
    }


def sequence_fidelity_scan_y90(system_params: Dict[str, float],
                               collapse_ops: List[Qobj],
                               optimized_payload: Dict[str, Any],
                               sequence_lengths: Sequence[int],
                               probe_states: Optional[ProbeStateSet] = None) -> Dict[str, Any]:
    """Average probe fidelity for √Y gate (RY(π/2)) across sequence lengths."""
    if probe_states is None:
        probe_states = default_probe_states()
    coeffs = optimized_payload['optimization_result'].x
    period = optimized_payload.get('control_period')
    is_iq = bool(optimized_payload.get('process_cost') == 'ptm_iq') or (
        'drive_frequency' in optimized_payload and len(coeffs) % 2 == 0
    )
    drive_freq = float(optimized_payload.get('drive_frequency', 0.0)) if is_iq else None
    if period is None:
        raise ValueError("optimized_payload missing 'control_period'")
    Uq = _rotation_unitary('y', np.pi/2.0)
    U = tensor(qeye(NUM_FOCK_STATES), Uq)
    m_vals, avg_fids, per_probe = [], [], []
    for m in sequence_lengths:
        fids_this_m = []
        for psi in probe_states:
            state = psi
            for _ in range(m):
                state = _apply_control_once(coeffs, period, state, 
                                            system_params, collapse_ops, drive_frequency=drive_freq)
            ideal = (U ** m) * psi
            fids_this_m.append(0.0 if state is None else float(fidelity(ideal, state)))
        m_vals.append(m)
        per_probe.append(fids_this_m)
        avg_fids.append(np.mean(fids_this_m))
    return {
        'm': np.array(m_vals, dtype=int),
        'avg_fidelity': np.array(avg_fids, dtype=float),
        'per_probe_fidelity': np.array(per_probe, dtype=float),
        'num_probes': len(probe_states),
        'gate_name': 'y90_gate',
        'is_iq': is_iq,
        'drive_frequency': drive_freq,
    }


def run_sequence_benchmark_x90(system_params: Dict[str, float],
                               collapse_ops: List[Qobj],
                               optimized_payload: Dict[str, Any],
                               sequence_lengths: Iterable[int] = (1, 2, 3, 4, 5, 6),
                               probe_states: Optional[ProbeStateSet] = None,
                               fit: bool = True,
                               save: bool = True,
                               results_dir: str = 'results') -> Dict[str, Any]:
    """Run sequence fidelity benchmark for √X gate using optimized payload."""
    seq = sorted({int(m) for m in sequence_lengths if int(m) > 0})
    if len(seq) == 0:
        raise ValueError("sequence_lengths must contain at least one positive integer")
    scan_result = sequence_fidelity_scan_x90(system_params, collapse_ops, optimized_payload, seq, probe_states)
    fit_params = fit_exponential_decay(scan_result['m'], scan_result['avg_fidelity']) if fit else None

    opt_res = optimized_payload['optimization_result']
    pulse_metadata = {
        'fourier_modes': optimized_payload.get('fourier_modes'),
        'control_period': optimized_payload.get('control_period'),
        'max_iterations': optimized_payload.get('max_iterations'),
        'coefficient_bounds': optimized_payload.get('coefficient_bounds'),
        'leakage_weight': optimized_payload.get('leakage_weight'),
        'l2_regularization': optimized_payload.get('l2_regularization'),
        'optimization_cost': float(getattr(opt_res, 'fun', np.nan)),
        'optimization_success': bool(getattr(opt_res, 'success', False)),
        'optimization_nfev': int(getattr(opt_res, 'nfev', -1)),
        'optimization_message': str(getattr(opt_res, 'message', '')),
        'optimization_save_base_name': optimized_payload.get('save_base_name'),
    }

    payload: Dict[str, Any] = {
        'benchmark_type': 'sequence_fidelity_scan',
        'gate_name': 'x90_gate',
        'scan_result': scan_result,
        'fit_params': fit_params,
        'pulse_metadata': pulse_metadata,
        'timestamp': _timestamp(),
    }

    if save:
        base_opt_name = optimized_payload.get('save_base_name', 'unsaved_opt')
        base_name = f"rb_x90_gate_from={base_opt_name}_mmin={seq[0]}_mmax={seq[-1]}_{_timestamp()}"
        meta = {
            'benchmark_type': 'sequence_fidelity_scan',
            'gate_name': 'x90_gate',
            'sequence_lengths': scan_result['m'].tolist(),
            'avg_fidelity': [float(x) for x in scan_result['avg_fidelity']],
            'num_probes': scan_result['num_probes'],
            'fit_params': fit_params if fit_params else None,
            'pulse_metadata': pulse_metadata,
            'timestamp': payload['timestamp']
        }
        arrays = {
            'm': scan_result['m'],
            'avg_fidelity': scan_result['avg_fidelity'],
            'per_probe_fidelity': scan_result['per_probe_fidelity'],
            'optimal_coefficients': opt_res.x,
        }
        saved_paths = _save_optimization_results(base_name, meta, arrays, results_dir=results_dir)
        payload['saved_paths'] = saved_paths
        payload['save_base_name'] = base_name

    return payload


def run_sequence_benchmark_y90(system_params: Dict[str, float],
                               collapse_ops: List[Qobj],
                               optimized_payload: Dict[str, Any],
                               sequence_lengths: Iterable[int] = (1, 2, 3, 4, 5, 6),
                               probe_states: Optional[ProbeStateSet] = None,
                               fit: bool = True,
                               save: bool = True,
                               results_dir: str = 'results',
                               ) -> Dict[str, Any]:
    """Run sequence fidelity benchmark for √Y gate using optimized payload."""
    seq = sorted({int(m) for m in sequence_lengths if int(m) > 0})
    if len(seq) == 0:
        raise ValueError("sequence_lengths must contain at least one positive integer")
    scan_result = sequence_fidelity_scan_y90(system_params, collapse_ops, optimized_payload, seq, probe_states)
    fit_params = fit_exponential_decay(scan_result['m'], scan_result['avg_fidelity']) if fit else None

    opt_res = optimized_payload['optimization_result']
    pulse_metadata = {
        'fourier_modes': optimized_payload.get('fourier_modes'),
        'control_period': optimized_payload.get('control_period'),
        'max_iterations': optimized_payload.get('max_iterations'),
        'coefficient_bounds': optimized_payload.get('coefficient_bounds'),
        'leakage_weight': optimized_payload.get('leakage_weight'),
        'l2_regularization': optimized_payload.get('l2_regularization'),
        'optimization_cost': float(getattr(opt_res, 'fun', np.nan)),
        'optimization_success': bool(getattr(opt_res, 'success', False)),
        'optimization_nfev': int(getattr(opt_res, 'nfev', -1)),
        'optimization_message': str(getattr(opt_res, 'message', '')),
        'optimization_save_base_name': optimized_payload.get('save_base_name'),
    }

    payload: Dict[str, Any] = {
        'benchmark_type': 'sequence_fidelity_scan',
        'gate_name': 'y90_gate',
        'scan_result': scan_result,
        'fit_params': fit_params,
        'pulse_metadata': pulse_metadata,
        'timestamp': _timestamp(),
    }

    if save:
        base_opt_name = optimized_payload.get('save_base_name', 'unsaved_opt')
        base_name = f"rb_y90_gate_from={base_opt_name}_mmin={seq[0]}_mmax={seq[-1]}_{_timestamp()}"
        meta = {
            'benchmark_type': 'sequence_fidelity_scan',
            'gate_name': 'y90_gate',
            'sequence_lengths': scan_result['m'].tolist(),
            'avg_fidelity': [float(x) for x in scan_result['avg_fidelity']],
            'num_probes': scan_result['num_probes'],
            'fit_params': fit_params if fit_params else None,
            'pulse_metadata': pulse_metadata,
            'timestamp': payload['timestamp']
        }
        arrays = {
            'm': scan_result['m'],
            'avg_fidelity': scan_result['avg_fidelity'],
            'per_probe_fidelity': scan_result['per_probe_fidelity'],
            'optimal_coefficients': opt_res.x,
        }
        saved_paths = _save_optimization_results(base_name, meta, arrays, results_dir=results_dir)
        payload['saved_paths'] = saved_paths
        payload['save_base_name'] = base_name

    return payload
