"""
Optimization module for quantum control.

This module provides cost functions and optimization routines for finding
optimal control fields to achieve target quantum evolutions.
"""

import os
import json
from datetime import datetime
from typing import Callable, Dict, Any, Optional, Tuple, Sequence, List
from collections.abc import Mapping

import numpy as np
from scipy.optimize import minimize
from qutip import (
    tensor, Qobj, qeye, expect, ptrace, fidelity,
    basis, basis as qt_basis, sigmax, sigmay, sigmaz, tensor as qt_tensor,
)

from .dynamics import simulate_controlled_evolution_iq, simulate_controlled_evolution_iq_multi
from .operators import create_operators
from .states import create_fock_state
from .config import (
    DEFAULT_FOURIER_MODES, DEFAULT_OPTIMIZATION_ITERATIONS,
    DEFAULT_OPTIMIZATION_METHOD, COEFFICIENT_LOWER_BOUND, COEFFICIENT_UPPER_BOUND,
    time_array, QUBIT_FREQUENCY, NUM_FOCK_STATES,
)
from .utils import split_iq_vector_nd

def global_optimization_multistart(cost_function: Callable[[np.ndarray], float],
                                 vector_dimension: int = DEFAULT_FOURIER_MODES,
                                 max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
                                 optimization_method: str = DEFAULT_OPTIMIZATION_METHOD,
                                 bounds: Optional[Tuple[float, float]] = None,
                                 verbose: bool = True,
                                 initial_guess: Optional[np.ndarray] = None) -> Any:
    """
    Perform global optimization using multiple random starting points.
    
    This function implements a multi-start optimization strategy to find
    the global minimum of the cost function, which is crucial for quantum
    control problems that often have multiple local minima.
    
    Parameters:
    -----------
    cost_function : callable
        The objective function to minimize f(x) -> float
    vector_dimension : int, default from config
        Dimension of optimization parameter space (number of Fourier modes)
    max_iterations : int, default from config
        Number of random starting points to try
    optimization_method : str, default from config
        Optimization algorithm (L-BFGS-B is good for bounded problems)
    bounds : tuple, optional
        Bounds for variables (lower, upper). Default from config.
    verbose : bool, default=True
        Whether to print optimization progress
    initial_guess : np.ndarray, optional
        An initial parameter vector. If provided, multistart is disabled and
        only this guess is used.
        
    Returns:
    --------
    OptimizeResult
        Optimization result object containing the best solution found
        
    Notes:
    ------
    The L-BFGS-B algorithm is particularly suitable for this problem because:
    1. It handles box constraints naturally
    2. It's efficient for moderately high-dimensional problems  
    3. It has good convergence properties for smooth cost functions
    """

    nfev = 0  # Total function evaluations counter

    if bounds is None:
        bounds = (COEFFICIENT_LOWER_BOUND, COEFFICIENT_UPPER_BOUND)
    
    # Create bounds array for scipy
    optimization_bounds = [bounds] * vector_dimension #* \
        #np.expand_dims(np.sort(np.random.exponential(1/bounds[1], vector_dimension))[::-1], axis=1)

    if initial_guess is not None:
        min_val = min(np.min(initial_guess), np.min(bounds))
        max_val = max(np.max(initial_guess), np.max(bounds))
        bounds = (min_val, max_val)
        optimization_bounds = [bounds] * vector_dimension #* \
            #np.expand_dims(np.sort(np.random.standard_exponential(vector_dimension))[::-1], axis=1)

        if verbose:
            print("Starting optimization with provided initial guess...")
            print(f"Parameter space dimension: {vector_dimension}")
            print(f"Optimization method: {optimization_method}")
            print(f"Variable bounds: {bounds}")
            print("-" * 60)
        
        best_result = minimize(
            cost_function, initial_guess, 
            method=optimization_method, bounds=optimization_bounds
        )
        if verbose:
            print(f"Cost = {best_result.fun:.6e}")
            print("-" * 60)
            print(f"Optimization completed. Best cost: {best_result.fun:.6e}")
            print(f"Optimization success: {best_result.success}")
            print(f"Function evaluations: {best_result.nfev}")

            nfev += best_result.nfev

        # Attach nfev attribute explicitly (some tests expect direct OptimizeResult)
        if not hasattr(best_result, 'total_nfev'):
            try:
                setattr(best_result, 'total_nfev', best_result.nfev)
            except Exception:
                pass
        return best_result

    if verbose:
        print(f"Starting global optimization with {max_iterations} random initializations...")
        print(f"Parameter space dimension: {vector_dimension}")
        print(f"Optimization method: {optimization_method}")
        print(f"Variable bounds: {bounds}")
        print("-" * 60)
    
    # First optimization attempt
    # initial_guess_random = np.random.exponential(scale=bounds[1]/2, size=vector_dimension) * \
    #                     np.exp(-np.arange(1, vector_dimension + 1)) *\
    #                     np.random.choice([-1, 1], vector_dimension)
    initial_guess_random = np.random.normal(loc=0, scale=0.5, size=vector_dimension)# * \
                            #np.exp(-np.arange(1, vector_dimension + 1)/4)

    best_result = minimize(
        cost_function, initial_guess_random, 
        method=optimization_method, bounds=optimization_bounds
    )

    nfev += best_result.nfev
    
    if verbose:
        print(f"Iteration 1: Parameters = {len(best_result.x)}, Cost = {best_result.fun:.6e}")
    
    # Additional attempts to find global minimum
    for iteration in range(2, max_iterations + 1):
        current_guess = np.random.normal(loc=0, scale=0.5, size=vector_dimension)
     
        current_result = minimize(
            cost_function, current_guess,
            method=optimization_method, bounds=optimization_bounds
        )

        nfev += current_result.nfev

        if verbose:
            print(f"Iteration {iteration}: Parameters = {len(current_result.x)}, Cost = {current_result.fun:.6e}")
        
        # Keep the best result found so far
        if current_result.fun < best_result.fun:
            best_result = current_result
            if verbose:
                print(f"  → New best solution found!")
    
    if verbose:
        print("-" * 60)
        print(f"Optimization completed. Best cost: {best_result.fun:.6e}")
        print(f"Optimization success: {best_result.success}")
        print(f"Function evaluations: {best_result.nfev}")
    
    # Attach aggregate nfev for callers that care
    try:
        setattr(best_result, 'total_nfev', nfev)
    except Exception:
        pass
    return best_result

def create_mse_cost_function(target_evolution: np.ndarray,
                           drive_period: float,
                           system_params: Dict[str, float],
                           collapse_ops: List[Qobj]) -> Callable[[np.ndarray], float]:
    """
    Create a mean-squared error cost function for optimization.
    
    Parameters:
    -----------
    target_evolution : np.ndarray
        Target evolution trajectory
    drive_period : float
        Control field period
    system_params : dict, optional
        System parameters
        
    Returns:
    --------
    callable
        Cost function f(fourier_coefficients) -> float
    """
    def cost_function(coeffs_iq: np.ndarray) -> float:
        """
        Mean-squared error cost function.
        
        Parameters:
        -----------
        fourier_coefficients : np.ndarray
            Control field Fourier coefficients
            
        Returns:
        --------
        float
            Mean-squared error between target and achieved evolution
        """
        # IQ-only: split 2N vector into I and Q and simulate lab-frame evolution
        half = len(coeffs_iq) // 2
        coeffs_I = coeffs_iq[:half]
        coeffs_Q = coeffs_iq[half:]
        result = simulate_controlled_evolution_iq(
            time_array, coeffs_I, coeffs_Q, drive_period, float(QUBIT_FREQUENCY), 
            system_params=system_params, collapse_ops=collapse_ops
        )
        # Use qubit excitation probability ⟨σ+σ-⟩ (last measurement trace)
        simulated_evolution = result['expect'][-1]
        
        # Calculate mean-squared error
        error_vector = simulated_evolution - target_evolution
        mse = np.mean(error_vector**2)
        
        return mse
    
    return cost_function

# -----------------------------------------------------------------------------
# Multi-qubit (two-qubit+) helpers — backward-compatible additions
# -----------------------------------------------------------------------------

def create_mse_cost_function_multi(
    target_evolution_per_qubit: Sequence[np.ndarray],
    drive_period: float,
    drive_frequencies: Sequence[float],
    system_params: Dict[str, Any],
    collapse_ops: List[Qobj],
    *,
    num_qubits: int = 2,
) -> Callable[[np.ndarray], float]:
    """Create a multi-qubit MSE cost over per-qubit expectation traces.

    This compares each qubit's ⟨σ_z⟩(t) trajectory to a provided target trace.

    Notes
    -----
    - Input coefficient vector must be of length 2*N*num_qubits: per-qubit (I,Q) blocks.
    - Uses the lab-frame multi-qubit simulator.
    - For backward compatibility, this is an additive API and doesn't change
      single-qubit behavior.
    """

    # Basic shape checks are deferred to runtime to stay lightweight
    targets = [np.asarray(te, dtype=float) for te in target_evolution_per_qubit]

    def cost_function(coeff_vector: np.ndarray) -> float:
        # Simulate multi-qubit evolution
        res = simulate_controlled_evolution_iq_multi(
            time_array,
            coeff_vector=coeff_vector,
            drive_period=float(drive_period),
            drive_frequencies=drive_frequencies,
            system_params=system_params,
            num_qubits=num_qubits,
            collapse_ops=collapse_ops
        )

        states = res.get('states', [])
        if not states:
            # If simulation didn't produce states, return large penalty
            return 1.0

        # Build per-qubit sigma_z operators
        ops = create_operators(num_qubits=num_qubits)
        sz_ops = [ops[f'qubit_sigma_z_{j}'] for j in range(num_qubits)]

        # Compute per-qubit expectation traces
        traces: List[np.ndarray] = []
        for j in range(num_qubits):
            trj = np.array([expect(sz_ops[j], st) for st in states], dtype=float)
            traces.append(trj)

        # Accumulate MSE over provided targets; if fewer targets than qubits,
        # we compare only for those provided.
        mse_sum = 0.0
        count = 0
        for j, target in enumerate(targets):
            if j >= num_qubits:
                break
            if target.shape != traces[j].shape:
                # Simple alignment: interpolate target to time_array length
                target_aligned = np.interp(np.linspace(0.0, 1.0, len(time_array)),
                                           np.linspace(0.0, 1.0, len(target)), target)
            else:
                target_aligned = target
            diff = traces[j] - target_aligned
            mse_sum += float(np.mean(diff**2))
            count += 1
        if count == 0:
            return 1.0
        return mse_sum / count

    return cost_function


def create_final_excitation_sum_cost(
    drive_period: float,
    drive_frequencies: Sequence[float],
    system_params: Dict[str, Any],
    collapse_ops: List[Qobj],
    *,
    num_qubits: int = 2,
    maximize: bool = True,
) -> Callable[[np.ndarray], float]:
    """Cost that targets the sum of final excited-state populations across qubits.

    If maximize=True (default), the cost returns the negative of the sum so that
    minimizing the cost increases the final excitation across qubits.
    """

    def cost_function(coeff_vector: np.ndarray) -> float:
        res = simulate_controlled_evolution_iq_multi(
            time_array,
            coeff_vector=coeff_vector,
            drive_period=float(drive_period),
            drive_frequencies=drive_frequencies,
            system_params=system_params,
            num_qubits=num_qubits,
            collapse_ops=collapse_ops
        )
        states = res.get('states', [])
        if not states:
            return 1.0
        psi_f = states[-1]

        ops = create_operators(num_qubits=num_qubits)
        # Projection onto |e> for each qubit j: σ+_j σ-_j
        total = 0.0
        for j in range(num_qubits):
            pj = ops[f'qubit_raising_{j}'] * ops[f'qubit_lowering_{j}']
            try:
                val = expect(pj, psi_f)
                # Ensure scalar float
                if np.ndim(val) == 0:
                    total += float(np.real(val))
                else:
                    total += float(np.real(np.asarray(val).ravel()[0]))
            except Exception:
                # Penalize if expectation failed
                return 1.0
        return -total if maximize else total

    return cost_function


def optimize_multi_qubit_mse(
    targets_per_qubit: Sequence[np.ndarray],
    drive_period: float,
    drive_frequencies: Sequence[float],
    system_params: Dict[str, Any],
    collapse_ops: List[Qobj],
    *,
    num_qubits: int = 2,
    fourier_modes: int = DEFAULT_FOURIER_MODES,
    max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
    coefficient_bounds: Optional[Tuple[float, float]] = None,
    optimization_method: str = DEFAULT_OPTIMIZATION_METHOD,
    initial_guess: Optional[np.ndarray] = None,
    save: bool = True,
    results_dir: str = "results/multi_qubit_mse",
) -> Dict[str, Any]:
    """Optimize per-qubit controls to match per-qubit target trajectories (⟨σz⟩).

    Backward compatible addition; does not alter single-qubit APIs.
    """
    if coefficient_bounds is None:
        coefficient_bounds = (COEFFICIENT_LOWER_BOUND, COEFFICIENT_UPPER_BOUND)

    cost = create_mse_cost_function_multi(
        targets_per_qubit,
        drive_period,
        drive_frequencies,
        system_params=system_params,
        num_qubits=num_qubits,
        collapse_ops=collapse_ops
    )

    vec_dim = 2 * fourier_modes * num_qubits
    result = global_optimization_multistart(
        cost,
        vector_dimension=vec_dim,
        max_iterations=max_iterations,
        optimization_method=optimization_method,
        bounds=coefficient_bounds,
        initial_guess=initial_guess
    )

    payload: Dict[str, Any] = {
        'optimization_result': result,
        'target_type': 'multi_qubit_mse',
        'num_qubits': num_qubits,
        'fourier_modes': fourier_modes,
        'control_period': float(drive_period),
        'drive_frequencies': list(drive_frequencies),
        'max_iterations': max_iterations,
        'coefficient_bounds': coefficient_bounds,
    }

    if save:
        base_name = (
            f"opt_multi_mse_nq={num_qubits}_modes={fourier_modes}_iters={max_iterations}_cost={result.fun:.3e}_{_timestamp()}"
        )
        meta = {
            'target_type': 'multi_qubit_mse',
            'num_qubits': num_qubits,
            'fourier_modes': fourier_modes,
            'max_iterations': max_iterations,
            'control_period': float(drive_period),
            'drive_frequencies': list(map(float, drive_frequencies)),
            'time_points': len(time_array),
            'timestamp': _timestamp(),
            'optimization_result': _serialize_optimize_result(result),
        }
        # Unpack per-qubit I/Q for convenience when loading
        arrays: Dict[str, np.ndarray] = {}
        try:
            per_qubit = split_iq_vector_nd(result.x, n_qubits=num_qubits)
            for j, (cI, cQ) in enumerate(per_qubit):
                arrays[f'optimal_coefficients_I_q{j}'] = np.asarray(cI)
                arrays[f'optimal_coefficients_Q_q{j}'] = np.asarray(cQ)
        except Exception:
            arrays['optimal_coefficients'] = np.asarray(result.x)
        paths = _save_optimization_results(base_name, meta, arrays, results_dir=results_dir)
        payload['saved_paths'] = paths
        payload['save_base_name'] = base_name

    return payload


def optimize_multi_qubit_final_excitation(
    drive_period: float,
    drive_frequencies: Sequence[float],
    system_params: Dict[str, Any],
    collapse_ops: List[Qobj],
    *,
    num_qubits: int = 2,
    fourier_modes: int = DEFAULT_FOURIER_MODES,
    max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
    coefficient_bounds: Optional[Tuple[float, float]] = None,
    optimization_method: str = DEFAULT_OPTIMIZATION_METHOD,
    initial_guess: Optional[np.ndarray] = None,
    maximize: bool = True,
    save: bool = True,
    results_dir: str = "results/multi_qubit_excite",
) -> Dict[str, Any]:
    """Optimize to maximize (or minimize) the sum of final excited populations across qubits.

    The parameter vector packs per-qubit I/Q Fourier coefficients.
    """
    if coefficient_bounds is None:
        coefficient_bounds = (COEFFICIENT_LOWER_BOUND, COEFFICIENT_UPPER_BOUND)

    cost = create_final_excitation_sum_cost(
        drive_period=drive_period,
        drive_frequencies=drive_frequencies,
        system_params=system_params,
        num_qubits=num_qubits,
        maximize=maximize,
        collapse_ops=collapse_ops
    )

    vec_dim = 2 * fourier_modes * num_qubits
    result = global_optimization_multistart(
        cost,
        vector_dimension=vec_dim,
        max_iterations=max_iterations,
        optimization_method=optimization_method,
        bounds=coefficient_bounds,
        initial_guess=initial_guess
    )

    payload: Dict[str, Any] = {
        'optimization_result': result,
        'target_type': 'multi_qubit_final_excitation',
        'num_qubits': num_qubits,
        'fourier_modes': fourier_modes,
        'control_period': float(drive_period),
        'drive_frequencies': list(drive_frequencies),
        'max_iterations': max_iterations,
        'coefficient_bounds': coefficient_bounds,
        'maximize': bool(maximize),
    }

    if save:
        base_name = (
            f"opt_multi_excite_nq={num_qubits}_modes={fourier_modes}_iters={max_iterations}_cost={result.fun:.3e}_{_timestamp()}"
        )
        meta = {
            'target_type': 'multi_qubit_final_excitation',
            'num_qubits': num_qubits,
            'fourier_modes': fourier_modes,
            'max_iterations': max_iterations,
            'control_period': float(drive_period),
            'drive_frequencies': list(map(float, drive_frequencies)),
            'time_points': len(time_array),
            'timestamp': _timestamp(),
            'optimization_result': _serialize_optimize_result(result),
            'maximize': bool(maximize),
        }
        arrays: Dict[str, np.ndarray] = {}
        try:
            per_qubit = split_iq_vector_nd(result.x, n_qubits=num_qubits)
            for j, (cI, cQ) in enumerate(per_qubit):
                arrays[f'optimal_coefficients_I_q{j}'] = np.asarray(cI)
                arrays[f'optimal_coefficients_Q_q{j}'] = np.asarray(cQ)
        except Exception:
            arrays['optimal_coefficients'] = np.asarray(result.x)
        paths = _save_optimization_results(base_name, meta, arrays, results_dir=results_dir)
        payload['saved_paths'] = paths
        payload['save_base_name'] = base_name

    return payload

def create_fidelity_cost_function(target_state,
                                target_time: float,
                                drive_period: float,
                                system_params: Dict[str, float],
                                collapse_ops: List[Qobj]) -> Callable[[np.ndarray], float]:
    """
    Create a state fidelity cost function for gate optimization.
    
    Parameters:
    -----------
    target_state : Qobj
        Target quantum state at target_time
    target_time : float
        Time at which to evaluate fidelity
    drive_period : float
        Control field period
    system_params : dict, optional
        System parameters
        
    Returns:
    --------
    callable
        Cost function that minimizes 1 - fidelity
    """
    def fidelity_cost_function(coeffs_iq: np.ndarray) -> float:
        """
        State fidelity cost function (1 - fidelity).
        
        Parameters:
        -----------
        fourier_coefficients : np.ndarray
            Control field Fourier coefficients
            
        Returns:
        --------
        float
            Cost = 1 - fidelity(target_state, final_state)
        """
        # Create time array up to target time
        time_points = np.linspace(0, target_time, 
                                 int(target_time / time_array[1] * len(time_array)))
        
        # Simulate evolution (IQ-only)
        half = len(coeffs_iq) // 2
        coeffs_I = coeffs_iq[:half]
        coeffs_Q = coeffs_iq[half:]
        result = simulate_controlled_evolution_iq(
            time_points, coeffs_I, coeffs_Q, drive_period, float(QUBIT_FREQUENCY), 
            system_params, collapse_ops=collapse_ops
        )
        final_state = result['final_state']
        # Calculate fidelity
        fid_val = fidelity(target_state, final_state)
        # Return 1 - fidelity (to minimize)
        return 1.0 - float(fid_val)
    
    return fidelity_cost_function

def create_composite_cost_function(cost_components: list,
                                 weights: Optional[np.ndarray] = None) -> Callable[[np.ndarray], float]:
    """
    Create a composite cost function from multiple components.
    
    Parameters:
    -----------
    cost_components : list
        List of cost functions
    weights : np.ndarray, optional
        Weights for each component. Default: equal weights.
        
    Returns:
    --------
    callable
        Composite cost function
    """
    if weights is None:
        weights = np.ones(len(cost_components)) / len(cost_components)
    
    def composite_cost_function(fourier_coefficients: np.ndarray) -> float:
        """
        Weighted sum of multiple cost components.
        
        Parameters:
        -----------
        fourier_coefficients : np.ndarray
            Control field Fourier coefficients
            
        Returns:
        --------
        float
            Weighted sum of all cost components
        """
        total_cost = 0.0
        
        for weight, cost_func in zip(weights, cost_components):
            component_cost = cost_func(fourier_coefficients)
            total_cost += weight * component_cost
        
        return total_cost
    
    return composite_cost_function

def _prepare_results_directory(directory: str = "results") -> str:
    """Ensure results directory exists and return its path."""
    os.makedirs(directory, exist_ok=True)
    return directory

def _timestamp() -> str:
    """Return a UTC timestamp string formatted as YYYYMMDDTHHMMSS."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S")

def _serialize_optimize_result(opt_result: Any) -> Dict[str, Any]:
    """Extract serializable fields from scipy OptimizeResult."""
    return {
        'x': opt_result.x.tolist(),
        'fun': float(opt_result.fun),
        'success': bool(opt_result.success),
        'status': int(getattr(opt_result, 'status', -1)),
        'message': str(opt_result.message),
        'nfev': int(getattr(opt_result, 'nfev', -1)),
        'njev': int(getattr(opt_result, 'njev', -1)) if getattr(opt_result, 'njev', None) is not None else None
    }

def _save_optimization_results(base_name: str,
                               data: Dict[str, Any],
                               arrays: Dict[str, np.ndarray],
                               results_dir: str = "results") -> Dict[str, str]:
    """
    Save optimization metadata (JSON) and arrays (NPZ) to disk.

    Returns mapping with paths used.
    """
    _prepare_results_directory(results_dir)
    json_path = os.path.join(results_dir, f"{base_name}.json")
    npz_path = os.path.join(results_dir, f"{base_name}.npz")

    # Write JSON metadata
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    # Save arrays
    np.savez_compressed(npz_path, **{k: v for k, v in arrays.items()})

    return {'json': json_path, 'npz': npz_path}

def optimize_sinusoidal_target(system_params: Dict[str, float],
                             collapse_ops: List[Qobj],
                             frequency_factor: float = 1.0,
                             fourier_modes: int = DEFAULT_FOURIER_MODES,
                             max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
                             save: bool = True,
                             results_dir: str = "results/sinusoidal",
                             optimization_method = DEFAULT_OPTIMIZATION_METHOD,
                             bounds: Optional[Tuple[float, float]] = None,
                             initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Optimize control for sinusoidal target |sin(ft)|.
    
    Parameters:
    -----------
    frequency_factor : float, default=1.0
        Frequency scaling factor for sine function
    fourier_modes : int
        Number of Fourier modes in control field
    max_iterations : int
        Number of optimization iterations
        
    Returns:
    --------
    dict
        Optimization results and metadata
    """
    # Create target evolution
    target_evolution = np.abs(np.sin(frequency_factor * time_array))
    
    # Control field period
    control_period = time_array[-1]
    
    # Create cost function (IQ expects 2N vector: [I..., Q...])
    cost_function = create_mse_cost_function(target_evolution, control_period, 
                                             system_params=system_params, collapse_ops=collapse_ops)

    # Optimize over 2N parameters
    result, nfev = global_optimization_multistart(
        cost_function, 2 * fourier_modes, max_iterations,
        optimization_method=optimization_method,
        bounds=bounds,
        initial_guess=initial_guess
    )
    
    payload = {
        'optimization_result': result,
        'target_evolution': target_evolution,
        'control_period': control_period,
        'fourier_modes': fourier_modes,
        'target_type': 'sinusoidal',
        'frequency_factor': frequency_factor,
        'nfev': nfev
    }

    if save:
        base_name = (
            f"opt_sinusoidal_freq={frequency_factor:g}_modes={fourier_modes}_iters={max_iterations}_" \
            f"cost={result.fun:.3e}_{_timestamp()}"
        )
        meta = {
            'target_type': 'sinusoidal',
            'frequency_factor': frequency_factor,
            'fourier_modes': fourier_modes,
            'max_iterations': max_iterations,
            'nfev': nfev,
            'control_period': control_period,
            'time_points': len(time_array),
            'timestamp': _timestamp(),
            'optimization_result': _serialize_optimize_result(result)
        }
        half = len(result.x) // 2
        arrays = {
            'target_evolution': target_evolution,
            'optimal_coefficients_I': result.x[:half],
            'optimal_coefficients_Q': result.x[half:]
        }
        paths = _save_optimization_results(base_name, meta, arrays, results_dir=results_dir)
        payload['saved_paths'] = paths
        payload['save_base_name'] = base_name
    return payload

def optimize_gaussian_target(system_params: Dict[str, float],
                           collapse_ops: List[Qobj],
                           gaussian_width: float = 1.0,
                           fourier_modes: int = DEFAULT_FOURIER_MODES,
                           max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
                           save: bool = True,
                           results_dir: str = "results/gaussian",
                           optimization_method = DEFAULT_OPTIMIZATION_METHOD,
                           bounds: Optional[Tuple[float, float]] = None,
                           initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Optimize control for Gaussian target pulse.
    
    Parameters:
    -----------
    gaussian_width : float, default=1.0
        Standard deviation of Gaussian pulse
    fourier_modes : int
        Number of Fourier modes
    max_iterations : int
        Number of optimization iterations
        
    Returns:
    --------
    dict
        Optimization results and metadata

    Notes:
    ------
    The mean-squared error is computed against the qubit excitation probability
    trace ⟨σ+σ−⟩ produced by simulate_controlled_evolution (consistent with the
    sinusoidal target). No change to cost code is required because
    create_mse_cost_function already selects that observable.
    """
    # Create target evolution (Gaussian centered at middle time)
    time_center = time_array[len(time_array) // 2]
    target_evolution = np.exp(-((time_array - time_center) / gaussian_width)**2)
    
    # Use full time window as period
    control_period = time_array[-1]
    
    # Create cost function (IQ expects 2N vector)
    cost_function = create_mse_cost_function(target_evolution, control_period,
                                             system_params=system_params, collapse_ops=collapse_ops)
    
    # Optimize over 2N parameters
    result, nfev = global_optimization_multistart(
        cost_function, 2 * fourier_modes, max_iterations, 
        optimization_method=optimization_method,
        initial_guess=initial_guess, bounds=bounds
    )
    
    payload = {
        'optimization_result': result,
        'target_evolution': target_evolution,
        'control_period': control_period,
        'fourier_modes': fourier_modes,
        'target_type': 'gaussian',
        'gaussian_width': gaussian_width,
        'time_center': time_center,
        'nfev': nfev
    }

    if save:
        base_name = (
            f"opt_gaussian_width={gaussian_width:g}_modes={fourier_modes}_iters={max_iterations}_" \
            f"cost={result.fun:.3e}_{_timestamp()}"
        )
        meta = {
            'target_type': 'gaussian',
            'gaussian_width': gaussian_width,
            'time_center': float(time_center),
            'fourier_modes': fourier_modes,
            'max_iterations': max_iterations,
            'nfev': nfev,
            'control_period': control_period,
            'time_points': len(time_array),
            'timestamp': _timestamp(),
            'optimization_result': _serialize_optimize_result(result)
        }
        half = len(result.x) // 2
        arrays = {
            'target_evolution': target_evolution,
            'optimal_coefficients_I': result.x[:half],
            'optimal_coefficients_Q': result.x[half:]
        }
        paths = _save_optimization_results(base_name, meta, arrays, results_dir=results_dir)
        payload['saved_paths'] = paths
        payload['save_base_name'] = base_name
    return payload

def optimize_rectangular_target(system_params: Dict[str, float],
                              collapse_ops: List[Qobj],
                              pulse_fraction: float = 1.0/3.0,
                              fourier_modes: int = 40,  # More modes for sharp edges
                              max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
                              save: bool = True,
                              results_dir: str = "results/rectangular",
                              optimization_method: str = DEFAULT_OPTIMIZATION_METHOD,
                              initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Optimize control for rectangular (step) target function.
    
    Parameters:
    -----------
    pulse_fraction : float, default=1/3
        Fraction of total time for pulse (centered)
    fourier_modes : int, default=40
        Number of Fourier modes (more needed for sharp edges)
    max_iterations : int
        Number of optimization iterations
        
    Returns:
    --------
    dict
        Optimization results and metadata

    Notes:
    ------
    The cost function compares the target against the qubit excitation probability
    ⟨σ+σ−⟩ over time (same observable used in sinusoidal and Gaussian routines).
    """
    # Create rectangular target evolution
    target_evolution = np.zeros_like(time_array)
    pulse_start = int((1 - pulse_fraction) / 2 * len(time_array))
    pulse_end = int((1 + pulse_fraction) / 2 * len(time_array))
    target_evolution[pulse_start:pulse_end] = 1.0
    
    # Use full time window as period
    control_period = time_array[-1]
    
    # Create cost function (IQ expects 2N vector)
    cost_function = create_mse_cost_function(target_evolution, control_period,
                                             system_params=system_params, collapse_ops=collapse_ops)
    
    # Optimize over 2N parameters
    result, nfev = global_optimization_multistart(
        cost_function, 2 * fourier_modes, max_iterations,
        optimization_method=optimization_method,
        initial_guess=initial_guess
    )
    
    payload = {
        'optimization_result': result,
        'target_evolution': target_evolution,
        'control_period': control_period,
        'fourier_modes': fourier_modes,
        'target_type': 'rectangular',
        'pulse_fraction': pulse_fraction,
        'pulse_start_index': pulse_start,
        'pulse_end_index': pulse_end,
        'nfev': nfev
    }

    if save:
        base_name = (
            f"opt_rectangular_frac={pulse_fraction:g}_modes={fourier_modes}_iters={max_iterations}_" \
            f"cost={result.fun:.3e}_{_timestamp()}"
        )
        meta = {
            'target_type': 'rectangular',
            'pulse_fraction': pulse_fraction,
            'pulse_start_index': pulse_start,
            'pulse_end_index': pulse_end,
            'fourier_modes': fourier_modes,
            'max_iterations': max_iterations,
            'nfev': nfev,
            'control_period': control_period,
            'time_points': len(time_array),
            'timestamp': _timestamp(),
            'optimization_result': _serialize_optimize_result(result)
        }
        half = len(result.x) // 2
        arrays = {
            'target_evolution': target_evolution,
            'optimal_coefficients_I': result.x[:half],
            'optimal_coefficients_Q': result.x[half:]
        }
        paths = _save_optimization_results(base_name, meta, arrays, results_dir=results_dir)
        payload['saved_paths'] = paths
        payload['save_base_name'] = base_name
    return payload

def analyze_optimization_convergence(optimization_results: list) -> Dict[str, Any]:
    """
    Analyze convergence properties of multiple optimization runs.
    
    Parameters:
    -----------
    optimization_results : list
        List of optimization result objects
        
    Returns:
    --------
    dict
        Convergence analysis metrics
    """
    costs = [result.fun for result in optimization_results]
    success_rates = [result.success for result in optimization_results]
    function_evaluations = [result.nfev for result in optimization_results]
    
    analysis = {
        'num_runs': len(optimization_results),
        'best_cost': np.min(costs),
        'worst_cost': np.max(costs),
        'mean_cost': np.mean(costs),
        'std_cost': np.std(costs),
        'success_rate': np.mean(success_rates),
        'mean_function_evals': np.mean(function_evaluations),
        'cost_distribution': costs,
        'convergence_consistency': np.std(costs) / np.mean(costs)  # Coefficient of variation
    }
    
    return analysis

def gradient_based_optimization(cost_function: Callable[[np.ndarray], float],
                              initial_guess: np.ndarray,
                              method: str = 'L-BFGS-B',
                              bounds: Optional[list] = None,
                              options: Optional[Dict] = None) -> Any:
    """
    Perform single gradient-based optimization.
    
    Parameters:
    -----------
    cost_function : callable
        Objective function to minimize
    initial_guess : np.ndarray
        Initial parameter guess
    method : str, default='L-BFGS-B'
        Optimization method
    bounds : list, optional
        Parameter bounds
    options : dict, optional
        Additional options for optimizer
        
    Returns:
    --------
    OptimizeResult
        Single optimization result
    """
    if bounds is None and method == 'L-BFGS-B':
        bounds = [(COEFFICIENT_LOWER_BOUND, COEFFICIENT_UPPER_BOUND)] * len(initial_guess)
    
    if options is None:
        options = {'maxiter': 1000, 'ftol': 1e-9}
    
    result = minimize(
        cost_function, 
        initial_guess,
        method=method,
        bounds=bounds,
        options=options
    )
    
    return result

def print_optimization_summary(optimization_result: Dict[str, Any]):
    """
    Print formatted summary of optimization results.
    
    Parameters:
    -----------
    optimization_result : dict
        Results from optimize_*_target functions
    """
    result = optimization_result['optimization_result']
    
    print(f"\nOptimization Summary: {optimization_result['target_type'].title()} Target")
    print("=" * 60)


# =============================
# Gate Optimization Routines
# =============================

def _single_qubit_gate_targets() -> Dict[str, Qobj]:
    """Return dictionary of single-qubit target unitaries (embedded in cavity ⊗ qubit space)."""
    # Qubit computational basis |g>=|0>, |e>=|1>
    # Define unitaries on qubit space then tensor with cavity identity (assume vacuum stays factorized)
    I_cav = qeye(NUM_FOCK_STATES)
    # Hadamard matrix explicitly
    H_qubit = (1/np.sqrt(2)) * Qobj([[1, 1],[1, -1]])
    # Phase gates
    S_qubit = Qobj([[1, 0],[0, 1j]])
    T_qubit = Qobj([[1, 0],[0, np.exp(1j*np.pi/4)]])
    # X/Y pi rotations (up to global phase)
    X_qubit = sigmax()
    Y_qubit = sigmay()
    # π/2 rotations (√X, √Y)
    Rx_pi_2 = _rotation_unitary('x', np.pi/2.0)
    Ry_pi_2 = _rotation_unitary('y', np.pi/2.0)
    # Build mapping and include convenient aliases
    gates: Dict[str, Qobj] = {
        'hadamard': tensor(I_cav, H_qubit),
        's_gate': tensor(I_cav, S_qubit),
        't_gate': tensor(I_cav, T_qubit),
        'x_gate': tensor(I_cav, X_qubit),
        'y_gate': tensor(I_cav, Y_qubit),
        'x90_gate': tensor(I_cav, Rx_pi_2),
        'y90_gate': tensor(I_cav, Ry_pi_2),
    }
    # Aliases without _gate suffix
    gates['x'] = gates['x_gate']
    gates['y'] = gates['y_gate']
    gates['x90'] = gates['x90_gate']
    gates['y90'] = gates['y90_gate']
    # Short aliases for phase gates
    gates['s'] = gates['s_gate']
    gates['t'] = gates['t_gate']
    return gates

def _simulate_and_get_final_state(coeffs_iq: np.ndarray, period: float, 
                                  system_params: Dict[str,float], collapse_ops: List[Qobj]) -> Optional[Qobj]:
    """Run IQ simulation for full `time_array` and return the final state if available."""
    half = len(coeffs_iq) // 2
    coeffs_I = coeffs_iq[:half]
    coeffs_Q = coeffs_iq[half:]
    result = simulate_controlled_evolution_iq(
        time_array, coeffs_I, coeffs_Q, period, float(QUBIT_FREQUENCY), 
        system_params, collapse_ops=collapse_ops
    )
    return result.get('final_state')

def _direct_final_state_simulation(coeffs_iq: np.ndarray,
                                   drive_period: float,
                                   initial_state: Qobj,
                                   system_params: Dict[str, float], collapse_ops: List[Qobj]) -> Optional[Qobj]:
    """Obtain final state using the lab-frame I/Q simulator with stored states."""
    half = len(coeffs_iq) // 2
    coeffs_I = coeffs_iq[:half]
    coeffs_Q = coeffs_iq[half:]
    res = simulate_controlled_evolution_iq(
        time_array, coeffs_I, coeffs_Q, drive_period, float(QUBIT_FREQUENCY),
        system_params=system_params, initial_state=initial_state, collapse_ops=collapse_ops
    )
    if res.get('final_state') is not None:
        return res['final_state']
    states = res.get('states', [])
    return states[-1] if states else None

def _gate_fidelity_cost(target_unitary: Qobj,
                        initial_state: Qobj,
                        drive_period: float,
                        system_params: Dict[str, float], collapse_ops: List[Qobj]) -> Callable[[np.ndarray], float]:
    """Return cost function minimizing 1 - fidelity(U_target|ψ0>, |ψ(T)>)"""
    target_state = target_unitary * initial_state
    def cost(coeffs_iq: np.ndarray) -> float:
        # IQ-only final state extraction
        final_state = _direct_final_state_simulation(coeffs_iq, drive_period, initial_state, 
                                                     system_params, collapse_ops)
        if final_state is None:
            print("[gate_cost_debug] Unable to obtain final state; returning cost 1.0")
            return 1.0
        return 1.0 - float(fidelity(target_state, final_state))
    return cost

def _gate_fidelity_with_leakage_cost(target_unitary: Qobj,
                                     initial_state: Qobj,
                                     drive_period: float,
                                     system_params: Dict[str, float], collapse_ops: List[Qobj],
                                     leakage_weight: float = 0.0) -> Callable[[np.ndarray], float]:
    """Return cost function minimizing 1 - fidelity plus optional cavity leakage penalty.

    cost = (1 - F) + w * <n_cavity>
    """
    # Precompute target state once
    target_state = target_unitary * initial_state
    # Prepare cavity number operator (a^\dag a)
    ops = create_operators()
    a = ops['cavity_annihilation']
    n_op = a.dag() * a
    def cost(coeffs_iq: np.ndarray) -> float:
        final_state = _direct_final_state_simulation(coeffs_iq, drive_period, initial_state,
                                                     system_params, collapse_ops)
        if final_state is None:
            return 1.0  # maximal cost if evolution failed
        base = 1.0 - float(fidelity(target_state, final_state))
        if leakage_weight > 0.0:
            try:
                leakage_val = expect(n_op, final_state)
                # Expect may return complex with negligible imag part
                if np.ndim(leakage_val) == 0:
                    leakage = float(np.real(leakage_val))
                else:
                    leakage = float(np.real(leakage_val[0]))
            except Exception:
                leakage = 1.0
            return base + leakage_weight * leakage
        return base
    return cost

def _l2_regularization(term_weight: float, order: int = 2) -> Callable[[np.ndarray], float]:
    """Return a regularization cost function on Fourier coefficients.

    Parameters
    ----------
    term_weight : float
        Prefactor multiplying the regularization term. If 0, returns zero cost.
    order : int, default 2
        Norm order to use (currently only 2 supported robustly).
    """
    if term_weight <= 0.0:
        return lambda coeffs: 0.0
    if order != 2:
        # Fallback to L2; keep interface minimal
        order = 2
    def cost(coeffs: np.ndarray) -> float:
        return term_weight * float(np.sum(coeffs**2)) / len(coeffs)
    return cost

def _combine_costs(*cost_functions: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], float]:
    """Combine multiple scalar cost functions by summation."""
    def total(coeffs: np.ndarray) -> float:
        val = 0.0
        for f in cost_functions:
            val += f(coeffs)
        return val
    return total

# =============================
# Process (Unitary) PTM Cost
# =============================

def _unitary_to_ptm(target_unitary: Qobj) -> np.ndarray:
    """Compute 3x3 Pauli Transfer Matrix (PTM) for a target 2-level unitary.

    R_ij = (1/2) Tr[ σ_i U σ_j U^† ], i,j in {x,y,z}.
    """
    paulis = [sigmax(), sigmay(), sigmaz()]
    R = np.zeros((3, 3), dtype=float)
    U = target_unitary
    for j, sj in enumerate(paulis):
        U_sj = U * sj * U.dag()
        for i, si in enumerate(paulis):
            val = (si * U_sj).tr() / 2.0
            R[i, j] = float(np.real(val))
    return R

def _build_probes_xyz() -> Dict[str, Qobj]:
    """Return pure qubit states for ±X, ±Y, ±Z (kets)."""
    # Computational basis
    q0 = qt_basis(2, 0)
    q1 = qt_basis(2, 1)
    # |±x> = (|0> ± |1>)/√2; |±y> = (|0> ± i|1>)/√2; |±z> = |0>, |1>
    plus_x = (q0 + q1).unit()
    minus_x = (q0 - q1).unit()
    plus_y = (q0 + 1j*q1).unit()
    minus_y = (q0 - 1j*q1).unit()
    plus_z = q0
    minus_z = q1
    return {
        '+x': plus_x, '-x': minus_x,
        '+y': plus_y, '-y': minus_y,
        '+z': plus_z, '-z': minus_z,
    }

def _bloch_vector(rho_qubit: Qobj) -> np.ndarray:
    """Return Bloch vector (rx, ry, rz) for a 2x2 density matrix."""
    return np.array([
        float(np.real((sigmax() * rho_qubit).tr())),
        float(np.real((sigmay() * rho_qubit).tr())),
        float(np.real((sigmaz() * rho_qubit).tr())),
    ])

def _estimate_ptm_from_pulses(
    fourier_coeffs: np.ndarray,
    drive_period: float,
    system_params: Dict[str, float],
    collapse_ops: List[Qobj],
    cavity_leakage_op: Optional[Qobj] = None,
) -> Tuple[np.ndarray, float]:
    """Estimate 3x3 PTM by driving ±X, ±Y, ±Z inputs and tracing to the qubit.

    Returns (R_est, avg_leakage_final), where R_est is 3x3 and leakage is the
    average cavity photon number at the final time across probes if an operator
    is provided; otherwise 0.0.
    """
    # Cavity vacuum
    cav0 = qt_basis(NUM_FOCK_STATES, 0)
    probes = _build_probes_xyz()

    # Columns from differences: R[:,x] ~ (r(+x) - r(-x))/2, etc.
    r_plus = {}
    r_minus = {}
    leakages = []
    for axis in ['x', 'y', 'z']:
        for sign in ['+', '-']:
            ket_q = probes[f'{sign}{axis}']
            ket = tensor(cav0, ket_q)  # composite ket
            # Robust final state extraction (ensures states stored)
            final_state = _direct_final_state_simulation(
                fourier_coeffs, drive_period, ket, system_params, collapse_ops
            )
            # Reduced qubit state
            rho_q = ptrace(final_state, 1)
            r_vec = _bloch_vector(rho_q)
            if sign == '+':
                r_plus[axis] = r_vec
            else:
                r_minus[axis] = r_vec
            # Leakage measure
            if cavity_leakage_op is not None:
                try:
                    lv = float(np.real((cavity_leakage_op * final_state).tr()))
                except Exception:
                    lv = 0.0
                leakages.append(lv)

    R = np.zeros((3, 3), dtype=float)
    axes = {'x': 0, 'y': 1, 'z': 2}
    for a, col in axes.items():
        R[:, col] = 0.5 * (r_plus[a] - r_minus[a])

    avg_leak = float(np.mean(leakages)) if leakages else 0.0
    return R, avg_leak

def _ptm_process_cost(target_unitary: Qobj,
                      drive_period: float,
                      system_params: Dict[str, float],
                      collapse_ops: List[Qobj],
                      leakage_weight: float = 0.0,
                      l2_reg: float = 0.0) -> Callable[[np.ndarray], float]:
    """Return a cost function comparing estimated PTM to target PTM plus leakage/L2.

    Cost = ||R_est - R_target||_F^2 / 9 + w_leak * <n_cav> + w_l2 * ||c||^2 / N
    """
    # Prepare target PTM
    R_target = _unitary_to_ptm(target_unitary)
    # Leakage operator (a^† a) for final-state penalty
    ops = create_operators()
    a = ops['cavity_annihilation']
    n_op = a.dag() * a

    def cost(coeffs: np.ndarray) -> float:
        R_est, avg_leak = _estimate_ptm_from_pulses(coeffs, drive_period, system_params,
                                                    collapse_ops=collapse_ops, cavity_leakage_op=n_op)
        # Frobenius normalized MSE
        mse_ptm = float(np.mean((R_est - R_target)**2))
        # Regularization
        l2 = float(np.sum(coeffs**2)) / len(coeffs) if l2_reg > 0.0 else 0.0
        return mse_ptm + leakage_weight * avg_leak + l2_reg * l2

    return cost

# =============================
# IQ (two-quadrature) PTM Cost
# =============================

def _split_iq(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a 2N vector into (I, Q) halves."""
    half = len(vec) // 2
    return vec[:half], vec[half:]

def _estimate_ptm_from_iq_pulses(
    coeffs_I: np.ndarray,
    coeffs_Q: np.ndarray,
    drive_period: float,
    drive_frequency: float,
    system_params: Dict[str, float],
    collapse_ops: List[Qobj],
    cavity_leakage_op: Optional[Qobj] = None,
) -> Tuple[np.ndarray, float]:
    """Estimate PTM using the IQ simulator over ±X, ±Y, ±Z probes."""
    cav0 = qt_basis(NUM_FOCK_STATES, 0)
    probes = _build_probes_xyz()
    r_plus: Dict[str, np.ndarray] = {}
    r_minus: Dict[str, np.ndarray] = {}
    leakages = []
    for axis in ['x', 'y', 'z']:
        for sign in ['+', '-']:
            ket_q = probes[f'{sign}{axis}']
            ket = tensor(cav0, ket_q)
            res = simulate_controlled_evolution_iq(
                time_array, coeffs_I, coeffs_Q, drive_period, drive_frequency,
                system_params=system_params, collapse_ops=collapse_ops, initial_state=ket
            )
            # Robust final state
            final_state = res.get('final_state')
            if final_state is None:
                states = res.get('states', [])
                final_state = states[-1] if states else ket
            rho_q = ptrace(final_state, 1)
            r_vec = _bloch_vector(rho_q)
            if sign == '+':
                r_plus[axis] = r_vec
            else:
                r_minus[axis] = r_vec
            if cavity_leakage_op is not None:
                try:
                    lv = float(np.real((cavity_leakage_op * final_state).tr()))
                except Exception:
                    lv = 0.0
                leakages.append(lv)

    R = np.zeros((3, 3), dtype=float)
    axes = {'x': 0, 'y': 1, 'z': 2}
    for a, col in axes.items():
        R[:, col] = 0.5 * (r_plus[a] - r_minus[a])
    avg_leak = float(np.mean(leakages)) if leakages else 0.0
    return R, avg_leak

def _ptm_process_cost_iq(target_unitary: Qobj,
                         drive_period: float,
                         drive_frequency: float,
                         system_params: Dict[str, float],
                         collapse_ops: List[Qobj],
                         leakage_weight: float = 0.0,
                         l2_reg: float = 0.0) -> Callable[[np.ndarray], float]:
    """Cost builder for IQ pulses: input coeff vector is 2N: [I... Q...]."""
    R_target = _unitary_to_ptm(target_unitary)
    ops = create_operators()
    a = ops['cavity_annihilation']
    n_op = a.dag() * a

    def cost(coeffs_iq: np.ndarray) -> float:
        cI, cQ = _split_iq(coeffs_iq)
        R_est, avg_leak = _estimate_ptm_from_iq_pulses(cI, cQ, drive_period, drive_frequency,
                                                       system_params, collapse_ops, n_op)
        mse_ptm = float(np.mean((R_est - R_target)**2))
        reg = float(np.sum(coeffs_iq**2)) / len(coeffs_iq) if l2_reg > 0.0 else 0.0
        return mse_ptm + leakage_weight * avg_leak + l2_reg * reg

    return cost

def optimize_single_qubit_gate_process(gate_name: str,
                                       system_params: Dict[str, float],
                                       collapse_ops: List[Qobj],
                                       fourier_modes: int = DEFAULT_FOURIER_MODES,
                                       max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
                                       save: bool = True,
                                       results_dir: str = 'results',
                                       drive_period: Optional[float] = None,
                                       coefficient_bounds: Optional[Tuple[float, float]] = (-1.0, 1.0),
                                       leakage_weight: float = 0.0,
                                       l2_regularization: float = 0.0,
                                       initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Optimize control field to implement a single-qubit gate via multi-state (process-like) fidelity.

    We approximate average gate fidelity by averaging state fidelities over a small
    set of probe input states spanning the Bloch sphere vertices commonly used for
    single-qubit process tomography: |0>, |1>, |+>, |+i> (embedded with cavity |0>). 
    A leakage penalty <n_cavity> and optional L2 coefficient regularization can be added.

    Parameters
    ----------
    gate_name : str
        One of 'hadamard', 's_gate', 't_gate'
    fourier_modes : int
        Number of Fourier coefficients to optimize.
    max_iterations : int
        Number of multistart attempts.
    coefficient_bounds : tuple
        (lower, upper) bounds for each coefficient.
    leakage_weight : float
        Weight of cavity photon number penalty.
    l2_regularization : float
        Weight of L2 norm penalty on coefficients.
    system_params : dict, optional
        Physical parameters override.
    """
    if drive_period is None:
        drive_period = float(time_array[-1])
    gates = _single_qubit_gate_targets()
    if gate_name not in gates:
        raise ValueError(f"Unknown gate '{gate_name}'. Valid: {list(gates.keys())}")
    target_U = gates[gate_name]

    # Define probe states (cavity in vacuum): |0>, |1>, |+>, |+i>
    # create_fock_state(cavity_n, qubit_state)
    psi_0 = create_fock_state(0, 0)
    psi_1 = create_fock_state(0, 1)
    # Superpositions manually (normalize)
    cav0 = basis(NUM_FOCK_STATES, 0)
    q0 = basis(2, 0)
    q1 = basis(2, 1)
    psi_plus = (qt_tensor(cav0, (q0 + q1)) / np.sqrt(2)).unit()
    psi_plus_i = (qt_tensor(cav0, (q0 + 1j * q1)) / np.sqrt(2)).unit()
    probes = [psi_0, psi_1, psi_plus, psi_plus_i]

    # Build average fidelity cost (with leakage) across probes
    probe_costs = [
        _gate_fidelity_with_leakage_cost(target_U, p, drive_period,
                                         leakage_weight=leakage_weight,
                                         system_params=system_params,
                                         collapse_ops=collapse_ops)
        for p in probes
    ]
    avg_cost = lambda coeffs: sum(f(coeffs) for f in probe_costs) / len(probe_costs)
    reg_cost = _l2_regularization(l2_regularization)
    total_cost = _combine_costs(avg_cost, reg_cost)

    result = global_optimization_multistart(total_cost, 2 * fourier_modes, max_iterations,
                                            bounds=coefficient_bounds, initial_guess=initial_guess)
    payload = {
        'optimization_result': result,
        'target_type': f'{gate_name}_process',
        'fourier_modes': fourier_modes,
        'control_period': drive_period,
        'max_iterations': max_iterations,
        'coefficient_bounds': coefficient_bounds,
        'leakage_weight': leakage_weight,
        'l2_regularization': l2_regularization,
        'num_probes': len(probes)
    }
    if save:
        # Reuse gate result saver with adapted name
        payload = _save_gate_result(f'{gate_name}_process', result, target_U, drive_period, psi_0,
                                    fourier_modes, max_iterations, results_dir)
        # Add metadata not in _save_gate_result
        payload['leakage_weight'] = leakage_weight
        payload['l2_regularization'] = l2_regularization
        payload['coefficient_bounds'] = coefficient_bounds
        payload['num_probes'] = len(probes)
    return payload

def _save_gate_result(name: str, result_obj: Any, target_unitary: Qobj, drive_period: float,
                      initial_state: Qobj, fourier_modes: int, max_iterations: int,
                      results_dir: str) -> Dict[str, Any]:
    """Persist gate optimization metadata/arrays and return a payload with save paths."""
    payload = {
        'optimization_result': result_obj,
        'target_type': name,
        'fourier_modes': fourier_modes,
        'control_period': drive_period,
        'max_iterations': max_iterations
    }
    base_name = (
        f"opt_{name}_modes={fourier_modes}_iters={max_iterations}_cost={result_obj.fun:.3e}_{_timestamp()}"
    )
    meta = {
        'target_type': name,
        'fourier_modes': fourier_modes,
        'max_iterations': max_iterations,
        'control_period': drive_period,
        'time_points': len(time_array),
        'timestamp': _timestamp(),
        'optimization_result': _serialize_optimize_result(result_obj)
    }
    arrays = {
        'optimal_coefficients': result_obj.x,
        'target_unitary_matrix': target_unitary.full()
    }
    paths = _save_optimization_results(base_name, meta, arrays, results_dir=results_dir)
    payload['saved_paths'] = paths
    payload['save_base_name'] = base_name
    return payload

def optimize_hadamard_gate(system_params: Dict[str, float],
                           collapse_ops: List[Qobj],
                           fourier_modes: int = DEFAULT_FOURIER_MODES,
                           max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
                           save: bool = True,
                           results_dir: str = 'results/hadamard',
                           drive_period: Optional[float] = None,
                           initial_state: Optional[Qobj] = None,
                           coefficient_bounds: Optional[Tuple[float,float]] = (-1.0, 1.0),
                           leakage_weight: float = 0.1,
                           l2_regularization: float = 0.0,
                           optimization_method: str = DEFAULT_OPTIMIZATION_METHOD,
                           initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """IQ-only wrapper that delegates to optimize_hadamard_gate_iq."""
    return optimize_hadamard_gate_iq(
        system_params=system_params,
        collapse_ops=collapse_ops,
        fourier_modes=fourier_modes,
        max_iterations=max_iterations,
        save=save,
        results_dir=results_dir,
        drive_period=drive_period,
        coefficient_bounds=coefficient_bounds,
        leakage_weight=leakage_weight,
        l2_regularization=l2_regularization,
        optimization_method=optimization_method,
        initial_guess=initial_guess,
    )

def optimize_s_gate(system_params: Dict[str, float],
                    collapse_ops: List[Qobj],
                    fourier_modes: int = DEFAULT_FOURIER_MODES,
                    max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
                    save: bool = True,
                    results_dir: str = 'results/s_gate',
                    drive_period: Optional[float] = None,
                    initial_state: Optional[Qobj] = None,
                    coefficient_bounds: Optional[Tuple[float,float]] = (-1.0, 1.0),
                    leakage_weight: float = 0.0,
                    l2_regularization: float = 0.0,
                    optimization_method: str = DEFAULT_OPTIMIZATION_METHOD,
                    initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """IQ-only wrapper that delegates to optimize_s_gate_iq."""
    return optimize_s_gate_iq(
        system_params=system_params,
        collapse_ops=collapse_ops,
        fourier_modes=fourier_modes,
        max_iterations=max_iterations,
        save=save,
        results_dir=results_dir,
        drive_period=drive_period,
        coefficient_bounds=coefficient_bounds,
        leakage_weight=leakage_weight,
        l2_regularization=l2_regularization,
        optimization_method=optimization_method,
        initial_guess=initial_guess,
    )

def optimize_t_gate(system_params: Dict[str, float],
                    collapse_ops: List[Qobj],
                    fourier_modes: int = DEFAULT_FOURIER_MODES,
                    max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
                    save: bool = True,
                    results_dir: str = 'results/t_gate',
                    drive_period: Optional[float] = None,
                    initial_state: Optional[Qobj] = None,
                    coefficient_bounds: Optional[Tuple[float,float]] = (-1.0, 1.0),
                    leakage_weight: float = 0.0,
                    l2_regularization: float = 0.0,
                    optimization_method: str = DEFAULT_OPTIMIZATION_METHOD,
                    initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """IQ-only wrapper that delegates to optimize_t_gate_iq."""
    return optimize_t_gate_iq(
        system_params=system_params,
        collapse_ops=collapse_ops,
        fourier_modes=fourier_modes,
        max_iterations=max_iterations,
        save=save,
        results_dir=results_dir,
        drive_period=drive_period,
        coefficient_bounds=coefficient_bounds,
        leakage_weight=leakage_weight,
        l2_regularization=l2_regularization,
        optimization_method=optimization_method,
        initial_guess=initial_guess,
    )

def optimize_hadamard_gate_iq(system_params: Dict[str, float],
                              collapse_ops: List[Qobj],
                              fourier_modes: int = DEFAULT_FOURIER_MODES,
                              max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
                              save: bool = True,
                              results_dir: str = 'results',
                              drive_period: Optional[float] = None,
                              drive_frequency: Optional[float] = None,
                              coefficient_bounds: Optional[Tuple[float,float]] = (-1.0, 1.0),
                              leakage_weight: float = 0.1,
                              l2_regularization: float = 0.0,
                              optimization_method: str = DEFAULT_OPTIMIZATION_METHOD,
                              initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Optimize Hadamard via two-quadrature (IQ) PTM cost. Vector is length 2N (I,Q)."""
    if drive_period is None:
        drive_period = float(time_array[-1])
    if drive_frequency is None:
        drive_frequency = float(QUBIT_FREQUENCY)
    # Hadamard 2x2
    U_qubit = (1/np.sqrt(2)) * Qobj([[1, 1], [1, -1]])
    cost = _ptm_process_cost_iq(
        U_qubit,
        drive_period,
        drive_frequency,
        leakage_weight=leakage_weight,
        l2_reg=l2_regularization,
        system_params=system_params,
        collapse_ops=collapse_ops
    )
    # Optimize 2N vars
    vec_dim = 2 * fourier_modes
    result = global_optimization_multistart(cost, vec_dim, max_iterations, bounds=coefficient_bounds,
                                            optimization_method=optimization_method, initial_guess=initial_guess)
    payload: Dict[str, Any] = {
        'optimization_result': result,
        'target_type': 'hadamard_gate_iq',
        'fourier_modes': fourier_modes,
        'control_period': drive_period,
        'drive_frequency': drive_frequency,
        'max_iterations': max_iterations,
        'coefficient_bounds': coefficient_bounds,
        'leakage_weight': leakage_weight,
        'l2_regularization': l2_regularization,
        'process_cost': 'ptm_iq',
    }
    if save:
        base_name = (
            f"opt_hadamard_gate_iq_modes={fourier_modes}_iters={max_iterations}_cost={result.fun:.3e}_{_timestamp()}"
        )
        meta = {
            'target_type': 'hadamard_gate_iq',
            'fourier_modes': fourier_modes,
            'max_iterations': max_iterations,
            'control_period': drive_period,
            'drive_frequency': drive_frequency,
            'time_points': len(time_array),
            'timestamp': _timestamp(),
            'optimization_result': _serialize_optimize_result(result),
            'process_cost': 'ptm_iq',
            'leakage_weight': leakage_weight,
            'l2_regularization': l2_regularization,
        }
        # Split and save arrays as I and Q
        cI, cQ = _split_iq(result.x)
        arrays = {
            'optimal_coefficients_I': cI,
            'optimal_coefficients_Q': cQ,
        }
        paths = _save_optimization_results(base_name, meta, arrays, results_dir=results_dir)
        payload['saved_paths'] = paths
        payload['save_base_name'] = base_name
    return payload

def optimize_s_gate_iq(system_params: Dict[str, float],
                       collapse_ops: List[Qobj],
                       fourier_modes: int = DEFAULT_FOURIER_MODES,
                       max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
                       save: bool = True,
                       results_dir: str = 'results',
                       drive_period: Optional[float] = None,
                       drive_frequency: Optional[float] = None,
                       coefficient_bounds: Optional[Tuple[float,float]] = (-1.0, 1.0),
                       leakage_weight: float = 0.0,
                       l2_regularization: float = 0.0,
                       optimization_method: str = DEFAULT_OPTIMIZATION_METHOD,
                       initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Optimize S gate via IQ PTM cost; vector length 2N (I,Q)."""
    if drive_period is None:
        drive_period = float(time_array[-1])
    if drive_frequency is None:
        drive_frequency = float(QUBIT_FREQUENCY)
    U_qubit = Qobj([[1, 0], [0, 1j]])
    cost = _ptm_process_cost_iq(U_qubit, drive_period, drive_frequency,
                                leakage_weight=leakage_weight, l2_reg=l2_regularization,
                                system_params=system_params, collapse_ops=collapse_ops)
    vec_dim = 2 * fourier_modes
    result = global_optimization_multistart(cost, vec_dim, max_iterations, bounds=coefficient_bounds,
                                            optimization_method=optimization_method, initial_guess=initial_guess)
    payload: Dict[str, Any] = {
        'optimization_result': result,
        'target_type': 's_gate_iq',
        'fourier_modes': fourier_modes,
        'control_period': drive_period,
        'drive_frequency': drive_frequency,
        'max_iterations': max_iterations,
        'coefficient_bounds': coefficient_bounds,
        'leakage_weight': leakage_weight,
        'l2_regularization': l2_regularization,
        'process_cost': 'ptm_iq',
    }
    if save:
        base_name = (
            f"opt_s_gate_iq_modes={fourier_modes}_iters={max_iterations}_cost={result.fun:.3e}_{_timestamp()}"
        )
        meta = {
            'target_type': 's_gate_iq',
            'fourier_modes': fourier_modes,
            'max_iterations': max_iterations,
            'control_period': drive_period,
            'drive_frequency': drive_frequency,
            'time_points': len(time_array),
            'timestamp': _timestamp(),
            'optimization_result': _serialize_optimize_result(result),
            'process_cost': 'ptm_iq',
            'leakage_weight': leakage_weight,
            'l2_regularization': l2_regularization,
        }
        cI, cQ = _split_iq(result.x)
        arrays = {'optimal_coefficients_I': cI, 'optimal_coefficients_Q': cQ}
        paths = _save_optimization_results(base_name, meta, arrays, results_dir=results_dir)
        payload['saved_paths'] = paths
        payload['save_base_name'] = base_name
    return payload

def optimize_t_gate_iq(system_params: Dict[str, float],
                       collapse_ops: List[Qobj],
                       fourier_modes: int = DEFAULT_FOURIER_MODES,
                       max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
                       save: bool = True,
                       results_dir: str = 'results',
                       drive_period: Optional[float] = None,
                       drive_frequency: Optional[float] = None,
                       coefficient_bounds: Optional[Tuple[float,float]] = (-1.0, 1.0),
                       leakage_weight: float = 0.0,
                       l2_regularization: float = 0.0,
                       optimization_method: str = DEFAULT_OPTIMIZATION_METHOD,
                       initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Optimize T (pi/8) gate via IQ PTM cost; vector length 2N (I,Q)."""
    if drive_period is None:
        drive_period = float(time_array[-1])
    if drive_frequency is None:
        drive_frequency = float(QUBIT_FREQUENCY)
    U_qubit = Qobj([[1, 0], [0, np.exp(1j * np.pi / 4)]])
    cost = _ptm_process_cost_iq(U_qubit, drive_period, drive_frequency,
                                leakage_weight=leakage_weight, l2_reg=l2_regularization,
                                system_params=system_params, collapse_ops=collapse_ops)
    vec_dim = 2 * fourier_modes
    result = global_optimization_multistart(cost, vec_dim, max_iterations, bounds=coefficient_bounds,
                                            optimization_method=optimization_method, initial_guess=initial_guess)
    payload: Dict[str, Any] = {
        'optimization_result': result,
        'target_type': 't_gate_iq',
        'fourier_modes': fourier_modes,
        'control_period': drive_period,
        'drive_frequency': drive_frequency,
        'max_iterations': max_iterations,
        'coefficient_bounds': coefficient_bounds,
        'leakage_weight': leakage_weight,
        'l2_regularization': l2_regularization,
        'process_cost': 'ptm_iq',
    }
    if save:
        base_name = (
            f"opt_t_gate_iq_modes={fourier_modes}_iters={max_iterations}_cost={result.fun:.3e}_{_timestamp()}"
        )
        meta = {
            'target_type': 't_gate_iq',
            'fourier_modes': fourier_modes,
            'max_iterations': max_iterations,
            'control_period': drive_period,
            'drive_frequency': drive_frequency,
            'time_points': len(time_array),
            'timestamp': _timestamp(),
            'optimization_result': _serialize_optimize_result(result),
            'process_cost': 'ptm_iq',
            'leakage_weight': leakage_weight,
            'l2_regularization': l2_regularization,
        }
        cI, cQ = _split_iq(result.x)
        arrays = {'optimal_coefficients_I': cI, 'optimal_coefficients_Q': cQ}
        paths = _save_optimization_results(base_name, meta, arrays, results_dir=results_dir)
        payload['saved_paths'] = paths
        payload['save_base_name'] = base_name
    return payload

# =============================
# New: X and Y Gate Optimization
# =============================

def optimize_x_gate(system_params: Dict[str, float],
                    collapse_ops: List[Qobj],
                    fourier_modes: int = DEFAULT_FOURIER_MODES,
                    max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
                    save: bool = True,
                    results_dir: str = 'results/x_gate',
                    drive_period: Optional[float] = None,
                    initial_state: Optional[Qobj] = None,
                    coefficient_bounds: Optional[Tuple[float,float]] = (-1.0, 1.0),
                    leakage_weight: float = 0.1,
                    l2_regularization: float = 0.0,
                    optimization_method: str = DEFAULT_OPTIMIZATION_METHOD,
                    initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """IQ-only wrapper that delegates to optimize_x_gate_iq."""
    return optimize_x_gate_iq(
        system_params=system_params,
        collapse_ops=collapse_ops,
        fourier_modes=fourier_modes,
        max_iterations=max_iterations,
        save=save,
        results_dir=results_dir,
        drive_period=drive_period,
        coefficient_bounds=coefficient_bounds,
        leakage_weight=leakage_weight,
        l2_regularization=l2_regularization,
        optimization_method=optimization_method,
        initial_guess=initial_guess,
    )

def optimize_y_gate(system_params: Dict[str, float],
                    collapse_ops: List[Qobj],
                    fourier_modes: int = DEFAULT_FOURIER_MODES,
                    max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
                    save: bool = True,
                    results_dir: str = 'results/y_gate',
                    drive_period: Optional[float] = None,
                    initial_state: Optional[Qobj] = None,
                    coefficient_bounds: Optional[Tuple[float,float]] = (-1.0, 1.0),
                    leakage_weight: float = 0.1,
                    l2_regularization: float = 0.0,
                    optimization_method: str = DEFAULT_OPTIMIZATION_METHOD,
                    initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """IQ-only wrapper that delegates to optimize_y_gate_iq."""
    return optimize_y_gate_iq(
        system_params=system_params,
        collapse_ops=collapse_ops,
        fourier_modes=fourier_modes,
        max_iterations=max_iterations,
        save=save,
        results_dir=results_dir,
        drive_period=drive_period,
        coefficient_bounds=coefficient_bounds,
        leakage_weight=leakage_weight,
        l2_regularization=l2_regularization,
        optimization_method=optimization_method,
        initial_guess=initial_guess,
    )

def optimize_x_gate_iq(system_params: Dict[str, float],
                       collapse_ops: List[Qobj],
                       fourier_modes: int = DEFAULT_FOURIER_MODES,
                       max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
                       save: bool = True,
                       results_dir: str = 'results',
                       drive_period: Optional[float] = None,
                       drive_frequency: Optional[float] = None,
                       coefficient_bounds: Optional[Tuple[float,float]] = (-1.0, 1.0),
                       leakage_weight: float = 0.1,
                       l2_regularization: float = 0.0,
                       optimization_method: str = DEFAULT_OPTIMIZATION_METHOD,
                       initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Optimize X (π rotation around X) via IQ PTM cost; vector length 2N (I,Q)."""
    if drive_period is None:
        drive_period = float(time_array[-1])
    if drive_frequency is None:
        drive_frequency = float(QUBIT_FREQUENCY)
    # X gate up to global phase is σ_x
    U_qubit = sigmax()
    cost = _ptm_process_cost_iq(U_qubit, drive_period, drive_frequency,
                                leakage_weight=leakage_weight, l2_reg=l2_regularization,
                                system_params=system_params, collapse_ops=collapse_ops)
    vec_dim = 2 * fourier_modes
    result, nfev = global_optimization_multistart(cost, vec_dim, max_iterations, bounds=coefficient_bounds,
                                            optimization_method=optimization_method, initial_guess=initial_guess)
    payload: Dict[str, Any] = {
        'optimization_result': result,
        'target_type': 'x_gate_iq',
        'fourier_modes': fourier_modes,
        'control_period': drive_period,
        'drive_frequency': drive_frequency,
        'max_iterations': max_iterations,
        'nfev': nfev,
        'coefficient_bounds': coefficient_bounds,
        'leakage_weight': leakage_weight,
        'l2_regularization': l2_regularization,
        'process_cost': 'ptm_iq',
    }
    if save:
        base_name = (
            f"opt_x_gate_iq_modes={fourier_modes}_iters={max_iterations}_cost={result.fun:.3e}_{_timestamp()}"
        )
        meta = {
            'target_type': 'x_gate_iq',
            'fourier_modes': fourier_modes,
            'max_iterations': max_iterations,
            'control_period': drive_period,
            'drive_frequency': drive_frequency,
            'nfev': nfev,
            'time_points': len(time_array),
            'timestamp': _timestamp(),
            'optimization_result': _serialize_optimize_result(result),
            'process_cost': 'ptm_iq',
            'leakage_weight': leakage_weight,
            'l2_regularization': l2_regularization,
        }
        cI, cQ = _split_iq(result.x)
        arrays = {'optimal_coefficients_I': cI, 'optimal_coefficients_Q': cQ}
        paths = _save_optimization_results(base_name, meta, arrays, results_dir=results_dir)
        payload['saved_paths'] = paths
        payload['save_base_name'] = base_name
    return payload

def optimize_y_gate_iq(system_params: Dict[str, float],
                       collapse_ops: List[Qobj],
                       fourier_modes: int = DEFAULT_FOURIER_MODES,
                       max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
                       save: bool = True,
                       results_dir: str = 'results',
                       drive_period: Optional[float] = None,
                       drive_frequency: Optional[float] = None,
                       coefficient_bounds: Optional[Tuple[float,float]] = (-1.0, 1.0),
                       leakage_weight: float = 0.1,
                       l2_regularization: float = 0.0,
                       optimization_method: str = DEFAULT_OPTIMIZATION_METHOD,
                       initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Optimize Y (π rotation around Y) via IQ PTM cost; vector length 2N (I,Q)."""
    if drive_period is None:
        drive_period = float(time_array[-1])
    if drive_frequency is None:
        drive_frequency = float(QUBIT_FREQUENCY)
    # Y gate up to global phase is σ_y
    U_qubit = sigmay()
    cost = _ptm_process_cost_iq(U_qubit, drive_period, drive_frequency,
                                leakage_weight=leakage_weight, l2_reg=l2_regularization,
                                system_params=system_params, collapse_ops=collapse_ops)
    vec_dim = 2 * fourier_modes
    result, nfev = global_optimization_multistart(cost, vec_dim, max_iterations, bounds=coefficient_bounds,
                                            optimization_method=optimization_method, initial_guess=initial_guess)
    payload: Dict[str, Any] = {
        'optimization_result': result,
        'target_type': 'y_gate_iq',
        'fourier_modes': fourier_modes,
        'control_period': drive_period,
        'drive_frequency': drive_frequency,
        'nfev': nfev,
        'max_iterations': max_iterations,
        'coefficient_bounds': coefficient_bounds,
        'leakage_weight': leakage_weight,
        'l2_regularization': l2_regularization,
        'process_cost': 'ptm_iq',
    }
    if save:
        base_name = (
            f"opt_y_gate_iq_modes={fourier_modes}_iters={max_iterations}_cost={result.fun:.3e}_{_timestamp()}"
        )
        meta = {
            'target_type': 'y_gate_iq',
            'fourier_modes': fourier_modes,
            'max_iterations': max_iterations,
            'control_period': drive_period,
            'drive_frequency': drive_frequency,
            'time_points': len(time_array),
            'timestamp': _timestamp(),
            'optimization_result': _serialize_optimize_result(result),
            'process_cost': 'ptm_iq',
            'leakage_weight': leakage_weight,
            'l2_regularization': l2_regularization,
        }
        cI, cQ = _split_iq(result.x)
        arrays = {'optimal_coefficients_I': cI, 'optimal_coefficients_Q': cQ}
        paths = _save_optimization_results(base_name, meta, arrays, results_dir=results_dir)
        payload['saved_paths'] = paths
        payload['save_base_name'] = base_name
    return payload

# =============================
# New: √X (X π/2) and √Y (Y π/2) Gate Optimization
# =============================

def optimize_x90_gate(system_params: Dict[str, float],
                      collapse_ops: List[Qobj],
                      fourier_modes: int = DEFAULT_FOURIER_MODES,
                      max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
                      save: bool = True,
                      results_dir: str = 'results/x90_gate',
                      drive_period: Optional[float] = None,
                      initial_state: Optional[Qobj] = None,
                      coefficient_bounds: Optional[Tuple[float,float]] = (-1.0, 1.0),
                      leakage_weight: float = 0.1,
                      l2_regularization: float = 0.0,
                      optimization_method: str = DEFAULT_OPTIMIZATION_METHOD,
                      initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """IQ-only wrapper that delegates to optimize_x90_gate_iq (√X)."""
    return optimize_x90_gate_iq(
        system_params=system_params,
        collapse_ops=collapse_ops,
        fourier_modes=fourier_modes,
        max_iterations=max_iterations,
        save=save,
        results_dir=results_dir,
        drive_period=drive_period,
        coefficient_bounds=coefficient_bounds,
        leakage_weight=leakage_weight,
        l2_regularization=l2_regularization,
        optimization_method=optimization_method,
        initial_guess=initial_guess,
    )

def optimize_y90_gate(system_params: Dict[str, float],
                      collapse_ops: List[Qobj],
                      fourier_modes: int = DEFAULT_FOURIER_MODES,
                      max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
                      save: bool = True,
                      results_dir: str = 'results/y90_gate',
                      drive_period: Optional[float] = None,
                      initial_state: Optional[Qobj] = None,
                      coefficient_bounds: Optional[Tuple[float,float]] = (-1.0, 1.0),
                      leakage_weight: float = 0.1,
                      l2_regularization: float = 0.0,
                      optimization_method: str = DEFAULT_OPTIMIZATION_METHOD,
                      initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """IQ-only wrapper that delegates to optimize_y90_gate_iq (√Y)."""
    return optimize_y90_gate_iq(
        system_params=system_params,
        collapse_ops=collapse_ops,
        fourier_modes=fourier_modes,
        max_iterations=max_iterations,
        save=save,
        results_dir=results_dir,
        drive_period=drive_period,
        coefficient_bounds=coefficient_bounds,
        leakage_weight=leakage_weight,
        l2_regularization=l2_regularization,
        optimization_method=optimization_method,
        initial_guess=initial_guess,
    )

def _rotation_unitary(axis: str, theta: float) -> Qobj:
    """Return 2x2 unitary for rotation R_axis(theta) = exp(-i theta σ_axis / 2)."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    if axis.lower() == 'x':
        return c * qeye(2) - 1j * s * sigmax()
    if axis.lower() == 'y':
        return c * qeye(2) - 1j * s * sigmay()
    if axis.lower() == 'z':
        return c * qeye(2) - 1j * s * sigmaz()
    raise ValueError("axis must be one of {'x','y','z'}")

def optimize_x90_gate_iq(system_params: Dict[str, float],
                         collapse_ops: List[Qobj],
                         fourier_modes: int = DEFAULT_FOURIER_MODES,
                         max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
                         save: bool = True,
                         results_dir: str = 'results',
                         drive_period: Optional[float] = None,
                         drive_frequency: Optional[float] = None,
                         coefficient_bounds: Optional[Tuple[float,float]] = (-1.0, 1.0),
                         leakage_weight: float = 0.1,
                         l2_regularization: float = 0.0,
                         optimization_method: str = DEFAULT_OPTIMIZATION_METHOD,
                         initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Optimize √X (RX(π/2)) via IQ PTM cost; vector length 2N (I,Q)."""
    if drive_period is None:
        drive_period = float(time_array[-1])
    if drive_frequency is None:
        drive_frequency = float(QUBIT_FREQUENCY)
    U_qubit = _rotation_unitary('x', np.pi / 2.0)
    cost = _ptm_process_cost_iq(U_qubit, drive_period, drive_frequency,
                                leakage_weight=leakage_weight, l2_reg=l2_regularization,
                                system_params=system_params, collapse_ops=collapse_ops)
    vec_dim = 2 * fourier_modes
    result, nfev = global_optimization_multistart(cost, vec_dim, max_iterations, bounds=coefficient_bounds,
                                            optimization_method=optimization_method, initial_guess=initial_guess)
    payload: Dict[str, Any] = {
        'optimization_result': result,
        'target_type': 'x90_gate_iq',
        'fourier_modes': fourier_modes,
        'control_period': drive_period,
        'drive_frequency': drive_frequency,
        'nfev': nfev,
        'max_iterations': max_iterations,
        'coefficient_bounds': coefficient_bounds,
        'leakage_weight': leakage_weight,
        'l2_regularization': l2_regularization,
        'process_cost': 'ptm_iq',
    }
    if save:
        base_name = (
            f"opt_x90_gate_iq_modes={fourier_modes}_iters={max_iterations}_cost={result.fun:.3e}_{_timestamp()}"
        )
        meta = {
            'target_type': 'x90_gate_iq',
            'fourier_modes': fourier_modes,
            'max_iterations': max_iterations,
            'control_period': drive_period,
            'drive_frequency': drive_frequency,
            'time_points': len(time_array),
            'timestamp': _timestamp(),
            'optimization_result': _serialize_optimize_result(result),
            'process_cost': 'ptm_iq',
            'leakage_weight': leakage_weight,
            'l2_regularization': l2_regularization,
        }
        cI, cQ = _split_iq(result.x)
        arrays = {'optimal_coefficients_I': cI, 'optimal_coefficients_Q': cQ}
        paths = _save_optimization_results(base_name, meta, arrays, results_dir=results_dir)
        payload['saved_paths'] = paths
        payload['save_base_name'] = base_name
    return payload

def optimize_y90_gate_iq(system_params: Dict[str, float],
                         collapse_ops: List[Qobj],
                         fourier_modes: int = DEFAULT_FOURIER_MODES,
                         max_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
                         save: bool = True,
                         results_dir: str = 'results',
                         drive_period: Optional[float] = None,
                         drive_frequency: Optional[float] = None,
                         coefficient_bounds: Optional[Tuple[float,float]] = (-1.0, 1.0),
                         leakage_weight: float = 0.1,
                         l2_regularization: float = 0.0,
                         optimization_method: str = DEFAULT_OPTIMIZATION_METHOD,
                         initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Optimize √Y (RY(π/2)) via IQ PTM cost; vector length 2N (I,Q)."""
    if drive_period is None:
        drive_period = float(time_array[-1])
    if drive_frequency is None:
        drive_frequency = float(QUBIT_FREQUENCY)
    U_qubit = _rotation_unitary('y', np.pi / 2.0)
    cost = _ptm_process_cost_iq(U_qubit, drive_period, drive_frequency,
                                leakage_weight=leakage_weight, l2_reg=l2_regularization,
                                system_params=system_params, collapse_ops=collapse_ops)
    vec_dim = 2 * fourier_modes
    result, nfev = global_optimization_multistart(cost, vec_dim, max_iterations, bounds=coefficient_bounds,
                                            optimization_method=optimization_method, initial_guess=initial_guess)
    payload: Dict[str, Any] = {
        'optimization_result': result,
        'target_type': 'y90_gate_iq',
        'fourier_modes': fourier_modes,
        'control_period': drive_period,
        'drive_frequency': drive_frequency,
        'nfev': nfev,
        'max_iterations': max_iterations,
        'coefficient_bounds': coefficient_bounds,
        'leakage_weight': leakage_weight,
        'l2_regularization': l2_regularization,
        'process_cost': 'ptm_iq',
    }
    if save:
        base_name = (
            f"opt_y90_gate_iq_modes={fourier_modes}_iters={max_iterations}_cost={result.fun:.3e}_{_timestamp()}"
        )
        meta = {
            'target_type': 'y90_gate_iq',
            'fourier_modes': fourier_modes,
            'max_iterations': max_iterations,
            'control_period': drive_period,
            'drive_frequency': drive_frequency,
            'time_points': len(time_array),
            'timestamp': _timestamp(),
            'optimization_result': _serialize_optimize_result(result),
            'process_cost': 'ptm_iq',
            'leakage_weight': leakage_weight,
            'l2_regularization': l2_regularization,
        }
        cI, cQ = _split_iq(result.x)
        arrays = {'optimal_coefficients_I': cI, 'optimal_coefficients_Q': cQ}
        paths = _save_optimization_results(base_name, meta, arrays, results_dir=results_dir)
        payload['saved_paths'] = paths
        payload['save_base_name'] = base_name
    return payload
