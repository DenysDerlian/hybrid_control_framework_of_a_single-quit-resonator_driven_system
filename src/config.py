"""
Configuration module for quantum control optimization.

This module contains all global constants, physical parameters, and configuration
settings used throughout the quantum control optimization framework.

All physical parameters are in units where ℏ = 1.
"""

import os
import numbers
from typing import Sequence, List, Union
import numpy as np

# ==============================================================================
# PHYSICAL SYSTEM PARAMETERS
# ==============================================================================

# Quantum system dimensions
NUM_FOCK_STATES = 5                  # Truncation of cavity Hilbert space

# New: default number of qubits (backward compatible)
# This controls multi-qubit operator/state construction when supported.
NUM_QUBITS: int = int(os.getenv("NUM_QUBITS", "1"))

# Physical frequencies (in units where ℏ = 1)
CAVITY_FREQUENCY = 3.0 * 2 * np.pi      # ωR: Resonator frequency (GHz)
QUBIT_FREQUENCY = 1.0 * 2 * np.pi       # ωT: Transmon frequency (GHz)
COUPLING_STRENGTH = 0.10 * 2 * np.pi    # g: Qubit-cavity coupling (GHz)

# Optional multi-qubit defaults (kept for forward-compatibility). These are not
# used by single-qubit paths and can be leveraged by multi-qubit extensions.
try:
    QUBIT_FREQUENCIES: Sequence[float] = (float(QUBIT_FREQUENCY),)  # type: ignore
except NameError:
    QUBIT_FREQUENCIES = ()

try:
    COUPLING_STRENGTHS: Sequence[float] = (float(COUPLING_STRENGTH),)  # type: ignore
except NameError:
    COUPLING_STRENGTHS = ()

# ==============================================================================
# DISSIPATIVE PARAMETERS
# ==============================================================================

# Dissipation rates (in units where ℏ = 1)
CAVITY_DISSIPATION = 1e-3 * 2 * np.pi    # κ: Cavity dissipation rate (GHz)
QUBIT_DISSIPATION = 1e-6 * 2 * np.pi     # γ: Qubit dissipation rate (GHz)
QUBIT_DEPHASING = 1e-6 * 2 * np.pi      # φ: Qubit dephasing rate (GHz)

# ==============================================================================
# TIME EVOLUTION PARAMETERS
# ==============================================================================

# Temporal discretization
# TIME_POINTS = 201                       # Number of time discretization points
# TOTAL_TIME = 10.0                       # Total evolution time (ns)
TIME_POINTS = 252
TOTAL_TIME = TIME_POINTS * (1/QUBIT_FREQUENCY / 4)

# Time array for all simulations
time_array = np.linspace(0, TOTAL_TIME, TIME_POINTS)

# ==============================================================================
# OPTIMIZATION PARAMETERS
# ==============================================================================

# Default optimization settings
DEFAULT_FOURIER_MODES = 32              # Default number of Fourier modes
DEFAULT_OPTIMIZATION_ITERATIONS = 6     # Default number of random starts
DEFAULT_OPTIMIZATION_METHOD = "L-BFGS-B"  # Default optimization algorithm

# Optimization bounds
COEFFICIENT_LOWER_BOUND = -2.0           # Lower bound for Fourier coefficients
COEFFICIENT_UPPER_BOUND = 2.0          # Upper bound for Fourier coefficients

# ==============================================================================
# PHYSICAL CONSTANTS AND DERIVED PARAMETERS
# ==============================================================================

# Drive system parameters (estimated values)
DRIVE_COUPLING_BETA = 0.2               # β = Cg/(CΣ + Cg)
ZERO_POINT_CHARGE = 1e-18               # Q_zpf ≈ sqrt(ℏC·ωr/2)

# ==============================================================================
# COMPUTATIONAL PARAMETERS
# ==============================================================================

# Numerical tolerances
EVOLUTION_TOLERANCE = 1e-8              # Tolerance for time evolution solver
OPTIMIZATION_TOLERANCE = 1e-6           # Tolerance for optimization convergence

# Performance settings
USE_NUMBA_JIT = True                    # Enable Numba just-in-time compilation
NUMBA_CACHE = True                      # Enable Numba compilation caching

# ==============================================================================
# VISUALIZATION PARAMETERS
# ==============================================================================

# Default figure parameters
DEFAULT_FIGURE_WIDTH_CM = 23            # Default figure width in cm
DEFAULT_FIGURE_HEIGHT_CM = 28           # Default figure height in cm
DEFAULT_DPI = 300                       # Default resolution for saved figures

# Font sizes for academic publications
FONT_SIZES = {
    'title': 16,
    'label': 14,
    'tick': 12,
    'legend': 12
}

# Color scheme for plots
PLOT_COLORS = {
    'controlled_evolution': 'forestgreen',
    'target_function': 'royalblue',
    'control_real': 'darkred',
    'control_imag': 'darkblue',
    'error': 'crimson'
}

# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def validate_system_parameters():
    """
    Validate that all system parameters are physically reasonable.
    
    Raises:
    -------
    ValueError
        If any parameter is outside expected physical range
    """
    if CAVITY_FREQUENCY <= 0:
        raise ValueError("Cavity frequency must be positive")
    
    if QUBIT_FREQUENCY <= 0:
        raise ValueError("Qubit frequency must be positive")
        
    if COUPLING_STRENGTH <= 0:
        raise ValueError("Coupling strength must be positive")
        
    if NUM_FOCK_STATES < 2:
        raise ValueError("Number of Fock states must be at least 2")
        
    if TIME_POINTS < 10:
        raise ValueError("Number of time points must be at least 10")
        
    if TOTAL_TIME <= 0:
        raise ValueError("Total evolution time must be positive")

def get_system_info():
    """
    Return a formatted string with system parameter information.
    
    Returns:
    --------
    str
        Formatted system information
    """
    info = f"""
    Quantum Control System Configuration:
    =====================================
    
    Physical Parameters:
    - Cavity frequency: {CAVITY_FREQUENCY/(2*np.pi):.3f} GHz
    - Qubit frequency: {QUBIT_FREQUENCY/(2*np.pi):.3f} GHz
    - Coupling strength: {COUPLING_STRENGTH/(2*np.pi):.4f} GHz
    - Hilbert space (single-qubit): {NUM_FOCK_STATES} × 2 = {NUM_FOCK_STATES * 2} dims
    - Default number of qubits: {NUM_QUBITS}
    
    Time Evolution:
    - Time points: {TIME_POINTS}
    - Total time: {TOTAL_TIME} ns
    - Time step: {TOTAL_TIME/TIME_POINTS:.3f} ns
    
    Optimization:
    - Default Fourier modes: {DEFAULT_FOURIER_MODES}
    - Default iterations: {DEFAULT_OPTIMIZATION_ITERATIONS}
    - Method: {DEFAULT_OPTIMIZATION_METHOD}
    """
    return info

# Validate parameters on import
validate_system_parameters()

# -----------------------------------------------------------------------------
# Utility helpers (forward-compat for multi-qubit refactors)
# -----------------------------------------------------------------------------

def _as_list(x: Union[float, int, Sequence[float]], n: int) -> List[float]:
    """Normalize a scalar or sequence to a length-n list of floats.

    Backward-compatible helper: used only by multi-qubit-aware code paths.
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        lst = list(x)
        if len(lst) != n:
            raise ValueError(f'Length mismatch: expected {n}, got {len(lst)}')
        return [float(v) for v in lst]
    if isinstance(x, numbers.Real):
        return [float(x)] * n
    raise TypeError("x must be a real number or a sequence of real numbers")
