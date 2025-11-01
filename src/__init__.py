"""
Quantum Control Optimization Package

A comprehensive framework for optimizing control fields in driven qubit systems
using Fourier series expansion and global optimization techniques.

This package provides modular components for:
- Quantum system initialization and operator construction
- Time-dependent Hamiltonian dynamics simulation
- Control field optimization using various target functions
- Publication-quality visualization and analysis

Author: Research Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Research Team"

# Import main components for easy access
from .config import *
from .states import get_initial_state
from .operators import create_operators, construct_system_hamiltonian
from .dynamics import (
    simulate_controlled_evolution,
    simulate_controlled_evolution_iq,
    compute_iq_control_fields,
    simulate_controlled_evolution_iq_multi,
    compute_iq_control_fields_multi,
)
from .optimization import (
    global_optimization_multistart,
    optimize_sinusoidal_target,
    optimize_gaussian_target,
    optimize_rectangular_target,
    optimize_hadamard_gate,
    optimize_s_gate,
    optimize_t_gate,
    optimize_single_qubit_gate_process,
    optimize_multi_qubit_mse,
    optimize_multi_qubit_final_excitation,
)
from .benchmarking import (
    sequence_fidelity_scan,
    fit_exponential_decay,
    benchmarking_summary,
    default_probe_states,
    run_sequence_benchmark
)
from .plotting import QuantumControlPlotter
from .pulse_shapes import (
    generate_sinusoidal_target,
    generate_gaussian_target, 
    generate_rectangular_target
)
from .utils import build_iq_vector, split_iq_vector, build_iq_vector_nd, split_iq_vector_nd

__all__ = [
    # Configuration
    'NUM_FOCK_STATES', 'CAVITY_FREQUENCY', 'QUBIT_FREQUENCY', 'COUPLING_STRENGTH',
    'TIME_POINTS', 'TOTAL_TIME', 'time_array', 'DEFAULT_FOURIER_MODES',
    
    # Core functions
    'get_initial_state',
    'create_operators', 'construct_system_hamiltonian',
    'simulate_controlled_evolution', 'simulate_controlled_evolution_iq', 'compute_iq_control_fields',
    'simulate_controlled_evolution_iq_multi', 'compute_iq_control_fields_multi',
    'global_optimization_multistart',
    'optimize_sinusoidal_target', 'optimize_gaussian_target', 'optimize_rectangular_target',
    'optimize_hadamard_gate', 'optimize_s_gate', 'optimize_t_gate', 'optimize_single_qubit_gate_process',
    'optimize_multi_qubit_mse', 'optimize_multi_qubit_final_excitation',
    'sequence_fidelity_scan', 'fit_exponential_decay', 'benchmarking_summary', 'default_probe_states', 'run_sequence_benchmark',
    'QuantumControlPlotter',
    
    # Target functions
    'generate_sinusoidal_target', 'generate_gaussian_target', 'generate_rectangular_target',
    # Utils
    'build_iq_vector', 'split_iq_vector', 'build_iq_vector_nd', 'split_iq_vector_nd'
]
