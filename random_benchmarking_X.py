
# Ensure latest package (exports run_sequence_benchmark) is loaded
import importlib, src
import src.benchmarking as bm
importlib.reload(bm)
importlib.reload(src)
from src import run_sequence_benchmark, benchmarking_summary

# Helper to display summary nicely
from pprint import pprint
import numpy as np

# Import the modular quantum control framework
import numpy as np
import matplotlib.pyplot as plt
import time
import importlib
import src.benchmarking as bm
from src import run_sequence_benchmark, benchmarking_summary
from pprint import pprint
import pandas as pd
from qutip import Bloch, ptrace, basis, sigmax, sigmay, sigmaz, expect
import numpy as np
import time

# Import our modular framework components
from src import (
    # Configuration and constants
    time_array, DEFAULT_FOURIER_MODES, QUBIT_FREQUENCY, CAVITY_FREQUENCY, COUPLING_STRENGTH,
    CAVITY_DISSIPATION, QUBIT_DISSIPATION, QUBIT_DEPHASING,
    
    # Core quantum mechanics
    get_initial_state, create_operators, construct_system_hamiltonian,
    simulate_controlled_evolution,
    
    # Visualization
    QuantumControlPlotter,
)

from src.operators import get_collapse_operators
from src.dynamics import simulate_controlled_evolution, compute_fourier_control_field, simulate_controlled_evolution_iq, compute_iq_control_fields
from src.utils import build_iq_vector, split_iq_vector
import src.optimization as _opt

# Initialize plotter (disable LaTeX for portability here)
plotter = QuantumControlPlotter(show_plots=True, use_latex=False, figure_format='png')

print(f"Quantum Control Framework Initialized")
print(f"System Parameters:")
print(f"  Cavity frequency: {CAVITY_FREQUENCY/(2*np.pi):.3f} GHz")
print(f"  Qubit frequency: {QUBIT_FREQUENCY/(2*np.pi):.3f} GHz")
print(f"  Coupling strength: {COUPLING_STRENGTH/(2*np.pi):.3f} GHz")
print(f"  Evolution time: {time_array[-1]:.1f} ns")
print(f"  Fourier modes: {DEFAULT_FOURIER_MODES}")

# Initialize quantum system components
initial_state = get_initial_state()
operators = create_operators()
collapse_ops = get_collapse_operators(gamma_cavity=CAVITY_DISSIPATION, # ~ 0.1-5 MHz
                                      gamma_qubit=QUBIT_DISSIPATION, # ~ 1-10 KHz
                                      gamma_dephasing=QUBIT_DEPHASING # ~ 1-20 KHz
)

print(f"Initial state: Ground state |g,0⟩")
print(f"Hilbert space dimension: {initial_state.shape[0]}")
print(f"System operators created: {len(operators)} operators")

# System parameters for dynamics simulation
system_params = {
    'omega_r': CAVITY_FREQUENCY,
    'omega_t': QUBIT_FREQUENCY,
    'coupling_g': COUPLING_STRENGTH
}



x_opt_payload = np.load(f'data/payloads/x_gate/x_opt_res.npz', allow_pickle=True)['x_opt_res'].item()
x_opt_payload_NN = np.load(f'data/payloads/x_gate/x_opt_res_NN.npz', allow_pickle=True)['x_opt_res_NN'].item()
x_opt_payload_nm = np.load(f'data/payloads/x_gate/x_opt_res_nm.npz', allow_pickle=True)['x_opt_res_nm'].item()
x_opt_payload_nm_NN = np.load(f'data/payloads/x_gate/x_opt_res_nm_NN.npz', allow_pickle=True)['x_opt_res_nm_NN'].item()

x_opt_payload_list = [x_opt_payload, x_opt_payload_NN, x_opt_payload_nm, x_opt_payload_nm_NN]

for i, payload in enumerate(x_opt_payload_list):
    print(f'\nBenchmarking result for payload {i}:')
    x_bench = run_sequence_benchmark(system_params=system_params, collapse_ops=collapse_ops,
                                    optimized_payload=payload, gate_name='x_gate', 
                                    sequence_lengths=np.arange(1,1000,1), save=True)
    print(benchmarking_summary(x_bench['scan_result'], x_bench['fit_params']))

    m = x_bench['scan_result']['m']
    Fm = x_bench['scan_result']['avg_fidelity']
    A = x_bench['fit_params']['A']
    p = x_bench['fit_params']['p']
    B = x_bench['fit_params']['B']
    plt.figure(figsize=(5,3.2))
    plt.plot(m, Fm, 'o', label='Avg fidelity')
    plt.plot(m, A * (p ** m) + B, '-', label=f'Fit p={p:.6f}')
    plt.xlabel('Sequence length m')
    plt.ylabel('Average fidelity')
    plt.ylim(0,1.02)
    plt.title(f'X gate sequence fidelity decay for payload {i}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print('\nSaved benchmark artifacts:' if 'saved_paths' in x_bench else 'Benchmark not saved.')
    if 'saved_paths' in x_bench:
        pprint(x_bench['saved_paths'])
