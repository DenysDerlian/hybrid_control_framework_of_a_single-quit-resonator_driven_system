"""
Quantum state initialization module.

This module provides functions to create and manipulate quantum states
for the driven qubit-cavity system.
"""

from qutip import (
    fock, fock_dm, tensor, coherent, thermal_dm, qeye, destroy,
    expect, ptrace, entropy_vn, fidelity, Qobj
)
import numpy as np
from .config import NUM_FOCK_STATES, NUM_QUBITS

def get_initial_state(num_qubits: int = NUM_QUBITS, excited_qubits: list[int] | None = None):
    """
    Create the initial quantum state for the system.

    Default is vacuum cavity and all qubits in ground state. For backward
    compatibility, when num_qubits==1 the state matches the legacy |0⟩⊗|g⟩.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (default comes from config.NUM_QUBITS)
    excited_qubits : list[int] | None
        Indices of qubits prepared in |e⟩ instead of |g⟩.
    """
    cavity_vacuum = fock(NUM_FOCK_STATES, 0)
    ex = set(excited_qubits or [])
    qubits = [fock(2, 1 if j in ex else 0) for j in range(int(num_qubits))]
    return tensor(cavity_vacuum, *qubits)

def create_fock_state(cavity_n: int, qubit_state: int):
    """
    Create a Fock state with n photons in cavity and qubit in specified state.
    
    Parameters:
    -----------
    cavity_n : int
        Number of photons in the cavity (0 ≤ n < NUM_FOCK_STATES)
    qubit_state : int
        Qubit state (0 for ground, 1 for excited)
        
    Returns:
    --------
    Qobj
        Fock state |n⟩_cavity ⊗ |state⟩_qubit
        
    Raises:
    -------
    ValueError
        If cavity_n is outside valid range or qubit_state is not 0 or 1
    """
    if not (0 <= cavity_n < NUM_FOCK_STATES):
        raise ValueError(f"cavity_n must be between 0 and {NUM_FOCK_STATES-1}")
    
    if qubit_state not in [0, 1]:
        raise ValueError("qubit_state must be 0 (ground) or 1 (excited)")
    
    cavity_state = fock(NUM_FOCK_STATES, cavity_n)
    qubit_state_obj = fock(2, qubit_state)
    
    return tensor(cavity_state, qubit_state_obj)

def create_coherent_state(alpha: complex, qubit_state: int = 0):
    """
    Create a coherent state in the cavity with qubit in specified state.
    
    Parameters:
    -----------
    alpha : complex
        Complex amplitude of the coherent state
    qubit_state : int, default=0
        Qubit state (0 for ground, 1 for excited)
        
    Returns:
    --------
    Qobj
        Coherent state |α⟩_cavity ⊗ |state⟩_qubit
    """
    if qubit_state not in [0, 1]:
        raise ValueError("qubit_state must be 0 (ground) or 1 (excited)")
    
    cavity_coherent = coherent(NUM_FOCK_STATES, alpha)
    qubit_state_obj = fock(2, qubit_state)
    
    return tensor(cavity_coherent, qubit_state_obj)

def create_superposition_state(cavity_amplitudes: np.ndarray, qubit_amplitudes: np.ndarray):
    """
    Create a general superposition state of the cavity-qubit system.
    
    Parameters:
    -----------
    cavity_amplitudes : np.ndarray
        Complex amplitudes for cavity Fock states [c₀, c₁, ..., cₙ]
        Length must be ≤ NUM_FOCK_STATES
    qubit_amplitudes : np.ndarray
        Complex amplitudes for qubit states [ground_amp, excited_amp]
        
    Returns:
    --------
    Qobj
        Superposition state Σᵢⱼ cᵢⱼ |i⟩_cavity ⊗ |j⟩_qubit
        
    Notes:
    ------
    The state is automatically normalized.
    """
    if len(cavity_amplitudes) > NUM_FOCK_STATES:
        raise ValueError(f"Too many cavity amplitudes (max {NUM_FOCK_STATES})")
    
    if len(qubit_amplitudes) != 2:
        raise ValueError("Qubit amplitudes must have exactly 2 elements")
    
    # Pad cavity amplitudes if necessary
    if len(cavity_amplitudes) < NUM_FOCK_STATES:
        padded_amplitudes = np.zeros(NUM_FOCK_STATES, dtype=complex)
        padded_amplitudes[:len(cavity_amplitudes)] = cavity_amplitudes
        cavity_amplitudes = padded_amplitudes
    
    # Create cavity superposition
    cavity_state = sum(amp * fock(NUM_FOCK_STATES, i) 
                      for i, amp in enumerate(cavity_amplitudes) if amp != 0)
    
    # Create qubit superposition
    qubit_state = sum(amp * fock(2, i) 
                     for i, amp in enumerate(qubit_amplitudes) if amp != 0)
    
    # Create composite state
    composite_state = tensor(cavity_state, qubit_state)
    
    # Normalize the state
    return composite_state.unit()

def get_thermal_state(n_thermal: float, qubit_state: int = 0):
    """
    Create a thermal state of the cavity with specified mean photon number.
    
    Parameters:
    -----------
    n_thermal : float
        Mean photon number in thermal state
    qubit_state : int, default=0
        Qubit state (0 for ground, 1 for excited)
        
    Returns:
    --------
    Qobj
        Thermal state (density matrix) of cavity ⊗ qubit state
    """
    if n_thermal < 0:
        raise ValueError("Mean photon number must be non-negative")
    
    if qubit_state not in [0, 1]:
        raise ValueError("qubit_state must be 0 (ground) or 1 (excited)")
    
    # Create thermal state of cavity
    cavity_thermal = thermal_dm(NUM_FOCK_STATES, n_thermal)
    
    # Create pure qubit state
    qubit_state_dm = fock_dm(2, qubit_state)
    
    # Tensor product
    return tensor(cavity_thermal, qubit_state_dm)

def analyze_state(state: Qobj):
    """
    Analyze properties of a quantum state.
    
    Parameters:
    -----------
    state : Qobj
        Quantum state to analyze
        
    Returns:
    --------
    dict
        Dictionary containing state properties:
        - 'is_pure': bool, whether state is pure
        - 'cavity_occupation': float, mean cavity photon number
        - 'qubit_excitation': float, qubit excitation probability
        - 'entanglement': float, entanglement measure (if pure state)
    """
    properties = {}
    
    # Check if state is pure (state vector) or mixed (density matrix)
    properties['is_pure'] = state.type == 'ket'
    
    # Convert to density matrix if necessary
    if properties['is_pure']:
        rho = state * state.dag()
    else:
        rho = state
    
    # Calculate cavity photon number
    # Cavity annihilation operator
    a_cavity = tensor(destroy(NUM_FOCK_STATES), qeye(2))
    cavity_number_op = a_cavity.dag() * a_cavity
    properties['cavity_occupation'] = expect(cavity_number_op, rho)
    
    # Calculate qubit excitation probability
    # Qubit excitation operator |e⟩⟨e|
    qubit_excitation_op = tensor(qeye(NUM_FOCK_STATES), fock_dm(2, 1))
    properties['qubit_excitation'] = expect(qubit_excitation_op, rho)
    
    # Calculate entanglement for pure states
    if properties['is_pure']:
        # Use partial trace to get reduced density matrices
        rho_cavity = ptrace(rho, 0)  # Trace out qubit
        rho_qubit = ptrace(rho, 1)   # Trace out cavity
        
        # Von Neumann entropy as entanglement measure
        properties['entanglement'] = entropy_vn(rho_cavity)
    else:
        properties['entanglement'] = None
    
    return properties

def state_fidelity(state1: Qobj, state2: Qobj):
    """
    Calculate the fidelity between two quantum states.
    
    Parameters:
    -----------
    state1, state2 : Qobj
        Quantum states to compare
        
    Returns:
    --------
    float
        Fidelity F(ρ₁, ρ₂) ∈ [0, 1]
    """
    return fidelity(state1, state2)

def print_state_info(state: Qobj, label: str = "State"):
    """
    Print detailed information about a quantum state.
    
    Parameters:
    -----------
    state : Qobj
        Quantum state to analyze
    label : str, default="State"
        Label for the state
    """
    props = analyze_state(state)
    
    print(f"\n{label} Analysis:")
    print("=" * (len(label) + 10))
    print(f"Type: {'Pure state' if props['is_pure'] else 'Mixed state'}")
    print(f"Cavity occupation: {props['cavity_occupation']:.4f} photons")
    print(f"Qubit excitation: {props['qubit_excitation']:.4f}")
    
    if props['entanglement'] is not None:
        print(f"Entanglement (S_vN): {props['entanglement']:.4f}")
    
    print(f"Hilbert space dimension: {state.shape[0]}")
    print(f"Norm: {state.norm():.6f}")
