"""
Quantum operators module.

This module provides functions to construct quantum operators and Hamiltonians
for the driven cavity–qubit system.

Backward compatible: defaults assume a single qubit. Multi-qubit helpers are
exposed via num_qubits parameter and indexed operator keys.
"""

from qutip import destroy, qeye, sigmaz, sigmax, sigmay, tensor, Qobj, expect
from typing import Optional, Dict, List, Union, Sequence, cast
import numpy as np
import warnings
from .config import NUM_FOCK_STATES, NUM_QUBITS

def _qubit_space_identities(nq: int) -> List[Qobj]:
    return [qeye(2) for _ in range(nq)]


def _embed_qubit_op(nq: int, j: int, op: Qobj) -> Qobj:
    """Embed a single-qubit operator into cavity⊗(qubits) Hilbert space.

    Order: [cavity, q0, q1, ...]
    """
    factors: List[Qobj] = [qeye(NUM_FOCK_STATES)] + _qubit_space_identities(nq)
    factors[1 + int(j)] = op
    return tensor(*factors)


def create_operators(num_qubits: int = NUM_QUBITS) -> Dict[str, Qobj]:
    """
    Create all quantum operators for the cavity–qubit(s) system.

    Backward compatibility: for num_qubits>=1, legacy single-qubit keys map to qubit 0.
    """
    ops: Dict[str, Qobj] = {}

    # Cavity operators
    a = tensor(destroy(NUM_FOCK_STATES), *(_qubit_space_identities(num_qubits)))
    ops['cavity_annihilation'] = a
    ops['cavity_creation'] = a.dag()
    ops['cavity_number'] = a.dag() * a
    ops['cavity_identity'] = tensor(qeye(NUM_FOCK_STATES), *(_qubit_space_identities(num_qubits)))

    # Per-qubit operators
    for j in range(num_qubits):
        sm_j = _embed_qubit_op(num_qubits, j, destroy(2))
        sp_j = sm_j.dag()
        sz_j = _embed_qubit_op(num_qubits, j, sigmaz())
        sx_j = _embed_qubit_op(num_qubits, j, sigmax())
        sy_j = _embed_qubit_op(num_qubits, j, sigmay())
        ops[f'qubit_lowering_{j}'] = sm_j
        ops[f'qubit_raising_{j}'] = sp_j
        ops[f'qubit_sigma_z_{j}'] = sz_j
        ops[f'qubit_sigma_x_{j}'] = sx_j
        ops[f'qubit_sigma_y_{j}'] = sy_j
        # Drive coupling alias (legacy used qubit lowering)
        ops[f'drive_coupling_{j}'] = sm_j

    # Identities
    ops['qubit_identity'] = tensor(qeye(NUM_FOCK_STATES), *(_qubit_space_identities(num_qubits)))
    ops['system_identity'] = ops['qubit_identity']

    # Legacy aliases → qubit 0
    if num_qubits >= 1:
        ops['qubit_lowering'] = ops['qubit_lowering_0']
        ops['qubit_raising'] = ops['qubit_raising_0']
        ops['qubit_sigma_z'] = ops['qubit_sigma_z_0']
        ops['qubit_sigma_x'] = ops['qubit_sigma_x_0']
        ops['qubit_sigma_y'] = ops['qubit_sigma_y_0']
        ops['drive_coupling'] = ops['drive_coupling_0']

    return ops

def construct_cavity_hamiltonian(omega_resonator: float, *, num_qubits: int = NUM_QUBITS) -> Qobj:
    """
    Construct the free cavity Hamiltonian: ℏωR â†â
    
    Parameters:
    -----------
    omega_resonator : float
        Resonator frequency (rad/s)
        
    Returns:
    --------
    Qobj
        Cavity Hamiltonian operator
    """
    operators = create_operators(num_qubits=num_qubits)
    return omega_resonator * operators['cavity_number']

def construct_qubit_hamiltonian(omega_transmon: Union[float, Sequence[float]], *, num_qubits: int = NUM_QUBITS) -> Qobj:
    """
    Construct the free qubit Hamiltonian: -ℏωT/2 σz
    
    Parameters:
    -----------
    omega_transmon : float
        Transmon frequency (rad/s)
        
    Returns:
    --------
    Qobj
        Qubit Hamiltonian operator
    """
    ops = create_operators(num_qubits=num_qubits)
    if isinstance(omega_transmon, (list, tuple, np.ndarray)):
        om = list(omega_transmon)
        if len(om) != num_qubits:
            raise ValueError(f'len(omega_transmon) must equal num_qubits={num_qubits}')
    else:
        om_scalar = float(cast(float, omega_transmon))
        om = [om_scalar] * num_qubits
    H = 0 * ops['cavity_number']
    for j, w in enumerate(om):
        H += -float(w) / 2.0 * ops[f'qubit_sigma_z_{j}']
    return H

def construct_interaction_hamiltonian(coupling_g: Union[float, Sequence[float]], *, num_qubits: int = NUM_QUBITS) -> Qobj:
    """
    Construct the lab-frame Jaynes–Cummings interaction Hamiltonian:
    -ℏ g (σ₊ â + σ₋ â†)

    Parameters
    ----------
    coupling_g : float
        Coupling strength (rad/s)

    Returns
    -------
    Qobj
        Interaction Hamiltonian operator
    """
    ops = create_operators(num_qubits=num_qubits)
    if isinstance(coupling_g, (list, tuple, np.ndarray)):
        gs = list(coupling_g)
        if len(gs) != num_qubits:
            raise ValueError(f'len(coupling_g) must equal num_qubits={num_qubits}')
    else:
        g_scalar = float(cast(float, coupling_g))
        gs = [g_scalar] * num_qubits
    H = 0 * ops['cavity_number']
    a = ops['cavity_annihilation']
    ad = ops['cavity_creation']
    for j, g in enumerate(gs):
        H += -float(g) * (ops[f'qubit_raising_{j}'] * a + ops[f'qubit_lowering_{j}'] * ad)
    return H

def construct_drive_hamiltonian_operator(*, num_qubits: int = NUM_QUBITS) -> Qobj:
    """
    Construct the drive coupling operator: (b̂† - b̂)
    
    This operator couples to the time-dependent drive field in the
    total Hamiltonian: H(t) = H₀ + V(t)(b̂† - b̂)
    
    Returns:
    --------
    Qobj
        Drive coupling operator
    """
    operators = create_operators(num_qubits=num_qubits)
    # Legacy: return qubit-0 drive coupling operator
    return (operators['drive_coupling'].dag() - operators['drive_coupling'])

def construct_system_hamiltonian(omega_r: float, omega_t: Union[float, Sequence[float]], 
                               coupling_g: Union[float, Sequence[float]], *, num_qubits: int = NUM_QUBITS) -> Qobj:
    """
    Construct the complete system Hamiltonian (without drive).
    
    H₀ = ℏωR â†â - (ℏωT/2) σz - ℏg(σ₊ - σ₋)(â† - â)
    
    Parameters:
    -----------
    omega_r : float
        Resonator frequency (rad/s)
    omega_t : float
        Transmon frequency (rad/s)
    coupling_g : float
        Coupling strength (rad/s)
        
    Returns:
    --------
    Qobj
        Complete system Hamiltonian
    """
    H_cavity = construct_cavity_hamiltonian(omega_r, num_qubits=num_qubits)
    H_qubit = construct_qubit_hamiltonian(omega_t, num_qubits=num_qubits)
    H_interaction = construct_interaction_hamiltonian(coupling_g, num_qubits=num_qubits)
    
    return H_cavity + H_qubit + H_interaction

def get_measurement_operators(*, num_qubits: int = NUM_QUBITS):
    """
    Get common measurement operators for the system.
    
    Returns:
    --------
    dict
        Dictionary of measurement operators:
        - 'cavity_occupation': â†â (cavity photon number)
        - 'qubit_excitation': σ₊σ₋ (qubit excitation probability)
        - 'qubit_z_expectation': σz (qubit z-component)
        - 'qubit_x_expectation': σx (qubit x-component)  
        - 'qubit_y_expectation': σy (qubit y-component)
    """
    operators = create_operators(num_qubits=num_qubits)
    
    measurements = {
        'cavity_occupation': operators['cavity_number'],
    'qubit_excitation': (operators['qubit_raising'] * operators['qubit_lowering']),
        'qubit_z_expectation': operators['qubit_sigma_z'],
        'qubit_x_expectation': operators['qubit_sigma_x'],
        'qubit_y_expectation': operators['qubit_sigma_y']
    }
    
    return measurements

def construct_dispersive_hamiltonian(omega_r: float, omega_t: float, 
                                   coupling_g: float, 
                                   detuning: Optional[float] = None) -> Qobj:
    """
    Construct Hamiltonian in the dispersive regime.
    
    In the dispersive limit (|Δ| >> g), the interaction becomes:
    H_disp ≈ ℏχ â†â σz
    
    where χ = g²/Δ is the dispersive shift and Δ = ωt - ωr.
    
    Parameters:
    -----------
    omega_r : float
        Resonator frequency (rad/s)
    omega_t : float  
        Transmon frequency (rad/s)
    coupling_g : float
        Coupling strength (rad/s)
    detuning : float, optional
        Detuning Δ = ωt - ωr. If None, calculated from frequencies.
        
    Returns:
    --------
    Qobj
        Dispersive Hamiltonian
    """
    if detuning is None:
        detuning = omega_t - omega_r
    
    if abs(detuning) <= 2 * abs(coupling_g):
        warnings.warn("Not in dispersive regime: |Δ| should be >> g")
    
    # Dispersive shift
    chi = coupling_g**2 / detuning
    
    operators = create_operators()
    
    H_cavity = omega_r * operators['cavity_number']
    H_qubit = -omega_t / 2 * operators['qubit_sigma_z']
    H_dispersive = chi * operators['cavity_number'] * operators['qubit_sigma_z']
    
    return H_cavity + H_qubit + H_dispersive

def get_collapse_operators(gamma_cavity: float = 0.0, 
                         gamma_qubit: float = 0.0,
                         gamma_dephasing: float = 0.0):
    """
    Get collapse operators for dissipative dynamics.
    
    Parameters:
    -----------
    gamma_cavity : float, default=0.0
        Cavity decay rate (1/ns)
    gamma_qubit : float, default=0.0
        Qubit decay rate (1/ns)
    gamma_dephasing : float, default=0.0
        Qubit dephasing rate (1/ns)
        
    Returns:
    --------
    list
        List of collapse operators with decay rates
    """
    collapse_ops = []
    operators = create_operators()
    
    if gamma_cavity > 0:
        # Cavity photon loss: √γ â
        collapse_ops.append(np.sqrt(gamma_cavity) * 
                           operators['cavity_annihilation'])
    
    if gamma_qubit > 0:
        # Qubit decay: √γ σ₋
        collapse_ops.append(np.sqrt(gamma_qubit) * 
                           operators['qubit_lowering'])
    
    if gamma_dephasing > 0:
        # Qubit dephasing: √γ_φ σz
        collapse_ops.append(np.sqrt(gamma_dephasing) * 
                           operators['qubit_sigma_z'])
    
    return collapse_ops

def operator_expectation_value(operator: Qobj, state: Qobj) -> complex:
    """
    Calculate expectation value of an operator in a given state.
    
    Parameters:
    -----------
    operator : Qobj
        Quantum operator
    state : Qobj
        Quantum state (ket or density matrix)
        
    Returns:
    --------
    complex
        Expectation value ⟨ψ|Ô|ψ⟩ or Tr(ρÔ)
    """
    val = expect(operator, state)
    # Ensure scalar complex return
    try:
        return complex(np.asarray(val).item())
    except Exception:
        arr = np.asarray(val)
        return complex(arr.ravel()[0])

def commutator(op1: Qobj, op2: Qobj) -> Qobj:
    """
    Calculate the commutator [Ô₁, Ô₂] = Ô₁Ô₂ - Ô₂Ô₁.
    
    Parameters:
    -----------
    op1, op2 : Qobj
        Quantum operators
        
    Returns:
    --------
    Qobj
        Commutator [Ô₁, Ô₂]
    """
    return op1 * op2 - op2 * op1

def anticommutator(op1: Qobj, op2: Qobj) -> Qobj:
    """
    Calculate the anticommutator {Ô₁, Ô₂} = Ô₁Ô₂ + Ô₂Ô₁.
    
    Parameters:
    -----------
    op1, op2 : Qobj
        Quantum operators
        
    Returns:
    --------
    Qobj
        Anticommutator {Ô₁, Ô₂}
    """
    return op1 * op2 + op2 * op1

def print_operator_info(operator: Qobj, name: str = "Operator"):
    """
    Print information about a quantum operator.
    
    Parameters:
    -----------
    operator : Qobj
        Quantum operator to analyze
    name : str, default="Operator"
        Name of the operator
    """
    print(f"\n{name} Information:")
    print("=" * (len(name) + 13))
    print(f"Type: {operator.type}")
    print(f"Shape: {operator.shape}")
    print(f"Dimensions: {operator.dims}")
    print(f"Is Hermitian: {operator.isherm}")
    print(f"Trace: {operator.tr():.6f}")
    
    if operator.isherm:
        try:
            eigenvals = np.asarray(operator.eigenenergies())
            print(f"Eigenvalue range: [{float(np.min(eigenvals)):.6f}, {float(np.max(eigenvals)):.6f}]")
        except Exception:
            pass
