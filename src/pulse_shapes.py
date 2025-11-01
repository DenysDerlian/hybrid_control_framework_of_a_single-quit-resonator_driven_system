"""
Pulse shapes and target functions module.

This module provides common pulse shapes and target functions used in
quantum control optimization.
"""

import numpy as np
from typing import Tuple, Optional
from .config import time_array
from scipy.signal import find_peaks

def generate_sinusoidal_target(frequency_factor: float = 1.0,
                             time_points: Optional[np.ndarray] = None,
                             amplitude: float = 1.0,
                             phase: float = 0.0,
                             absolute_value: bool = True) -> np.ndarray:
    """
    Generate sinusoidal target function.
    
    Parameters:
    -----------
    frequency_factor : float, default=1.0
        Frequency scaling factor
    time_points : np.ndarray, optional
        Time array. If None, uses global time_array
    amplitude : float, default=1.0
        Amplitude of sine function
    phase : float, default=0.0
        Phase offset in radians
    absolute_value : bool, default=True
        Whether to take absolute value |sin(t)|
        
    Returns:
    --------
    np.ndarray
        Sinusoidal target evolution
    """
    if time_points is None:
        time_points = time_array
    
    sinusoidal = amplitude * np.sin(frequency_factor * time_points + phase)
    
    if absolute_value:
        sinusoidal = np.abs(sinusoidal)
    
    return sinusoidal

def generate_gaussian_target(center_time: Optional[float] = None,
                           width: float = 1.0,
                           time_points: Optional[np.ndarray] = None,
                           amplitude: float = 1.0,
                           baseline: float = 0.0) -> np.ndarray:
    """
    Generate Gaussian pulse target function.
    
    Parameters:
    -----------
    center_time : float, optional
        Center time of Gaussian. If None, uses middle of time array
    width : float, default=1.0
        Standard deviation of Gaussian
    time_points : np.ndarray, optional
        Time array. If None, uses global time_array
    amplitude : float, default=1.0
        Peak amplitude of Gaussian
    baseline : float, default=0.0
        Baseline offset
        
    Returns:
    --------
    np.ndarray
        Gaussian target evolution
    """
    if time_points is None:
        time_points = time_array
    
    if center_time is None:
        center_time = time_points[len(time_points) // 2]
    
    gaussian = amplitude * np.exp(-((time_points - center_time) / width)**2) + baseline
    
    return gaussian

def generate_rectangular_target(pulse_start: Optional[float] = None,
                              pulse_end: Optional[float] = None,
                              time_points: Optional[np.ndarray] = None,
                              amplitude: float = 1.0,
                              baseline: float = 0.0,
                              pulse_fraction: Optional[float] = None) -> np.ndarray:
    """
    Generate rectangular (step) target function.
    
    Parameters:
    -----------
    pulse_start : float, optional
        Start time of pulse. If None, calculated from pulse_fraction
    pulse_end : float, optional
        End time of pulse. If None, calculated from pulse_fraction
    time_points : np.ndarray, optional
        Time array. If None, uses global time_array
    amplitude : float, default=1.0
        Amplitude during pulse
    baseline : float, default=0.0
        Baseline value outside pulse
    pulse_fraction : float, optional
        Fraction of total time for pulse (centered). Default 1/3
        
    Returns:
    --------
    np.ndarray
        Rectangular target evolution
    """
    if time_points is None:
        time_points = time_array
    
    # Initialize with baseline
    rectangular = np.full_like(time_points, baseline)
    
    # Determine pulse boundaries
    if pulse_start is None or pulse_end is None:
        if pulse_fraction is None:
            pulse_fraction = 1.0 / 3.0
        
        total_time = time_points[-1] - time_points[0]
        pulse_duration = pulse_fraction * total_time
        center_time = (time_points[-1] + time_points[0]) / 2
        
        pulse_start = center_time - pulse_duration / 2
        pulse_end = center_time + pulse_duration / 2
    
    # Set pulse region
    pulse_mask = (time_points >= pulse_start) & (time_points <= pulse_end)
    rectangular[pulse_mask] = amplitude
    
    return rectangular

def generate_exponential_target(decay_rate: float,
                              time_points: Optional[np.ndarray] = None,
                              amplitude: float = 1.0,
                              rising: bool = False) -> np.ndarray:
    """
    Generate exponential target function.
    
    Parameters:
    -----------
    decay_rate : float
        Exponential decay/rise rate
    time_points : np.ndarray, optional
        Time array. If None, uses global time_array
    amplitude : float, default=1.0
        Initial amplitude
    rising : bool, default=False
        If True, generates rising exponential (1 - exp(-t/τ))
        
    Returns:
    --------
    np.ndarray
        Exponential target evolution
    """
    if time_points is None:
        time_points = time_array
    
    if rising:
        exponential = amplitude * (1 - np.exp(-decay_rate * time_points))
    else:
        exponential = amplitude * np.exp(-decay_rate * time_points)
    
    return exponential

def generate_polynomial_target(coefficients: np.ndarray,
                             time_points: Optional[np.ndarray] = None,
                             normalize_time: bool = True) -> np.ndarray:
    """
    Generate polynomial target function.
    
    Parameters:
    -----------
    coefficients : np.ndarray
        Polynomial coefficients [a₀, a₁, a₂, ...] for a₀ + a₁t + a₂t² + ...
    time_points : np.ndarray, optional
        Time array. If None, uses global time_array
    normalize_time : bool, default=True
        Whether to normalize time to [0, 1] range
        
    Returns:
    --------
    np.ndarray
        Polynomial target evolution
    """
    if time_points is None:
        time_points = time_array
    
    if normalize_time:
        # Normalize time to [0, 1]
        t_norm = (time_points - time_points[0]) / (time_points[-1] - time_points[0])
    else:
        t_norm = time_points
    
    polynomial = np.polyval(coefficients[::-1], t_norm)  # polyval expects reverse order
    
    return polynomial

def generate_chirped_target(initial_frequency: float,
                          final_frequency: float,
                          time_points: Optional[np.ndarray] = None,
                          amplitude: float = 1.0,
                          phase: float = 0.0) -> np.ndarray:
    """
    Generate chirped (frequency-swept) target function.
    
    Parameters:
    -----------
    initial_frequency : float
        Initial frequency
    final_frequency : float
        Final frequency
    time_points : np.ndarray, optional
        Time array. If None, uses global time_array
    amplitude : float, default=1.0
        Amplitude
    phase : float, default=0.0
        Initial phase
        
    Returns:
    --------
    np.ndarray
        Chirped target evolution
    """
    if time_points is None:
        time_points = time_array
    
    # Linear frequency chirp
    total_time = time_points[-1] - time_points[0]
    chirp_rate = (final_frequency - initial_frequency) / total_time
    
    instantaneous_phase = (2 * np.pi * 
                          (initial_frequency * time_points + 
                           0.5 * chirp_rate * time_points**2) + phase)
    
    chirped = amplitude * np.sin(instantaneous_phase)
    
    return chirped

def generate_composite_target(target_functions: list,
                            weights: Optional[np.ndarray] = None,
                            time_points: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Generate composite target from multiple functions.
    
    Parameters:
    -----------
    target_functions : list
        List of target function arrays
    weights : np.ndarray, optional
        Weights for each function. If None, equal weights used
    time_points : np.ndarray, optional
        Time array for validation
        
    Returns:
    --------
    np.ndarray
        Composite target evolution
    """
    if time_points is None:
        time_points = time_array
    
    if weights is None:
        weights = np.ones(len(target_functions)) / len(target_functions)
    
    # Validate dimensions
    for func in target_functions:
        if len(func) != len(time_points):
            raise ValueError("All target functions must have same length as time array")
    
    # Weighted sum
    composite = np.zeros_like(time_points)
    for weight, func in zip(weights, target_functions):
        composite += weight * func
    
    return composite

def smooth_target_function(target: np.ndarray,
                         smoothing_width: float,
                         time_points: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply Gaussian smoothing to a target function.
    
    Parameters:
    -----------
    target : np.ndarray
        Target function to smooth
    smoothing_width : float
        Width of Gaussian smoothing kernel
    time_points : np.ndarray, optional
        Time array
        
    Returns:
    --------
    np.ndarray
        Smoothed target function
    """
    if time_points is None:
        time_points = time_array
    
    # Create Gaussian kernel
    dt = time_points[1] - time_points[0]
    kernel_size = int(6 * smoothing_width / dt)  # 6σ kernel
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd size
    
    kernel_times = np.linspace(-3 * smoothing_width, 3 * smoothing_width, kernel_size)
    kernel = np.exp(-(kernel_times / smoothing_width)**2)
    kernel = kernel / np.sum(kernel)  # Normalize
    
    # Apply convolution
    smoothed = np.convolve(target, kernel, mode='same')
    
    return smoothed

def analyze_target_properties(target: np.ndarray,
                            time_points: Optional[np.ndarray] = None) -> dict:
    """
    Analyze properties of a target function.
    
    Parameters:
    -----------
    target : np.ndarray
        Target function to analyze
    time_points : np.ndarray, optional
        Time array
        
    Returns:
    --------
    dict
        Dictionary of target properties
    """
    if time_points is None:
        time_points = time_array
    
    properties = {
        'mean': np.mean(target),
        'std': np.std(target),
        'min': np.min(target),
        'max': np.max(target),
        'range': np.max(target) - np.min(target),
        'energy': np.sum(target**2),
        'rms': np.sqrt(np.mean(target**2))
    }
    
    # Find peaks and valleys
    peaks, _ = find_peaks(target)
    valleys, _ = find_peaks(-target)
    
    properties['num_peaks'] = len(peaks)
    properties['num_valleys'] = len(valleys)
    
    if len(peaks) > 0:
        properties['peak_times'] = time_points[peaks]
        properties['peak_values'] = target[peaks]
    
    if len(valleys) > 0:
        properties['valley_times'] = time_points[valleys]
        properties['valley_values'] = target[valleys]
    
    # Calculate derivatives for smoothness analysis
    dt = time_points[1] - time_points[0]
    first_derivative = np.gradient(target, dt)
    second_derivative = np.gradient(first_derivative, dt)
    
    properties['max_slope'] = np.max(np.abs(first_derivative))
    properties['max_curvature'] = np.max(np.abs(second_derivative))
    properties['smoothness'] = 1.0 / (1.0 + properties['max_curvature'])
    
    return properties

def get_fourier_bandwidth_requirement(target: np.ndarray,
                                    time_points: Optional[np.ndarray] = None,
                                    threshold: float = 0.99) -> int:
    """
    Estimate required Fourier modes for target function representation.
    
    Parameters:
    -----------
    target : np.ndarray
        Target function
    time_points : np.ndarray, optional
        Time array
    threshold : float, default=0.99
        Fraction of energy to capture
        
    Returns:
    --------
    int
        Estimated number of Fourier modes needed
    """
    if time_points is None:
        time_points = time_array
    
    # Compute FFT
    fft_target = np.fft.fft(target)
    power_spectrum = np.abs(fft_target)**2
    
    # Sort by power and find cumulative energy
    sorted_indices = np.argsort(power_spectrum)[::-1]
    sorted_power = power_spectrum[sorted_indices]
    cumulative_energy = np.cumsum(sorted_power) / np.sum(sorted_power)
    
    # Find number of modes for threshold energy
    required_modes = int(np.argmax(cumulative_energy >= threshold) + 1)
    
    return required_modes

def print_target_summary(target: np.ndarray,
                       name: str = "Target Function",
                       time_points: Optional[np.ndarray] = None):
    """
    Print summary of target function properties.
    
    Parameters:
    -----------
    target : np.ndarray
        Target function
    name : str
        Name of target function
    time_points : np.ndarray, optional
        Time array
    """
    if time_points is None:
        time_points = time_array
    
    props = analyze_target_properties(target, time_points)
    
    print(f"\n{name} Summary:")
    print("=" * (len(name) + 9))
    print(f"Duration: {time_points[-1] - time_points[0]:.3f} ns")
    print(f"Time points: {len(time_points)}")
    print(f"Value range: [{props['min']:.4f}, {props['max']:.4f}]")
    print(f"Mean: {props['mean']:.4f} ± {props['std']:.4f}")
    print(f"RMS: {props['rms']:.4f}")
    print(f"Peaks: {props['num_peaks']}, Valleys: {props['num_valleys']}")
    print(f"Max slope: {props['max_slope']:.4f}")
    print(f"Smoothness: {props['smoothness']:.4f}")
    
    # Bandwidth estimate
    required_modes = get_fourier_bandwidth_requirement(target, time_points)
    print(f"Estimated Fourier modes needed: {required_modes}")
    print("=" * (len(name) + 9))
