"""
Visualization and plotting module for quantum control optimization.

This module provides comprehensive plotting capabilities for quantum control
results, including time evolution plots, optimization trajectories, and
publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from typing import List, Optional, Dict, Any, Tuple, Union
import os

from .config import time_array
try:
    from qutip import Bloch
except Exception:  # pragma: no cover
    Bloch = None  # type: ignore

# Default constants
DEFAULT_FONTSIZE = 14
DEFAULT_FIGSIZE = (10, 6)

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': DEFAULT_FONTSIZE,
    'font.family': 'serif',
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': False
})

class QuantumControlPlotter:
    """
    Comprehensive plotting class for quantum control optimization results.
    
    This class provides methods for visualizing time evolution, optimization
    results, control fields, and comparative analysis with publication-quality
    formatting.
    """
    
    def __init__(self, 
                 save_directory: str = "results",
                 figure_format: str = "pdf",
                 show_plots: bool = True,
                 use_latex: bool = True):
        """
        Initialize the quantum control plotter.
        
        Parameters:
        -----------
        save_directory : str, default="results"
            Directory to save figures
        figure_format : str, default="pdf"
            File format for saved figures
        show_plots : bool, default=True
            Whether to display plots interactively
        use_latex : bool, default=True
            Whether to use LaTeX rendering
        """
        self.save_directory = save_directory
        self.figure_format = figure_format
        self.show_plots = show_plots
        
        # Create save directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Configure LaTeX
        if use_latex:
            plt.rcParams['text.usetex'] = True
        else:
            plt.rcParams['text.usetex'] = False
            
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'tertiary': '#2ca02c',
            'quaternary': '#d62728',
            'target': '#9467bd',
            'optimized': '#17becf',
            'control': '#8c564b',
            'error': '#e377c2'
        }
        
    def plot_time_evolution(self,
                             time_points: np.ndarray,
                             evolution_data: np.ndarray,
                             target_data: Optional[np.ndarray] = None,
                             title: str = "Time Evolution",
                             xlabel: str = "Time (ns)",
                             ylabel: str = r"$\\langle\\sigma_z\\rangle$",
                             legend_labels: Optional[List[str]] = None,
                             filename: Optional[str] = None,
                             **kwargs) -> Optional[Figure]:
        """Plot time evolution of expectation values or states.

        Parameters:
            time_points: time array
            evolution_data: 1D (T,) or 2D (T, K) array
            target_data: optional target curve of shape (T,)
            title, xlabel, ylabel: labels
            legend_labels: optional labels for columns when evolution_data is 2D
            filename: optional save name
        """
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

        # Handle 1D or 2D data
        if evolution_data.ndim == 1:
            ax.plot(
                time_points,
                evolution_data,
                color=self.colors['primary'],
                linewidth=2.5,
                label=legend_labels[0] if legend_labels else "Evolution",
                **kwargs,
            )
        else:
            for i, data in enumerate(evolution_data.T):
                color = list(self.colors.values())[i % len(self.colors)]
                label = (
                    legend_labels[i]
                    if legend_labels and i < len(legend_labels)
                    else f"Series {i}"
                )
                ax.plot(time_points, data, color=color, linewidth=2.0, label=label, **kwargs)

        # Plot target if provided
        if target_data is not None:
            ax.plot(
                time_points,
                target_data,
                color=self.colors['target'],
                linestyle='--',
                linewidth=2.0,
                label="Target",
                alpha=0.8,
            )

        ax.set_xlabel(xlabel, fontsize=DEFAULT_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=DEFAULT_FONTSIZE)
        ax.set_title(title, fontsize=DEFAULT_FONTSIZE + 2, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=DEFAULT_FONTSIZE - 2)

        plt.tight_layout()

        if filename:
            self._save_figure(fig, filename)

        if self.show_plots:
            plt.show()
            return None
        return fig
    
    def plot_optimization_results(self,
                                time_points: np.ndarray,
                                initial_evolution: np.ndarray,
                                optimized_evolution: np.ndarray,
                                target_evolution: np.ndarray,
                                control_field: Optional[np.ndarray] = None,
                                title: str = "Optimization Results",
                                filename: Optional[str] = None) -> Optional[Figure]:
        """
        Create comprehensive optimization results plot.
        
        Parameters:
        -----------
        time_points : np.ndarray
            Time array
        initial_evolution : np.ndarray
            Evolution before optimization
        optimized_evolution : np.ndarray
            Evolution after optimization
        target_evolution : np.ndarray
            Target evolution
        control_field : np.ndarray, optional
            Optimized control field
        title : str
            Main title
        filename : str, optional
            Filename to save plot
            
        Returns:
        --------
        Optional[Figure]
            Figure object if show_plots is False; otherwise None.
        """
        if control_field is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DEFAULT_FIGSIZE[0], DEFAULT_FIGSIZE[1] * 1.5))
        else:
            fig, ax1 = plt.subplots(figsize=DEFAULT_FIGSIZE)
            ax2 = None
        
        # Main evolution plot
        ax1.plot(time_points, target_evolution,
                color=self.colors['target'],
                linestyle='-',
                linewidth=3.0,
                label=r'Target $f_{\mathrm{target}}(t)$',
                alpha=0.9)
        
        ax1.plot(time_points, initial_evolution,
                color=self.colors['secondary'],
                linestyle=':',
                linewidth=2.5,
                label=r'Initial $\langle\sigma_z^\dag\sigma_z\rangle_{\mathrm{initial}}(t)$',
                alpha=0.8)
        
        ax1.plot(time_points, optimized_evolution,
                color=self.colors['optimized'],
                linestyle='-',
                linewidth=2.5,
                label=r'Optimized $\langle\sigma_z^\dag\sigma_z\rangle_{\mathrm{opt}}(t)$')
        
        ax1.set_ylabel(r'$\langle\sigma_z^\dag\sigma_z\rangle$', fontsize=DEFAULT_FONTSIZE)
        ax1.set_title(title, fontsize=DEFAULT_FONTSIZE + 2, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=DEFAULT_FONTSIZE - 2)
        
        # Control field plot if provided
        if control_field is not None and ax2 is not None:
            ax2.plot(time_points, control_field/np.max(np.abs(control_field)),
                    color=self.colors['control'],
                    linewidth=2.0,
                    label=r'Control Field $\Omega(t)$')
            
            ax2.set_xlabel(r'Time (ns)', fontsize=DEFAULT_FONTSIZE)
            ax2.set_ylabel(r'$\Omega(t)$ (Normalized)', fontsize=DEFAULT_FONTSIZE)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=DEFAULT_FONTSIZE - 2)
        else:
            ax1.set_xlabel(r'Time (ns)', fontsize=DEFAULT_FONTSIZE)
        
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)

        if self.show_plots:
            plt.show()
            return None
        return fig
    
    def plot_cost_function_evolution(self,
                                   iteration_data: np.ndarray,
                                   cost_values: np.ndarray,
                                   title: str = "Cost Function Evolution",
                                   xlabel: str = "Iteration",
                                   ylabel: str = "Cost Function Value",
                                   log_scale: bool = True,
                                   filename: Optional[str] = None) -> Optional[Figure]:
        """
        Plot evolution of cost function during optimization.
        
        Parameters:
        -----------
        iteration_data : np.ndarray
            Iteration numbers or steps
        cost_values : np.ndarray
            Cost function values
        title : str
            Plot title
        xlabel : str
            X-axis label
        ylabel : str
            Y-axis label
        log_scale : bool, default=True
            Whether to use log scale for y-axis
        filename : str, optional
            Filename to save plot
            
        Returns:
        --------
        Optional[plt.Figure]
            Figure object if show_plots is False; otherwise None.
        """
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        
        ax.plot(iteration_data, cost_values,
               color=self.colors['primary'],
               linewidth=2.0,
               marker='o',
               markersize=4,
               alpha=0.8)
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.set_xlabel(xlabel, fontsize=DEFAULT_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=DEFAULT_FONTSIZE)
        ax.set_title(title, fontsize=DEFAULT_FONTSIZE + 2, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add final value annotation
        final_cost = cost_values[-1]
        ax.annotate(f'Final: {final_cost:.2e}',
                   xy=(iteration_data[-1], final_cost),
                   xytext=(10, 10),
                   textcoords='offset points',
                   fontsize=DEFAULT_FONTSIZE - 2,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)

        if self.show_plots:
            plt.show()
            return None
        return fig
    
    def plot_control_field_analysis(self,
                                   time_points: np.ndarray,
                                   control_field: Union[np.ndarray, Dict[str, np.ndarray]],
                                   fourier_coefficients: Optional[np.ndarray] = None,
                                   title: str = "Control Field Analysis",
                                   filename: Optional[str] = None) -> Optional[Figure]:
        """Plot control field(s) in time and frequency domains. Supports I/Q dict.

        control_field can be either a 1D array (single control) or a dict with
        keys 'composite', 'I', 'Q'. If 'composite' is missing, it will be
        computed as I*cos -+ Q*sin by the caller; here we just sum I and Q if both
        provided as a proxy for visualization.
        """
        is_iq = isinstance(control_field, dict)

        # Figure layout
        if fourier_coefficients is not None:
            if is_iq:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                    2, 2, figsize=(DEFAULT_FIGSIZE[0] * 1.7, DEFAULT_FIGSIZE[1] * 1.7)
                )
            else:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                    2, 2, figsize=(DEFAULT_FIGSIZE[0] * 1.5, DEFAULT_FIGSIZE[1] * 1.5)
                )
        else:
            if is_iq:
                fig, axes = plt.subplots(
                    1, 3, figsize=(DEFAULT_FIGSIZE[0] * 1.8, DEFAULT_FIGSIZE[1])
                )
                ax1, ax2, ax3 = axes
                ax4 = None
            else:
                fig, (ax1, ax2) = plt.subplots(
                    1, 2, figsize=(DEFAULT_FIGSIZE[0] * 1.5, DEFAULT_FIGSIZE[1])
                )
                ax3 = ax4 = None

        # Time-domain plots
        if is_iq:
            comp = control_field.get('composite')  # type: ignore[assignment]
            I = control_field.get('I')             # type: ignore[assignment]
            Q = control_field.get('Q')             # type: ignore[assignment]
            composite = (
                comp
                if comp is not None
                else ((I + Q) if (I is not None and Q is not None) else np.zeros_like(time_points))
            )
            ax1.plot(time_points, composite, color=self.colors['control'], linewidth=2.0, label='Composite')
            ax1.set_title('Composite Control', fontsize=DEFAULT_FONTSIZE)
            ax1.set_xlabel(r'Time (ns)', fontsize=DEFAULT_FONTSIZE)
            ax1.set_ylabel(r'$\Omega_{\mathrm{I}}\cos(\omega_d t)+\Omega_{\mathrm{Q}}\sin(\omega_d t)$', fontsize=DEFAULT_FONTSIZE)
            ax1.grid(True, alpha=0.3)

            ax2.plot(time_points, I if I is not None else np.zeros_like(time_points), color=self.colors['primary'], linewidth=2.0, label='I')
            ax2.set_title('I Component', fontsize=DEFAULT_FONTSIZE)
            ax2.set_xlabel(r'Time (ns)', fontsize=DEFAULT_FONTSIZE)
            ax2.set_ylabel(r'$\Omega_I(t)\cos(\omega_d t)$', fontsize=DEFAULT_FONTSIZE)
            ax2.grid(True, alpha=0.3)

            if ax3 is not None:
                ax3.plot(time_points, Q if Q is not None else np.zeros_like(time_points), color=self.colors['secondary'], linewidth=2.0, label='Q')
                ax3.set_title('Q Component', fontsize=DEFAULT_FONTSIZE)
                ax3.set_xlabel(r'Time (ns)', fontsize=DEFAULT_FONTSIZE)
                ax3.set_ylabel(r'$\Omega_Q(t)\sin(\omega_d t)$', fontsize=DEFAULT_FONTSIZE)
                ax3.grid(True, alpha=0.3)
        else:
            ax1.plot(time_points, control_field/np.max(np.abs(control_field)), color=self.colors['control'], linewidth=2.0)
            ax1.set_xlabel(r'Time (ns)', fontsize=DEFAULT_FONTSIZE)
            ax1.set_ylabel(r'$\Omega(t)$ (Normalized)', fontsize=DEFAULT_FONTSIZE)
            ax1.set_title('Control Field', fontsize=DEFAULT_FONTSIZE)
            ax1.grid(True, alpha=0.3)

        # Frequency spectrum (from composite or I if not provided)
        if ax2 is not None:
            if is_iq and isinstance(control_field, dict):
                base_cf = control_field.get('composite')
                if base_cf is None:
                    base_cf = control_field.get('I')
                if base_cf is None:
                    base_cf = np.zeros_like(time_points)
            else:
                base_cf = control_field if isinstance(control_field, np.ndarray) else np.zeros_like(time_points)

            base_cf = np.asarray(base_cf)
            if base_cf.size > 1:
                freqs = np.fft.fftfreq(len(base_cf), time_points[1] - time_points[0])
                fft_control = np.fft.fft(base_cf)
                power_spectrum = np.abs(fft_control) ** 2

                # Choose axis for spectrum: ax2 in non-IQ; in IQ-with-coeffs keep on ax2 for clarity
                spec_ax = ax2 if ax2 is not None else ax4
                if spec_ax is not None:
                    pos_mask1 = freqs >= 0
                    pos_mask2 = freqs <= 5.0
                    pos_mask = pos_mask1 & pos_mask2
                    spec_ax.semilogy(freqs[pos_mask], power_spectrum[pos_mask]/np.max(np.abs(power_spectrum)),
                                     color=self.colors['primary'], linewidth=2.0)
                    spec_ax.set_xlabel(r'Frequency (GHz)', fontsize=DEFAULT_FONTSIZE)
                    spec_ax.set_ylabel('Power Spectrum (Normalized)', fontsize=DEFAULT_FONTSIZE)
                    spec_ax.set_title('Frequency Spectrum', fontsize=DEFAULT_FONTSIZE)
                    spec_ax.grid(True, alpha=0.3)

        # Fourier coefficients (only when provided and not IQ to keep layout tidy)
        if (fourier_coefficients is not None) and (not is_iq) and (ax3 is not None) and (ax4 is not None):
            modes = np.arange(len(fourier_coefficients))
            ax3.bar(modes, np.abs(fourier_coefficients), color=self.colors['secondary'], alpha=0.7)
            ax3.set_xlabel('Fourier Mode', fontsize=DEFAULT_FONTSIZE)
            ax3.set_ylabel('|Coefficient|', fontsize=DEFAULT_FONTSIZE)
            ax3.set_title('Fourier Coefficients', fontsize=DEFAULT_FONTSIZE)
            ax3.grid(True, alpha=0.3)

            ax4.bar(modes, np.angle(fourier_coefficients), color=self.colors['tertiary'], alpha=0.7)
            ax4.set_xlabel('Fourier Mode', fontsize=DEFAULT_FONTSIZE)
            ax4.set_ylabel('Phase (rad)', fontsize=DEFAULT_FONTSIZE)
            ax4.set_title('Fourier Phases', fontsize=DEFAULT_FONTSIZE)
            ax4.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=DEFAULT_FONTSIZE + 2, fontweight='bold')
        plt.tight_layout()

        if filename:
            self._save_figure(fig, filename)

        if self.show_plots:
            plt.show()
            return None
        return fig
    
    def plot_parameter_space_exploration(self,
                                       parameter_values: np.ndarray,
                                       cost_values: np.ndarray,
                                       parameter_name: str = "Parameter",
                                       title: str = "Parameter Space Exploration",
                                       filename: Optional[str] = None) -> Optional[Figure]:
        """
        Plot cost function vs parameter values.
        
        Parameters:
        -----------
        parameter_values : np.ndarray
            Parameter values explored
        cost_values : np.ndarray
            Corresponding cost function values
        parameter_name : str
            Name of parameter
        title : str
            Plot title
        filename : str, optional
            Filename to save plot
            
        Returns:
        --------
        Optional[plt.Figure]
            Figure object if show_plots is False; otherwise None.
        """
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        
        # Scatter plot with connecting line
        ax.plot(parameter_values, cost_values,
               color=self.colors['primary'],
               linewidth=1.5,
               alpha=0.6)
        ax.scatter(parameter_values, cost_values,
                  color=self.colors['primary'],
                  s=50,
                  alpha=0.8,
                  edgecolors='white',
                  linewidth=1)
        
        # Mark minimum
        min_idx = np.argmin(cost_values)
        ax.scatter(parameter_values[min_idx], cost_values[min_idx],
                  color=self.colors['quaternary'],
                  s=100,
                  marker='*',
                  label=f'Minimum: {parameter_values[min_idx]:.3f}',
                  edgecolors='white',
                  linewidth=1)
        
        ax.set_xlabel(parameter_name, fontsize=DEFAULT_FONTSIZE)
        ax.set_ylabel('Cost Function Value', fontsize=DEFAULT_FONTSIZE)
        ax.set_title(title, fontsize=DEFAULT_FONTSIZE + 2, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=DEFAULT_FONTSIZE - 2)
        
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)

        if self.show_plots:
            plt.show()
            return None
        return fig
    
    def plot_multi_target_comparison(self,
                                   time_points: np.ndarray,
                                   target_data: Dict[str, np.ndarray],
                                   optimized_data: Dict[str, np.ndarray],
                                   cost_values: Dict[str, float],
                                   title: str = "Multi-Target Optimization Comparison",
                                   filename: Optional[str] = None) -> Optional[Figure]:
        """
        Compare optimization results for multiple targets.
        
        Parameters:
        -----------
        time_points : np.ndarray
            Time array
        target_data : Dict[str, np.ndarray]
            Target functions by name
        optimized_data : Dict[str, np.ndarray]
            Optimized results by name
        cost_values : Dict[str, float]
            Final cost values by name
        title : str
            Main title
        filename : str, optional
            Filename to save plot
            
        Returns:
        --------
        Optional[plt.Figure]
            Figure object if show_plots is False; otherwise None.
        """
        n_targets = len(target_data)
        fig, axes = plt.subplots(1, n_targets, figsize=(DEFAULT_FIGSIZE[0] * n_targets, DEFAULT_FIGSIZE[1]))
        
        if n_targets == 1:
            axes = [axes]
        
        colors = list(self.colors.values())
        
        for i, (name, target) in enumerate(target_data.items()):
            ax = axes[i]
            color = colors[i % len(colors)]
            
            # Plot target and optimized
            ax.plot(time_points, target,
                   color='black',
                   linestyle='--',
                   linewidth=2.5,
                   label='Target',
                   alpha=0.8)
            
            ax.plot(time_points, optimized_data[name],
                   color=color,
                   linewidth=2.5,
                   label='Optimized')
            
            ax.set_title(f'{name}\nCost: {cost_values[name]:.2e}',
                        fontsize=DEFAULT_FONTSIZE)
            ax.set_xlabel(r'Time (ns)', fontsize=DEFAULT_FONTSIZE)
            if i == 0:
                ax.set_ylabel(r'$\langle\sigma_z\rangle$', fontsize=DEFAULT_FONTSIZE)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=DEFAULT_FONTSIZE - 3)
        
        plt.suptitle(title, fontsize=DEFAULT_FONTSIZE + 2, fontweight='bold')
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)

        if self.show_plots:
            plt.show()
            return None
        return fig
    
    def plot_convergence_analysis(self,
                                convergence_data: Dict[str, List[float]],
                                title: str = "Convergence Analysis",
                                filename: Optional[str] = None) -> Optional[Figure]:
        """
        Plot convergence analysis for multiple optimization runs.
        
        Parameters:
        -----------
        convergence_data : Dict[str, List[float]]
            Convergence data by run name
        title : str
            Plot title
        filename : str, optional
            Filename to save plot
            
        Returns:
        --------
        Optional[plt.Figure]
            Figure object if show_plots is False; otherwise None.
        """
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        
        colors = list(self.colors.values())
        
        for i, (name, data) in enumerate(convergence_data.items()):
            color = colors[i % len(colors)]
            iterations = np.arange(len(data))
            
            ax.semilogy(iterations, data,
                       color=color,
                       linewidth=2.0,
                       label=name,
                       alpha=0.8)
        
        ax.set_xlabel('Iteration', fontsize=DEFAULT_FONTSIZE)
        ax.set_ylabel('Cost Function Value', fontsize=DEFAULT_FONTSIZE)
        ax.set_title(title, fontsize=DEFAULT_FONTSIZE + 2, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=DEFAULT_FONTSIZE - 2)
        
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)

        if self.show_plots:
            plt.show()
            return None
        return fig
    
    def create_summary_figure(self,
                            time_points: np.ndarray,
                            results_data: Dict[str, Dict[str, Any]],
                            title: str = "Quantum Control Optimization Summary",
                            filename: Optional[str] = None) -> Optional[Figure]:
        """
        Create comprehensive summary figure.
        
        Parameters:
        -----------
        time_points : np.ndarray
            Time array
        results_data : Dict[str, Dict[str, Any]]
            Complete results data structure
        title : str
            Main title
        filename : str, optional
            Filename to save plot
            
        Returns:
        --------
        Optional[plt.Figure]
            Figure object if show_plots is False; otherwise None.
        """
        fig = plt.figure(figsize=(DEFAULT_FIGSIZE[0] * 2, DEFAULT_FIGSIZE[1] * 2))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Evolution comparison (top row, spans both columns)
        ax1 = fig.add_subplot(gs[0, :])
        
        colors = list(self.colors.values())
        for i, (name, data) in enumerate(results_data.items()):
            color = colors[i % len(colors)]
            if 'target' in data:
                ax1.plot(time_points, data['target'],
                        color='black',
                        linestyle='--',
                        linewidth=2.0,
                        alpha=0.6,
                        label=f'Target' if i == 0 else "")
            
            if 'optimized' in data:
                ax1.plot(time_points, data['optimized'],
                        color=color,
                        linewidth=2.5,
                        label=f'{name} Optimized')

        ax1.set_ylabel(r'$\langle\hat{\sigma}_z^\dagger \hat{\sigma}_z\rangle$', fontsize=DEFAULT_FONTSIZE)
        ax1.set_title('Optimization Results Comparison', fontsize=DEFAULT_FONTSIZE + 1)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=DEFAULT_FONTSIZE - 3)
        
        # Cost comparison (bottom left)
        ax2 = fig.add_subplot(gs[1, 0])
        names = list(results_data.keys())
        costs = [results_data[name]['final_cost'] for name in names]
        
        bars = ax2.bar(names, costs, color=colors[:len(names)], alpha=0.7)
        ax2.set_ylabel('Final Cost', fontsize=DEFAULT_FONTSIZE)
        ax2.set_title('Final Cost Comparison', fontsize=DEFAULT_FONTSIZE)
        ax2.set_yscale('log')
        
        # Add value labels on bars
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{cost:.2e}',
                    ha='center', va='bottom',
                    fontsize=DEFAULT_FONTSIZE - 3)
            ax2.set_ylim(0, max(costs)*1.05)
        
        # Control fields (bottom right)
        ax3 = fig.add_subplot(gs[1, 1])
        for i, (name, data) in enumerate(results_data.items()):
            if 'control_field' in data:
                color = colors[i % len(colors)]
                ax3.plot(time_points, data['control_field']/np.max(np.abs(data['control_field'])),
                        color=color,
                        linewidth=2.0,
                        label=name,
                        alpha=0.8)
        
        ax3.set_xlabel(r'Time (ns)', fontsize=DEFAULT_FONTSIZE)
        ax3.set_ylabel(r'$\Omega(t)$ (Normalized)', fontsize=DEFAULT_FONTSIZE)
        ax3.set_title('Optimized Control Fields', fontsize=DEFAULT_FONTSIZE)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=DEFAULT_FONTSIZE - 3)
        
        # Statistics table (bottom)
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Create statistics table
        stats_data = []
        for name, data in results_data.items():
            fidelity = 1 - data['final_cost']  # Assuming cost = 1 - fidelity
            stats_data.append([
                name,
                f"{data['final_cost']:.2e}",
                f"{fidelity:.6f}",
                f"{data.get('nfev', 'N/A')}",
                f"{data.get('optimization_time', 'N/A'):.2f}s" if isinstance(data.get('optimization_time'), (int, float)) else "N/A"
            ])

        headers = ['Target', 'Final Cost', 'Fidelity', 'NFEV', 'Time (s)']
        table = ax4.table(cellText=stats_data,
                         colLabels=headers,
                         cellLoc='center',
                         loc='center',
                         cellColours=[['lightgray']*5]*len(stats_data))
        
        table.auto_set_font_size(False)
        table.set_fontsize(DEFAULT_FONTSIZE - 3)
        table.scale(1, 1.5)
        
        plt.suptitle(title, fontsize=DEFAULT_FONTSIZE + 4, fontweight='bold')
        
        if filename:
            self._save_figure(fig, filename)

        if self.show_plots:
            plt.show()
            return None
        return fig

    # ------------------------------------------------------------
    # Gate / Bloch-specific plotting helpers
    # ------------------------------------------------------------
    def plot_bloch_components(self,
                               time_points: np.ndarray,
                               ex: np.ndarray,
                               ey: np.ndarray,
                               ez: np.ndarray,
                               ideal_vector: Optional[List[float]] = None,
                               gate_name: str = "Gate",
                               filename: Optional[str] = None) -> Optional[Figure]:
        """Plot Bloch expectation components σx, σy, σz vs time in separate subplots.

        Parameters:
            time_points: time array
            ex, ey, ez: expectation value arrays
            ideal_vector: optional [x,y,z] to overlay as horizontal dashed lines
            gate_name: label for titles
            filename: optional save name
        """
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        comps = [(ex, r'$\langle \sigma_x \rangle$', 'crimson'),
                 (ey, r'$\langle \sigma_y \rangle$', 'seagreen'),
                 (ez, r'$\langle \sigma_z \rangle$', 'navy')]
        for idx, (data, label, color) in enumerate(comps):
            axes[idx].plot(time_points, data, color=color, linewidth=2.0, label=label)
            if ideal_vector is not None:
                axes[idx].axhline(ideal_vector[idx], ls='--', color='gray', alpha=0.7,
                                   label='Ideal')
            axes[idx].set_ylabel(label)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim(-1.05, 1.05)
            #axes[idx].legend(fontsize=16, loc='upper right')
        axes[-1].set_xlabel('Time (ns)')
        # fig.suptitle(f'{gate_name} Bloch Components', fontsize=8, fontweight='bold')
        plt.tight_layout(rect=(0,0,1,0.97))
        if filename:
            self._save_figure(fig, filename)
        if self.show_plots:
            plt.show()
            return None
        return fig

    def plot_bloch_trajectory(self,
                               ex: np.ndarray,
                               ey: np.ndarray,
                               ez: np.ndarray,
                               ideal_vector: Optional[List[float]] = None,
                               gate_name: str = "Gate",
                               filename: Optional[str] = None) -> Optional[Figure]:
        """Render a Bloch sphere trajectory using QuTiP's Bloch class if available."""
        if Bloch is None:
            print("Bloch plotting unavailable (qutip Bloch import failed).")
            return None
        try:
            fig = plt.figure(figsize=(5,5))
            b = Bloch(fig=fig)
            b.add_points([ex, ey, ez])
            if ideal_vector is not None:
                b.add_vectors(ideal_vector)
            # Set style (wrapped in try to avoid API differences)
            try:
                b.point_color = ['#1f77b4']  # type: ignore
                b.vector_color = ['#2ca02c']  # type: ignore
                b.point_size = [15]  # type: ignore
            except Exception:
                pass
            b.render()
            # plt.title(f'{gate_name} Bloch Trajectory')
            if filename:
                self._save_figure(fig, filename)
            if self.show_plots:
                plt.show()
                return None
            return fig
        except Exception as e:  # pragma: no cover
            print(f"Bloch trajectory plotting failed: {e}")
            return None
    
    def _save_figure(self, fig: Figure, filename: str):
        """
        Save figure to file.
        
        Parameters:
        -----------
        fig : plt.Figure
            Figure to save
        filename : str
            Filename (without extension)
        """
        if not filename.endswith(f'.{self.figure_format}'):
            filename = f"{filename}.{self.figure_format}"
        
        filepath = os.path.join(self.save_directory, filename)
        fig.savefig(filepath, 
                   format=self.figure_format,
                   dpi=600,
                   bbox_inches='tight',
                   transparent=False)
        print(f"Figure saved: {filepath}")
    
    def set_style(self, style_name: str = "publication"):
        """
        Set plotting style.
        
        Parameters:
        -----------
        style_name : str
            Style name ('publication', 'presentation', 'simple')
        """
        if style_name == "publication":
            plt.rcParams.update({
                'font.size': 12,
                'font.family': 'serif',
                'text.usetex': True,
                'figure.dpi': 300,
                'lines.linewidth': 2.0
            })
        elif style_name == "presentation":
            plt.rcParams.update({
                'font.size': 16,
                'font.family': 'sans-serif',
                'text.usetex': False,
                'figure.dpi': 150,
                'lines.linewidth': 3.0
            })
        elif style_name == "simple":
            plt.rcParams.update({
                'font.size': 10,
                'font.family': 'sans-serif',
                'text.usetex': False,
                'figure.dpi': 100,
                'lines.linewidth': 1.5
            })
    
    def close_all(self):
        """Close all open figures."""
        plt.close('all')

# Convenience functions for quick plotting

def quick_evolution_plot(time_points: np.ndarray,
                        evolution: np.ndarray,
                        target: Optional[np.ndarray] = None,
                        title: str = "Evolution",
                        save_name: Optional[str] = None):
    """Quick plot for evolution data."""
    plotter = QuantumControlPlotter()
    return plotter.plot_time_evolution(
        time_points, evolution, target, title=title, filename=save_name
    )

def quick_optimization_plot(time_points: np.ndarray,
                          target: np.ndarray,
                          initial: np.ndarray,
                          optimized: np.ndarray,
                          control: Optional[np.ndarray] = None,
                          title: str = "Optimization",
                          save_name: Optional[str] = None):
    """Quick plot for optimization results."""
    plotter = QuantumControlPlotter()
    return plotter.plot_optimization_results(
        time_points, initial, optimized, target, 
        control_field=control, title=title, filename=save_name
    )

def quick_comparison_plot(time_points: np.ndarray,
                        results_dict: Dict[str, Dict[str, np.ndarray]],
                        title: str = "Comparison",
                        save_name: Optional[str] = None):
    """Quick plot for comparing multiple results."""
    plotter = QuantumControlPlotter()
    
    targets = {name: data['target'] for name, data in results_dict.items()}
    optimized = {name: data['optimized'] for name, data in results_dict.items()}
    costs = {name: float(data.get('cost', 0.0)) for name, data in results_dict.items()}
    
    return plotter.plot_multi_target_comparison(
        time_points, targets, optimized, costs, title=title, filename=save_name
    )
