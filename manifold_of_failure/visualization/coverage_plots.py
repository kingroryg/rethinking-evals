"""
Coverage and performance plots for MAP-Elites experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import json


def plot_coverage_over_time(stats_history: List[Dict],
                           figsize: Tuple[int, int] = (10, 6),
                           save_path: Optional[str] = None,
                           dpi: int = 300):
    """
    Plot coverage percentage over iterations.
    
    Args:
        stats_history: List of statistics dictionaries
        figsize: Figure size
        save_path: Path to save figure
        dpi: DPI for saved figure
    """
    iterations = [s['iteration'] for s in stats_history]
    coverage = [s['coverage'] for s in stats_history]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(iterations, coverage, linewidth=2, color='#2E86AB')
    ax.fill_between(iterations, coverage, alpha=0.3, color='#2E86AB')
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Behavioral Space Coverage Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Coverage plot saved to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_diversity_over_time(stats_history: List[Dict],
                            figsize: Tuple[int, int] = (10, 6),
                            save_path: Optional[str] = None,
                            dpi: int = 300):
    """
    Plot diversity (number of failure modes) over iterations.
    """
    iterations = [s['iteration'] for s in stats_history]
    diversity = [s['diversity'] for s in stats_history]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(iterations, diversity, linewidth=2, color='#A23B72')
    ax.fill_between(iterations, diversity, alpha=0.3, color='#A23B72')
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Failure Modes (AD > 0.5)', fontsize=12, fontweight='bold')
    ax.set_title('Diversity of Discovered Failures Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    
    return fig


def plot_quality_metrics_over_time(stats_history: List[Dict],
                                  figsize: Tuple[int, int] = (12, 6),
                                  save_path: Optional[str] = None,
                                  dpi: int = 300):
    """
    Plot multiple quality metrics over time.
    """
    iterations = [s['iteration'] for s in stats_history]
    peak_quality = [s['peak_quality'] for s in stats_history]
    mean_quality = [s['mean_quality'] for s in stats_history]
    qd_score = [s['qd_score'] for s in stats_history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Peak and Mean AD
    ax1.plot(iterations, peak_quality, linewidth=2, label='Peak AD', color='#E63946')
    ax1.plot(iterations, mean_quality, linewidth=2, label='Mean AD', color='#457B9D')
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Alignment Deviation', fontsize=12, fontweight='bold')
    ax1.set_title('Quality Metrics Over Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Right: QD-Score
    ax2.plot(iterations, qd_score, linewidth=2, color='#F77F00')
    ax2.fill_between(iterations, qd_score, alpha=0.3, color='#F77F00')
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('QD-Score', fontsize=12, fontweight='bold')
    ax2.set_title('QD-Score Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    
    return fig


def plot_baseline_comparison(results: Dict[str, Dict],
                            metrics: List[str] = ['coverage', 'diversity', 'peak_quality', 'asr'],
                            figsize: Tuple[int, int] = (14, 8),
                            save_path: Optional[str] = None,
                            dpi: int = 300):
    """
    Compare MAP-Elites against baselines.
    
    Args:
        results: Dictionary mapping method name to results dict
        metrics: List of metrics to compare
        figsize: Figure size
        save_path: Path to save figure
        dpi: DPI for saved figure
    """
    methods = list(results.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    metric_labels = {
        'coverage': 'Coverage (%)',
        'diversity': 'Diversity (# modes)',
        'peak_quality': 'Peak AD',
        'asr': 'ASR (%)',
        'semantic_validity': 'Semantic Validity (%)'
    }
    
    colors = ['#2E86AB', '#A23B72', '#F77F00', '#06A77D', '#E63946']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        values = [results[method].get(metric, 0) for method in methods]
        errors = [results[method].get(f'{metric}_std', 0) for method in methods]
        
        bars = ax.bar(range(len(methods)), values, yerr=errors, 
                     capsize=5, color=colors[:len(methods)], alpha=0.8)
        
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=11, fontweight='bold')
        ax.set_title(metric_labels.get(metric, metric), fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # Highlight best
        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2)
    
    fig.suptitle('Method Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    
    return fig


def plot_efficiency_comparison(results: Dict[str, Dict],
                              figsize: Tuple[int, int] = (10, 6),
                              save_path: Optional[str] = None,
                              dpi: int = 300):
    """
    Plot coverage vs number of evaluations (efficiency).
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#2E86AB', '#A23B72', '#F77F00', '#06A77D', '#E63946']
    
    for idx, (method, data) in enumerate(results.items()):
        if 'evaluations' in data and 'coverage' in data:
            ax.scatter(data['evaluations'], data['coverage'], 
                      s=200, label=method, color=colors[idx % len(colors)],
                      alpha=0.7, edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel('Number of Evaluations', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Exploration Efficiency', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    
    return fig


def plot_ablation_study(ablation_results: Dict[str, Dict],
                       metrics: List[str] = ['coverage', 'diversity', 'asr'],
                       figsize: Tuple[int, int] = (12, 5),
                       save_path: Optional[str] = None,
                       dpi: int = 300):
    """
    Visualize ablation study results.
    """
    variants = list(ablation_results.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    metric_labels = {
        'coverage': 'Coverage (%)',
        'diversity': 'Diversity',
        'asr': 'ASR (%)'
    }
    
    colors = ['#06A77D', '#F77F00', '#E63946']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        values = [ablation_results[variant].get(metric, 0) for variant in variants]
        errors = [ablation_results[variant].get(f'{metric}_std', 0) for variant in variants]
        
        bars = ax.bar(range(len(variants)), values, yerr=errors,
                     capsize=5, color=colors[:len(variants)], alpha=0.8)
        
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=45, ha='right')
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=11, fontweight='bold')
        ax.set_title(metric_labels.get(metric, metric), fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # Highlight full method
        if 'full' in variants:
            full_idx = variants.index('full')
            bars[full_idx].set_edgecolor('black')
            bars[full_idx].set_linewidth(2)
    
    fig.suptitle('Ablation Study', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    
    return fig


def create_summary_dashboard(stats_history: List[Dict],
                            archive,
                            figsize: Tuple[int, int] = (16, 10),
                            save_path: Optional[str] = None,
                            dpi: int = 300):
    """
    Create a comprehensive dashboard with multiple plots.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Extract data
    iterations = [s['iteration'] for s in stats_history]
    coverage = [s['coverage'] for s in stats_history]
    diversity = [s['diversity'] for s in stats_history]
    peak_quality = [s['peak_quality'] for s in stats_history]
    mean_quality = [s['mean_quality'] for s in stats_history]
    
    # 1. Coverage over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iterations, coverage, linewidth=2, color='#2E86AB')
    ax1.fill_between(iterations, coverage, alpha=0.3, color='#2E86AB')
    ax1.set_xlabel('Iteration', fontsize=10)
    ax1.set_ylabel('Coverage (%)', fontsize=10)
    ax1.set_title('Coverage Over Time', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Diversity over time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(iterations, diversity, linewidth=2, color='#A23B72')
    ax2.fill_between(iterations, diversity, alpha=0.3, color='#A23B72')
    ax2.set_xlabel('Iteration', fontsize=10)
    ax2.set_ylabel('Diversity', fontsize=10)
    ax2.set_title('Diversity Over Time', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Quality metrics
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(iterations, peak_quality, linewidth=2, label='Peak AD', color='#E63946')
    ax3.plot(iterations, mean_quality, linewidth=2, label='Mean AD', color='#457B9D')
    ax3.set_xlabel('Iteration', fontsize=10)
    ax3.set_ylabel('Alignment Deviation', fontsize=10)
    ax3.set_title('Quality Metrics', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Heatmap (spans bottom row)
    ax4 = fig.add_subplot(gs[1, :])
    heatmap_data = archive.to_heatmap()
    im = ax4.imshow(heatmap_data.T, origin='lower', cmap='Reds',
                   vmin=0, vmax=1, aspect='auto', interpolation='bilinear')
    ax4.set_xlabel('Query Indirection (a₁)', fontsize=10)
    ax4.set_ylabel('Authority Framing (a₂)', fontsize=10)
    ax4.set_title('Behavioral Topology Heatmap', fontsize=11, fontweight='bold')
    
    grid_size = archive.grid_size
    tick_positions = np.linspace(0, grid_size-1, 5)
    tick_labels = ['0.0', '0.25', '0.5', '0.75', '1.0']
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels(tick_labels)
    ax4.set_yticks(tick_positions)
    ax4.set_yticklabels(tick_labels)
    
    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label('Alignment Deviation', fontsize=9)
    
    # Overall title
    fig.suptitle('MAP-Elites Summary Dashboard', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    
    return fig
