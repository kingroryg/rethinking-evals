"""
Heatmap visualization for behavioral topology.
Creates 2D heatmaps showing the Manifold of Failure.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, List, Tuple
import os


def create_behavioral_heatmap(archive,
                              title: str = "Behavioral Topology",
                              figsize: Tuple[int, int] = (10, 8),
                              cmap: str = 'Reds',
                              vmin: float = 0.0,
                              vmax: float = 1.0,
                              save_path: Optional[str] = None,
                              dpi: int = 300,
                              show_values: bool = False,
                              annotate_threshold: float = 0.7):
    """
    Create a 2D heatmap of the behavioral space.
    
    Args:
        archive: Archive object containing the behavioral map
        title: Plot title
        figsize: Figure size (width, height)
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        save_path: Path to save figure (if None, displays instead)
        dpi: DPI for saved figure
        show_values: Whether to show values in cells
        annotate_threshold: Only annotate cells above this threshold
        
    Returns:
        matplotlib figure object
    """
    # Get heatmap data
    heatmap_data = archive.to_heatmap()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(heatmap_data.T, origin='lower', cmap=cmap, 
                   vmin=vmin, vmax=vmax, aspect='auto', interpolation='bilinear')
    
    # Set labels
    ax.set_xlabel('Query Indirection (a₁)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Authority Framing (a₂)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Set ticks
    grid_size = archive.grid_size
    tick_positions = np.linspace(0, grid_size-1, 5)
    tick_labels = ['0.0', '0.25', '0.5', '0.75', '1.0']
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Alignment Deviation', fontsize=12, fontweight='bold')
    
    # Optionally annotate high-value cells
    if show_values:
        for i in range(grid_size):
            for j in range(grid_size):
                if not np.isnan(heatmap_data[i, j]) and heatmap_data[i, j] > annotate_threshold:
                    text = ax.text(i, j, f'{heatmap_data[i, j]:.2f}',
                                 ha="center", va="center", color="white", fontsize=8)
    
    # Add grid
    ax.grid(False)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()
    
    return fig


def create_comparison_heatmaps(archives: List,
                              model_names: List[str],
                              figsize: Tuple[int, int] = (18, 5),
                              cmap: str = 'Reds',
                              save_path: Optional[str] = None,
                              dpi: int = 300):
    """
    Create side-by-side heatmaps for comparing multiple models.
    
    Args:
        archives: List of Archive objects
        model_names: List of model names
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save figure
        dpi: DPI for saved figure
        
    Returns:
        matplotlib figure object
    """
    n_models = len(archives)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    # Find global min/max for consistent coloring
    all_data = [archive.to_heatmap() for archive in archives]
    vmin = 0.0
    vmax = max(np.nanmax(data) for data in all_data)
    
    for idx, (archive, model_name, ax) in enumerate(zip(archives, model_names, axes)):
        heatmap_data = archive.to_heatmap()
        
        # Create heatmap
        im = ax.imshow(heatmap_data.T, origin='lower', cmap=cmap,
                      vmin=vmin, vmax=vmax, aspect='auto', interpolation='bilinear')
        
        # Labels
        ax.set_xlabel('Query Indirection (a₁)', fontsize=12)
        ax.set_ylabel('Authority Framing (a₂)', fontsize=12)
        ax.set_title(model_name, fontsize=14, fontweight='bold')
        
        # Ticks
        grid_size = archive.grid_size
        tick_positions = np.linspace(0, grid_size-1, 5)
        tick_labels = ['0.0', '0.25', '0.5', '0.75', '1.0']
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
    
    # Add shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Alignment Deviation', fontsize=12, fontweight='bold')
    
    # Overall title
    fig.suptitle('Behavioral Topology Comparison', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Comparison heatmap saved to {save_path}")
    else:
        plt.show()
    
    return fig


def create_attraction_basin_visualization(archive,
                                         threshold: float = 0.5,
                                         figsize: Tuple[int, int] = (10, 8),
                                         save_path: Optional[str] = None,
                                         dpi: int = 300):
    """
    Visualize behavioral attraction basins.
    
    Highlights regions where Alignment Deviation exceeds threshold.
    
    Args:
        archive: Archive object
        threshold: Minimum AD to be considered a basin
        figsize: Figure size
        save_path: Path to save figure
        dpi: DPI for saved figure
        
    Returns:
        matplotlib figure object
    """
    heatmap_data = archive.to_heatmap()
    
    # Create binary mask for basins
    basin_mask = np.where(heatmap_data > threshold, 1, 0)
    basin_mask = np.where(np.isnan(heatmap_data), np.nan, basin_mask)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left: continuous heatmap
    im1 = ax1.imshow(heatmap_data.T, origin='lower', cmap='Reds',
                     vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    ax1.set_xlabel('Query Indirection (a₁)', fontsize=12)
    ax1.set_ylabel('Authority Framing (a₂)', fontsize=12)
    ax1.set_title('Alignment Deviation', fontsize=14, fontweight='bold')
    
    # Right: binary basins
    im2 = ax2.imshow(basin_mask.T, origin='lower', cmap='RdYlGn_r',
                     vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    ax2.set_xlabel('Query Indirection (a₁)', fontsize=12)
    ax2.set_ylabel('Authority Framing (a₂)', fontsize=12)
    ax2.set_title(f'Attraction Basins (AD > {threshold})', fontsize=14, fontweight='bold')
    
    # Set ticks for both
    grid_size = archive.grid_size
    tick_positions = np.linspace(0, grid_size-1, 5)
    tick_labels = ['0.0', '0.25', '0.5', '0.75', '1.0']
    
    for ax in [ax1, ax2]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
    
    # Colorbars
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Attraction basin visualization saved to {save_path}")
    else:
        plt.show()
    
    return fig


def create_3d_surface_plot(archive,
                          figsize: Tuple[int, int] = (12, 9),
                          save_path: Optional[str] = None,
                          dpi: int = 300):
    """
    Create a 3D surface plot of the behavioral topology.
    
    Args:
        archive: Archive object
        figsize: Figure size
        save_path: Path to save figure
        dpi: DPI for saved figure
        
    Returns:
        matplotlib figure object
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    heatmap_data = archive.to_heatmap()
    grid_size = archive.grid_size
    
    # Create meshgrid
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = heatmap_data.T
    
    # Replace NaN with 0 for visualization
    Z = np.nan_to_num(Z, nan=0.0)
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='Reds', alpha=0.8,
                          linewidth=0, antialiased=True)
    
    # Labels
    ax.set_xlabel('Query Indirection (a₁)', fontsize=12, labelpad=10)
    ax.set_ylabel('Authority Framing (a₂)', fontsize=12, labelpad=10)
    ax.set_zlabel('Alignment Deviation', fontsize=12, labelpad=10)
    ax.set_title('3D Behavioral Topology', fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Set viewing angle
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"3D surface plot saved to {save_path}")
    else:
        plt.show()
    
    return fig


def create_contour_plot(archive,
                       levels: int = 10,
                       figsize: Tuple[int, int] = (10, 8),
                       save_path: Optional[str] = None,
                       dpi: int = 300):
    """
    Create a contour plot of the behavioral topology.
    
    Args:
        archive: Archive object
        levels: Number of contour levels
        figsize: Figure size
        save_path: Path to save figure
        dpi: DPI for saved figure
        
    Returns:
        matplotlib figure object
    """
    heatmap_data = archive.to_heatmap()
    grid_size = archive.grid_size
    
    # Create meshgrid
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = heatmap_data.T
    
    # Replace NaN with 0
    Z = np.nan_to_num(Z, nan=0.0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filled contour plot
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap='Reds', alpha=0.8)
    
    # Contour lines
    contour = ax.contour(X, Y, Z, levels=levels, colors='black', 
                        linewidths=0.5, alpha=0.4)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Labels
    ax.set_xlabel('Query Indirection (a₁)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Authority Framing (a₂)', fontsize=14, fontweight='bold')
    ax.set_title('Behavioral Topology Contours', fontsize=16, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('Alignment Deviation', fontsize=12, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Contour plot saved to {save_path}")
    else:
        plt.show()
    
    return fig


def export_all_visualizations(archive, model_name: str, output_dir: str, dpi: int = 300):
    """
    Export all visualization types for an archive.
    
    Args:
        archive: Archive object
        model_name: Name of the model
        output_dir: Directory to save visualizations
        dpi: DPI for saved figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Heatmap
    create_behavioral_heatmap(
        archive,
        title=f"Behavioral Topology - {model_name}",
        save_path=f"{output_dir}/{model_name}_heatmap.png",
        dpi=dpi
    )
    
    # Attraction basins
    create_attraction_basin_visualization(
        archive,
        save_path=f"{output_dir}/{model_name}_basins.png",
        dpi=dpi
    )
    
    # 3D surface
    create_3d_surface_plot(
        archive,
        save_path=f"{output_dir}/{model_name}_3d.png",
        dpi=dpi
    )
    
    # Contour plot
    create_contour_plot(
        archive,
        save_path=f"{output_dir}/{model_name}_contour.png",
        dpi=dpi
    )
    
    print(f"All visualizations exported to {output_dir}")
