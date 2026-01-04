"""
Gaussian Process prediction visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import roc_auc_score, roc_curve


def plot_gp_predictions(archive,
                       gp_model: GaussianProcessRegressor,
                       figsize: Tuple[int, int] = (16, 5),
                       save_path: Optional[str] = None,
                       dpi: int = 300):
    """
    Plot GP predictions vs actual values.
    
    Args:
        archive: Archive with actual values
        gp_model: Trained GP model
        figsize: Figure size
        save_path: Path to save figure
        dpi: DPI for saved figure
    """
    grid_size = archive.grid_size
    
    # Get actual values
    actual_heatmap = archive.to_heatmap()
    
    # Generate prediction grid
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    # Flatten for prediction
    X_pred = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    # Predict
    y_pred_mean, y_pred_std = gp_model.predict(X_pred, return_std=True)
    
    # Reshape
    pred_mean = y_pred_mean.reshape(grid_size, grid_size)
    pred_std = y_pred_std.reshape(grid_size, grid_size)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Actual values
    im1 = axes[0].imshow(actual_heatmap.T, origin='lower', cmap='Reds',
                        vmin=0, vmax=1, aspect='auto', interpolation='bilinear')
    axes[0].set_title('Actual Alignment Deviation', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Query Indirection (a₁)', fontsize=10)
    axes[0].set_ylabel('Authority Framing (a₂)', fontsize=10)
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 2. GP predictions (mean)
    im2 = axes[1].imshow(pred_mean.T, origin='lower', cmap='Reds',
                        vmin=0, vmax=1, aspect='auto', interpolation='bilinear')
    axes[1].set_title('GP Predicted Mean', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Query Indirection (a₁)', fontsize=10)
    axes[1].set_ylabel('Authority Framing (a₂)', fontsize=10)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. GP uncertainty (std)
    im3 = axes[2].imshow(pred_std.T, origin='lower', cmap='viridis',
                        aspect='auto', interpolation='bilinear')
    axes[2].set_title('GP Uncertainty (Std Dev)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Query Indirection (a₁)', fontsize=10)
    axes[2].set_ylabel('Authority Framing (a₂)', fontsize=10)
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Set ticks for all
    tick_positions = np.linspace(0, grid_size-1, 5)
    tick_labels = ['0.0', '0.25', '0.5', '0.75', '1.0']
    for ax in axes:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
    
    fig.suptitle('Gaussian Process Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    
    return fig


def plot_gp_prediction_error(archive,
                            gp_model: GaussianProcessRegressor,
                            figsize: Tuple[int, int] = (10, 8),
                            save_path: Optional[str] = None,
                            dpi: int = 300):
    """
    Plot prediction error heatmap.
    """
    grid_size = archive.grid_size
    actual_heatmap = archive.to_heatmap()
    
    # Generate predictions
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X_grid, Y_grid = np.meshgrid(x, y)
    X_pred = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    y_pred_mean, _ = gp_model.predict(X_pred, return_std=True)
    pred_mean = y_pred_mean.reshape(grid_size, grid_size)
    
    # Compute error
    error = np.abs(actual_heatmap - pred_mean)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(error.T, origin='lower', cmap='YlOrRd',
                  aspect='auto', interpolation='bilinear')
    
    ax.set_xlabel('Query Indirection (a₁)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Authority Framing (a₂)', fontsize=12, fontweight='bold')
    ax.set_title('GP Prediction Error (|Actual - Predicted|)', fontsize=14, fontweight='bold')
    
    tick_positions = np.linspace(0, grid_size-1, 5)
    tick_labels = ['0.0', '0.25', '0.5', '0.75', '1.0']
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Absolute Error', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    
    return fig


def plot_gp_roc_curve(y_true, y_pred_proba,
                     threshold: float = 0.7,
                     figsize: Tuple[int, int] = (8, 8),
                     save_path: Optional[str] = None,
                     dpi: int = 300):
    """
    Plot ROC curve for GP high-risk prediction.
    
    Args:
        y_true: True binary labels (1 if AD > threshold, 0 otherwise)
        y_pred_proba: Predicted probabilities from GP
        threshold: Threshold for high-risk classification
        figsize: Figure size
        save_path: Path to save figure
        dpi: DPI for saved figure
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, linewidth=2, label=f'GP Predictor (AUC = {auc:.3f})', color='#2E86AB')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier', alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'ROC Curve for High-Risk Prediction (AD > {threshold})', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add AUC text box
    textstr = f'AUROC = {auc:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.6, 0.2, textstr, transform=ax.transAxes, fontsize=14,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    
    return fig


def plot_gp_slice_comparison(archive,
                            gp_model: GaussianProcessRegressor,
                            slice_dim: int = 0,
                            slice_value: float = 0.5,
                            figsize: Tuple[int, int] = (10, 6),
                            save_path: Optional[str] = None,
                            dpi: int = 300):
    """
    Plot 1D slice comparison of actual vs predicted.
    
    Args:
        archive: Archive with actual values
        gp_model: Trained GP model
        slice_dim: Dimension to slice (0 for a1, 1 for a2)
        slice_value: Value at which to slice
        figsize: Figure size
        save_path: Path to save figure
        dpi: DPI for saved figure
    """
    grid_size = archive.grid_size
    
    # Get actual values
    actual_heatmap = archive.to_heatmap()
    
    # Create slice coordinates
    if slice_dim == 0:
        # Slice along a1 at fixed a2
        slice_idx = int(slice_value * grid_size)
        actual_slice = actual_heatmap[:, slice_idx]
        
        x_coords = np.linspace(0, 1, grid_size)
        X_pred = np.column_stack([x_coords, np.full(grid_size, slice_value)])
        xlabel = 'Query Indirection (a₁)'
        title = f'Slice at Authority = {slice_value:.2f}'
    else:
        # Slice along a2 at fixed a1
        slice_idx = int(slice_value * grid_size)
        actual_slice = actual_heatmap[slice_idx, :]
        
        x_coords = np.linspace(0, 1, grid_size)
        X_pred = np.column_stack([np.full(grid_size, slice_value), x_coords])
        xlabel = 'Authority Framing (a₂)'
        title = f'Slice at Indirection = {slice_value:.2f}'
    
    # Predict
    y_pred_mean, y_pred_std = gp_model.predict(X_pred, return_std=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual (non-NaN values only)
    valid_mask = ~np.isnan(actual_slice)
    ax.scatter(x_coords[valid_mask], actual_slice[valid_mask], 
              s=50, label='Actual', color='#E63946', alpha=0.7, zorder=3)
    
    # Plot GP prediction
    ax.plot(x_coords, y_pred_mean, linewidth=2, label='GP Mean', color='#2E86AB')
    ax.fill_between(x_coords, 
                    y_pred_mean - 2*y_pred_std, 
                    y_pred_mean + 2*y_pred_std,
                    alpha=0.3, color='#2E86AB', label='95% Confidence')
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Alignment Deviation', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    
    return fig
