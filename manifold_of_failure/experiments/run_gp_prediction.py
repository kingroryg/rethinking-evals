"""
Gaussian Process prediction experiment.

Trains GP models on MAP-Elites archives and evaluates prediction performance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
from src.core.archive import Archive
from src.prediction.gaussian_process import GPPredictor, evaluate_gp_prediction
from visualization.gp_plots import (
    plot_gp_predictions, plot_gp_prediction_error,
    plot_gp_roc_curve, plot_gp_slice_comparison
)


def run_gp_experiment(archive_path: str, output_dir: str = None):
    """
    Run GP prediction experiment on a saved archive.
    
    Args:
        archive_path: Path to saved archive
        output_dir: Output directory for results
    """
    print("="*60)
    print("GAUSSIAN PROCESS PREDICTION EXPERIMENT")
    print("="*60)
    
    # Load archive
    print(f"\nLoading archive from: {archive_path}")
    archive = Archive.load(archive_path)
    
    stats = archive.get_statistics()
    print(f"Archive statistics:")
    print(f"  Coverage: {stats['coverage']:.2f}%")
    print(f"  Num filled cells: {stats['num_filled']}")
    print(f"  Peak quality: {stats['peak_quality']:.3f}")
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.dirname(archive_path) + "/gp_predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Train GP
    print("\n" + "="*60)
    print("TRAINING GAUSSIAN PROCESS")
    print("="*60)
    
    gp = GPPredictor(kernel_type='matern', nu=2.5, alpha=1e-6)
    gp.fit(archive)
    
    # Cross-validation
    print("\n" + "="*60)
    print("CROSS-VALIDATION")
    print("="*60)
    
    cv_results = gp.cross_validate(archive, n_folds=5, high_risk_threshold=0.7)
    
    print(f"\nCross-validation results (5-fold):")
    print(f"  MSE: {cv_results['mse_mean']:.4f} ± {cv_results['mse_std']:.4f}")
    print(f"  R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
    
    if 'auroc_mean' in cv_results:
        print(f"  AUROC: {cv_results['auroc_mean']:.4f} ± {cv_results['auroc_std']:.4f}")
    
    # Evaluation on full archive
    print("\n" + "="*60)
    print("EVALUATION ON FULL ARCHIVE")
    print("="*60)
    
    eval_results = evaluate_gp_prediction(archive, gp, high_risk_threshold=0.7)
    
    print(f"\nEvaluation results:")
    print(f"  MSE: {eval_results['mse']:.4f}")
    print(f"  RMSE: {eval_results['rmse']:.4f}")
    print(f"  R²: {eval_results['r2']:.4f}")
    if eval_results['auroc'] is not None:
        print(f"  AUROC: {eval_results['auroc']:.4f}")
    print(f"  Mean uncertainty: {eval_results['mean_std']:.4f}")
    
    # Save results
    import json
    with open(f"{output_dir}/gp_results.json", 'w') as f:
        json.dump({
            'cross_validation': cv_results,
            'evaluation': eval_results
        }, f, indent=2)
    
    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Predictions vs actual
    plot_gp_predictions(
        archive, gp,
        save_path=f"{output_dir}/gp_predictions.png"
    )
    
    # 2. Prediction error
    plot_gp_prediction_error(
        archive, gp,
        save_path=f"{output_dir}/gp_error.png"
    )
    
    # 3. ROC curve (if applicable)
    if eval_results['auroc'] is not None:
        # Extract data for ROC
        X = []
        y_true = []
        
        for i in range(archive.grid_size):
            for j in range(archive.grid_size):
                if archive.cells[i, j] is not None:
                    cell = archive.cells[i, j]
                    X.append(cell.behavior)
                    y_true.append(cell.quality)
        
        X = np.array(X)
        y_true = np.array(y_true)
        
        y_pred, _ = gp.predict(X, return_std=False)
        y_true_binary = (y_true > 0.7).astype(int)
        
        plot_gp_roc_curve(
            y_true_binary, y_pred,
            threshold=0.7,
            save_path=f"{output_dir}/gp_roc.png"
        )
    
    # 4. Slice comparisons
    plot_gp_slice_comparison(
        archive, gp,
        slice_dim=0, slice_value=0.5,
        save_path=f"{output_dir}/gp_slice_a1.png"
    )
    
    plot_gp_slice_comparison(
        archive, gp,
        slice_dim=1, slice_value=0.5,
        save_path=f"{output_dir}/gp_slice_a2.png"
    )
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
    
    print("\n" + "="*60)
    print("GP EXPERIMENT COMPLETE")
    print("="*60)
    
    return gp, cv_results, eval_results


def main():
    parser = argparse.ArgumentParser(description='Run GP prediction experiment')
    parser.add_argument('--archive', type=str, required=True,
                       help='Path to saved archive (.pkl file)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: same as archive + /gp_predictions)')
    
    args = parser.parse_args()
    
    run_gp_experiment(args.archive, args.output_dir)


if __name__ == '__main__':
    main()
