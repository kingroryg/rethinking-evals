"""
Gaussian Process prediction experiment.

Trains GP models on MAP-Elites archives and evaluates prediction performance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import json
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.core.archive import Archive
from src.prediction.gaussian_process import GPPredictor, evaluate_gp_prediction
from visualization.gp_plots import (
    plot_gp_predictions, plot_gp_prediction_error,
    plot_gp_roc_curve, plot_gp_slice_comparison
)


def run_single_fold(fold_data):
    """Run a single CV fold (for parallel execution)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel
    from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

    X_train, X_test, y_train, y_test, kernel_type, nu, alpha, high_risk_threshold = fold_data

    # Create kernel
    if kernel_type == 'matern':
        kernel = ConstantKernel(1.0) * Matern(length_scale=0.5, nu=nu)
    else:
        from sklearn.gaussian_process.kernels import RBF
        kernel = ConstantKernel(1.0) * RBF(length_scale=0.5)

    # Create and fit GP
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=10
    )
    gp.fit(X_train, y_train)

    # Predict
    y_pred = gp.predict(X_test, return_std=False)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # AUROC
    auroc = None
    y_test_binary = (y_test > high_risk_threshold).astype(int)
    if len(np.unique(y_test_binary)) > 1:
        auroc = roc_auc_score(y_test_binary, y_pred)

    return {'mse': mse, 'r2': r2, 'auroc': auroc}


def parallel_cross_validate(archive, n_folds=5, high_risk_threshold=0.7,
                           kernel_type='matern', nu=2.5, alpha=1e-6, num_workers=4):
    """
    Perform k-fold cross-validation in parallel.

    Args:
        archive: Archive object
        n_folds: Number of CV folds
        high_risk_threshold: Threshold for high-risk classification
        kernel_type: Kernel type for GP
        nu: Matern smoothness parameter
        alpha: Noise level
        num_workers: Number of parallel workers

    Returns:
        Dictionary with CV metrics
    """
    from sklearn.model_selection import KFold

    # Extract data
    X = []
    y = []

    for i in range(archive.grid_size):
        for j in range(archive.grid_size):
            if archive.cells[i, j] is not None:
                cell = archive.cells[i, j]
                X.append(cell.behavior)
                y.append(cell.quality)

    X = np.array(X)
    y = np.array(y)

    # Prepare fold data
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_data_list = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        fold_data_list.append((
            X_train, X_test, y_train, y_test,
            kernel_type, nu, alpha, high_risk_threshold
        ))

    # Run folds in parallel
    print(f"Running {n_folds}-fold CV with {num_workers} workers...")
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(run_single_fold, fd): i for i, fd in enumerate(fold_data_list)}

        for future in tqdm(as_completed(futures), total=n_folds, desc="CV Folds"):
            results.append(future.result())

    # Aggregate results
    mse_scores = [r['mse'] for r in results]
    r2_scores = [r['r2'] for r in results]
    auroc_scores = [r['auroc'] for r in results if r['auroc'] is not None]

    cv_results = {
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores),
        'r2_mean': np.mean(r2_scores),
        'r2_std': np.std(r2_scores),
        'n_folds': n_folds
    }

    if auroc_scores:
        cv_results['auroc_mean'] = np.mean(auroc_scores)
        cv_results['auroc_std'] = np.std(auroc_scores)

    return cv_results


def run_gp_experiment(archive_path: str, output_dir: str = None, num_workers: int = 4):
    """
    Run GP prediction experiment on a saved archive.

    Args:
        archive_path: Path to saved archive
        output_dir: Output directory for results
        num_workers: Number of parallel workers for CV
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

    # Parallel Cross-validation
    print("\n" + "="*60)
    print("CROSS-VALIDATION (PARALLEL)")
    print("="*60)

    cv_results = parallel_cross_validate(
        archive, n_folds=5, high_risk_threshold=0.7,
        kernel_type='matern', nu=2.5, alpha=1e-6,
        num_workers=num_workers
    )

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
    parser.add_argument('--archive', type=str, default=None,
                       help='Path to saved archive (.pkl file)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name to find latest archive for')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: same as archive + /gp_predictions)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers for cross-validation')

    args = parser.parse_args()

    # Handle model shortcut
    if args.model and not args.archive:
        # Find latest archive for model (use script-relative path)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        pattern = os.path.join(project_root, 'data', 'results', f'{args.model}_*', 'final_archive.pkl')

        archives = sorted(glob.glob(pattern))
        if archives:
            args.archive = archives[-1]  # Use most recent
            print(f"Using archive: {args.archive}")
        else:
            raise ValueError(f"No archives found for model {args.model}")
    elif not args.archive:
        raise ValueError("Either --archive or --model must be specified")

    run_gp_experiment(args.archive, args.output_dir, args.workers)


if __name__ == '__main__':
    main()
