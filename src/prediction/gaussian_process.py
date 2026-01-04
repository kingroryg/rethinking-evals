"""
Gaussian Process modeling for predicting Alignment Deviation.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class GPPredictor:
    """
    Gaussian Process predictor for Alignment Deviation.
    """
    
    def __init__(self, 
                 kernel_type: str = 'matern',
                 nu: float = 2.5,
                 alpha: float = 1e-6,
                 normalize_y: bool = True):
        """
        Initialize GP predictor.
        
        Args:
            kernel_type: 'matern', 'rbf', or 'combined'
            nu: Smoothness parameter for Matern kernel
            alpha: Noise level
            normalize_y: Whether to normalize target values
        """
        self.kernel_type = kernel_type
        self.nu = nu
        self.alpha = alpha
        self.normalize_y = normalize_y
        
        # Create kernel
        if kernel_type == 'matern':
            kernel = ConstantKernel(1.0) * Matern(length_scale=0.5, nu=nu)
        elif kernel_type == 'rbf':
            kernel = ConstantKernel(1.0) * RBF(length_scale=0.5)
        elif kernel_type == 'combined':
            kernel = (ConstantKernel(1.0) * Matern(length_scale=0.5, nu=nu) +
                     ConstantKernel(1.0) * RBF(length_scale=0.5))
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        # Create GP
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=normalize_y,
            n_restarts_optimizer=10
        )
        
        self.is_fitted = False
    
    def fit(self, archive):
        """
        Fit GP to archive data.
        
        Args:
            archive: Archive object with behavioral data
        """
        # Extract (behavior, quality) pairs from archive
        X = []
        y = []
        
        for i in range(archive.grid_size):
            for j in range(archive.grid_size):
                if archive.cells[i, j] is not None:
                    cell = archive.cells[i, j]
                    X.append(cell.behavior)
                    y.append(cell.quality)
        
        if len(X) == 0:
            raise ValueError("Archive is empty, cannot fit GP")
        
        X = np.array(X)
        y = np.array(y)
        
        # Fit GP
        self.gp.fit(X, y)
        self.is_fitted = True
        
        print(f"GP fitted with {len(X)} data points")
        print(f"Kernel: {self.gp.kernel_}")
    
    def predict(self, behaviors: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict Alignment Deviation for given behaviors.
        
        Args:
            behaviors: Array of shape (n_samples, 2) with (a1, a2) coordinates
            return_std: Whether to return standard deviation
            
        Returns:
            (mean, std) if return_std=True, else just mean
        """
        if not self.is_fitted:
            raise ValueError("GP not fitted yet. Call fit() first.")
        
        if return_std:
            mean, std = self.gp.predict(behaviors, return_std=True)
            return mean, std
        else:
            mean = self.gp.predict(behaviors, return_std=False)
            return mean, None
    
    def predict_high_risk(self, behaviors: np.ndarray, threshold: float = 0.7) -> np.ndarray:
        """
        Predict binary high-risk classification.
        
        Args:
            behaviors: Array of shape (n_samples, 2)
            threshold: Threshold for high-risk (AD > threshold)
            
        Returns:
            Binary predictions (1 = high-risk, 0 = low-risk)
        """
        mean, _ = self.predict(behaviors, return_std=False)
        return (mean > threshold).astype(int)
    
    def predict_full_grid(self, grid_size: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on a full grid for visualization.
        
        Args:
            grid_size: Size of grid
            
        Returns:
            (mean_grid, std_grid) both of shape (grid_size, grid_size)
        """
        # Create grid
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        X_grid, Y_grid = np.meshgrid(x, y)
        
        # Flatten for prediction
        X_pred = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        
        # Predict
        mean, std = self.predict(X_pred, return_std=True)
        
        # Reshape
        mean_grid = mean.reshape(grid_size, grid_size)
        std_grid = std.reshape(grid_size, grid_size)
        
        return mean_grid, std_grid
    
    def cross_validate(self, archive, n_folds: int = 5, 
                      high_risk_threshold: float = 0.7) -> Dict:
        """
        Perform k-fold cross-validation.
        
        Args:
            archive: Archive object
            n_folds: Number of CV folds
            high_risk_threshold: Threshold for high-risk classification
            
        Returns:
            Dictionary with CV metrics
        """
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
        
        # K-fold CV
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        mse_scores = []
        r2_scores = []
        auroc_scores = []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit GP on train set
            self.gp.fit(X_train, y_train)
            
            # Predict on test set
            y_pred, _ = self.gp.predict(X_test, return_std=True)
            
            # Regression metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            mse_scores.append(mse)
            r2_scores.append(r2)
            
            # Classification metrics (high-risk prediction)
            y_test_binary = (y_test > high_risk_threshold).astype(int)
            
            if len(np.unique(y_test_binary)) > 1:  # Need both classes for AUROC
                auroc = roc_auc_score(y_test_binary, y_pred)
                auroc_scores.append(auroc)
        
        # Refit on full data
        self.fit(archive)
        
        # Return results
        results = {
            'mse_mean': np.mean(mse_scores),
            'mse_std': np.std(mse_scores),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'n_folds': n_folds
        }
        
        if auroc_scores:
            results['auroc_mean'] = np.mean(auroc_scores)
            results['auroc_std'] = np.std(auroc_scores)
        
        return results
    
    def get_acquisition_function(self, strategy: str = 'ucb', kappa: float = 2.0) -> callable:
        """
        Get acquisition function for active learning.
        
        Args:
            strategy: 'ucb' (upper confidence bound) or 'ei' (expected improvement)
            kappa: Exploration parameter for UCB
            
        Returns:
            Acquisition function
        """
        if strategy == 'ucb':
            def ucb(behaviors):
                mean, std = self.predict(behaviors, return_std=True)
                return mean + kappa * std
            return ucb
        
        elif strategy == 'ei':
            def ei(behaviors):
                mean, std = self.predict(behaviors, return_std=True)
                # Expected improvement (simplified)
                return std  # Higher uncertainty = higher EI
            return ei
        
        else:
            raise ValueError(f"Unknown acquisition strategy: {strategy}")


def train_gp_on_archive(archive, kernel_type: str = 'matern', 
                       nu: float = 2.5) -> GPPredictor:
    """
    Convenience function to train GP on archive.
    
    Args:
        archive: Archive object
        kernel_type: Kernel type
        nu: Matern smoothness parameter
        
    Returns:
        Fitted GPPredictor
    """
    gp = GPPredictor(kernel_type=kernel_type, nu=nu)
    gp.fit(archive)
    return gp


def evaluate_gp_prediction(archive, gp: GPPredictor, 
                          high_risk_threshold: float = 0.7) -> Dict:
    """
    Evaluate GP predictions against archive.
    
    Args:
        archive: Archive with ground truth
        gp: Fitted GP predictor
        high_risk_threshold: Threshold for high-risk
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Extract data
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
    
    # Predict
    y_pred, y_std = gp.predict(X, return_std=True)
    
    # Regression metrics
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Classification metrics
    y_true_binary = (y_true > high_risk_threshold).astype(int)
    
    auroc = None
    if len(np.unique(y_true_binary)) > 1:
        auroc = roc_auc_score(y_true_binary, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'auroc': auroc,
        'mean_std': np.mean(y_std),
        'n_samples': len(X)
    }
