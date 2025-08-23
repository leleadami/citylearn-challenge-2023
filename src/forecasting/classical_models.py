"""
Classical Machine Learning Models for CityLearn Challenge

This module implements traditional machine learning algorithms for building
energy forecasting. While neural networks capture complex patterns, classical
methods often provide:

1. Interpretability: Clear understanding of feature importance
2. Speed: Fast training and inference
3. Robustness: Less prone to overfitting with small datasets
4. Baseline: Strong baselines for comparison with advanced methods

Classical models are particularly valuable for:
- Quick prototyping and feature selection
- Situations with limited training data
- When interpretability is crucial for stakeholders
- Ensemble methods combining multiple approaches
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from .base_models import BaseForecaster


class RandomForestForecaster(BaseForecaster):
    """
    Random Forest ensemble method for energy forecasting.
    
    Random Forest is particularly effective for building energy prediction because:
    - Handles non-linear relationships between weather and energy consumption
    - Provides feature importance rankings (which variables matter most)
    - Robust to outliers and missing data
    - No need for feature scaling or normalization
    - Natural ensemble reduces overfitting
    
    Applications in building energy:
    - Peak demand prediction based on weather conditions
    - HVAC energy consumption modeling
    - Solar generation forecasting with weather features
    - Multi-building portfolio optimization
    """
    
    def __init__(self, 
                 n_estimators: int = 200,  # Increased from 100
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2,
                 max_features: str = 'sqrt',
                 random_state: int = 42):
        """
        Initialize Random Forest forecaster.
        
        Args:
            n_estimators: Number of decision trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples required to split internal node
            min_samples_leaf: Minimum samples required at leaf node
            max_features: Number of features to consider for best split
            random_state: Random seed for reproducibility
        """
        super().__init__("Random_Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        
        # Initialize scikit-learn model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> None:
        """
        Train Random Forest on building energy data.
        
        Args:
            X_train: Training features of shape (n_samples, n_features)
            y_train: Training targets of shape (n_samples,) or (n_samples, 1)
        """
        # Flatten input for sklearn compatibility
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        y_train_flat = y_train.flatten() if len(y_train.shape) > 1 else y_train
        
        # Train the random forest
        self.model.fit(X_train_flat, y_train_flat)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using trained Random Forest."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_flat = X.reshape(X.shape[0], -1)
        predictions = self.model.predict(X_flat)
        return predictions
        
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores from trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        return self.model.feature_importances_


class LinearForecaster(BaseForecaster):
    """
    Linear regression models for energy forecasting.
    
    Linear models are valuable baselines that often perform surprisingly well
    for energy prediction due to strong linear relationships between:
    - Temperature and heating/cooling demand
    - Solar irradiance and solar generation
    - Time-of-day patterns and electricity consumption
    
    Supports multiple regularization strategies:
    - OLS: Standard linear regression
    - Ridge: L2 regularization to prevent overfitting
    - Lasso: L1 regularization for feature selection
    - ElasticNet: Combined L1/L2 regularization
    """
    
    def __init__(self, 
                 method: str = 'ols',
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5):
        """
        Initialize linear forecaster.
        
        Args:
            method: Type of linear regression ('ols', 'ridge', 'lasso', 'elasticnet')
            alpha: Regularization strength (ignored for OLS)
            l1_ratio: L1 ratio for ElasticNet (0=Ridge, 1=Lasso)
        """
        super().__init__(f"Linear_{method.upper()}")
        self.method = method.lower()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
        # Initialize appropriate scikit-learn model
        if self.method == 'ols':
            self.model = LinearRegression()
        elif self.method == 'ridge':
            self.model = Ridge(alpha=self.alpha, random_state=42)
        elif self.method == 'lasso':
            self.model = Lasso(alpha=self.alpha, random_state=42, max_iter=2000)
        elif self.method == 'elasticnet':
            self.model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                random_state=42,
                max_iter=2000
            )
        else:
            raise ValueError(f"Unknown method: {method}")
            
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> None:
        """Train linear regression model."""
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        y_train_flat = y_train.flatten() if len(y_train.shape) > 1 else y_train
        
        self.model.fit(X_train_flat, y_train_flat)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate linear predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_flat = X.reshape(X.shape[0], -1)
        predictions = self.model.predict(X_flat)
        return predictions
        
    def get_coefficients(self) -> np.ndarray:
        """Get linear regression coefficients."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get coefficients")
        return self.model.coef_


class PolynomialForecaster(BaseForecaster):
    """
    Polynomial regression for capturing non-linear energy patterns.
    
    Polynomial features can capture important non-linear relationships such as:
    - Quadratic temperature effects (comfort zone behavior)
    - Interaction effects between weather variables
    - Seasonal variations in energy efficiency
    - Equipment performance curves
    
    The model creates polynomial features up to a specified degree and then
    applies linear regression with optional regularization.
    """
    
    def __init__(self, 
                 degree: int = 2,
                 include_bias: bool = True,
                 interaction_only: bool = False,
                 regularization: str = 'ridge',
                 alpha: float = 1.0):
        """
        Initialize polynomial forecaster.
        
        Args:
            degree: Degree of polynomial features (2=quadratic, 3=cubic, etc.)
            include_bias: Whether to include bias/intercept term
            interaction_only: Only include interaction terms, not powers
            regularization: Regularization method ('ridge', 'lasso', 'elasticnet')
            alpha: Regularization strength
        """
        super().__init__(f"Polynomial_Deg{degree}")
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.regularization = regularization
        self.alpha = alpha
        
        # Create polynomial feature transformer
        self.poly_features = PolynomialFeatures(
            degree=self.degree,
            include_bias=self.include_bias,
            interaction_only=self.interaction_only
        )
        
        # Select regression model based on regularization
        if regularization == 'ridge':
            regressor = Ridge(alpha=self.alpha, random_state=42)
        elif regularization == 'lasso':
            regressor = Lasso(alpha=self.alpha, random_state=42, max_iter=2000)
        elif regularization == 'elasticnet':
            regressor = ElasticNet(alpha=self.alpha, random_state=42, max_iter=2000)
        else:
            regressor = LinearRegression()
            
        # Create pipeline: polynomial features -> regression
        self.model = Pipeline([
            ('poly', self.poly_features),
            ('regressor', regressor)
        ])
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> None:
        """Train polynomial regression model."""
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        y_train_flat = y_train.flatten() if len(y_train.shape) > 1 else y_train
        
        self.model.fit(X_train_flat, y_train_flat)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate polynomial predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_flat = X.reshape(X.shape[0], -1)
        predictions = self.model.predict(X_flat)
        return predictions


class GaussianForecaster(BaseForecaster):
    """
    Gaussian Process regression for probabilistic energy forecasting.
    
    Gaussian Processes are particularly valuable for energy systems because they:
    - Provide uncertainty estimates alongside predictions
    - Automatically adapt complexity to available data
    - Handle irregular and sparse time series naturally
    - Incorporate prior knowledge through kernel selection
    
    Uncertainty quantification is crucial for:
    - Risk assessment in energy trading
    - Robust control system design
    - Confidence intervals for planning
    - Detection of anomalous consumption patterns
    """
    
    def __init__(self, 
                 kernel_type: str = 'rbf',
                 length_scale: float = 1.0,
                 noise_level: float = 0.1,
                 alpha: float = 1e-10):
        """
        Initialize Gaussian Process forecaster.
        
        Args:
            kernel_type: Type of kernel ('rbf', 'matern', 'combined')
            length_scale: Characteristic length scale for kernel
            noise_level: Expected noise level in observations
            alpha: Regularization parameter
        """
        super().__init__(f"GP_{kernel_type.upper()}")
        self.kernel_type = kernel_type.lower()
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.alpha = alpha
        
        # Define kernel based on type
        if self.kernel_type == 'rbf':
            kernel = ConstantKernel(1.0) * RBF(length_scale=self.length_scale)
        elif self.kernel_type == 'matern':
            kernel = ConstantKernel(1.0) * Matern(length_scale=self.length_scale, nu=2.5)
        elif self.kernel_type == 'combined':
            # Combination of RBF for smooth trends and Matern for local variations
            kernel = (
                ConstantKernel(1.0) * RBF(length_scale=self.length_scale) +
                ConstantKernel(0.1) * Matern(length_scale=self.length_scale/3, nu=1.5)
            )
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
            
        # Initialize Gaussian Process regressor
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            random_state=42,
            normalize_y=True,  # Normalize targets for numerical stability
            n_restarts_optimizer=3  # Multiple random starts for optimization
        )
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> None:
        """Train Gaussian Process model."""
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        y_train_flat = y_train.flatten() if len(y_train.shape) > 1 else y_train
        
        # Limit training set size for computational efficiency
        # GP complexity is O(n³), so large datasets require subsampling
        # Increased limit for better performance
        max_samples = 3000  # Increased from 2000
        if len(X_train_flat) > max_samples:
            indices = np.random.choice(len(X_train_flat), max_samples, replace=False)
            X_train_flat = X_train_flat[indices]
            y_train_flat = y_train_flat[indices]
            
        self.model.fit(X_train_flat, y_train_flat)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        """
        Generate GP predictions with optional uncertainty estimates.
        
        Args:
            X: Input features
            return_std: Whether to return uncertainty estimates
            
        Returns:
            Predictions (and standard deviations if return_std=True)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_flat = X.reshape(X.shape[0], -1)
        
        if return_std:
            predictions, std = self.model.predict(X_flat, return_std=True)
            return predictions, std
        else:
            predictions = self.model.predict(X_flat)
            return predictions
            
    def predict_with_uncertainty(self, X: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with confidence intervals.
        
        Args:
            X: Input features
            confidence: Confidence level (e.g., 0.95 for 95% confidence interval)
            
        Returns:
            (predictions, lower_bound, upper_bound)
        """
        from scipy.stats import norm
        
        predictions, std = self.predict(X, return_std=True)
        
        # Calculate confidence interval based on normal distribution
        z_score = norm.ppf((1 + confidence) / 2)
        margin = z_score * std
        
        lower_bound = predictions - margin
        upper_bound = predictions + margin
        
        return predictions, lower_bound, upper_bound


class SVRForecaster(BaseForecaster):
    """
    Support Vector Regression for robust energy forecasting.
    
    SVR is particularly useful for energy data because it:
    - Handles outliers well (e.g., extreme weather events)
    - Works with high-dimensional feature spaces
    - Provides sparse solutions (only uses support vectors)
    - Robust to noise and missing data
    """
    
    def __init__(self, 
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 gamma: str = 'scale',
                 epsilon: float = 0.1):
        """Initialize SVR forecaster."""
        super().__init__(f"SVR_{kernel.upper()}")
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon
        
        from sklearn.svm import SVR
        self.model = SVR(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            epsilon=self.epsilon
        )
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> None:
        """Train SVR model."""
        from sklearn.preprocessing import StandardScaler
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        y_train_flat = y_train.flatten() if len(y_train.shape) > 1 else y_train
        
        # SVR benefits from feature scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        
        self.model.fit(X_train_scaled, y_train_flat)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate SVR predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.transform(X_flat)
        predictions = self.model.predict(X_scaled)
        return predictions