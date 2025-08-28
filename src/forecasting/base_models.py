"""
Base classes and core utilities for CityLearn Challenge forecasting models.

This module provides the fundamental BaseForecaster class that all forecasting
models inherit from. Specific implementations are organized in separate modules:

- lstm_models.py: LSTM and recurrent neural networks
- neural_models.py: Feedforward neural networks (ANN, ResNet, etc.)  
- classical_models.py: Traditional ML (Random Forest, Linear, GP, etc.)
- transformer_models.py: Transformer and attention-based models

The BaseForecaster class standardizes the interface for:
1. Model training (fit method)
2. Prediction generation (predict method)  
3. Model persistence (save_model/load_model methods)
4. Performance evaluation utilities
"""

import numpy as np
from typing import Optional, Tuple, Any, Dict
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor


class BaseForecaster(ABC):
    """
    Base class for all forecasting models in the CityLearn Challenge.
    
    This abstract class defines the common interface that all forecasting models
    must implement. It provides standardized methods for training, prediction,
    and model persistence across different algorithm types.
    
    The class supports various time series forecasting approaches including:
    - Deep learning models (LSTM, ANN, Transformers)
    - Machine learning models (Random Forest, Gaussian Process)
    - Traditional statistical models (Linear, Polynomial regression)
    """
    
    def __init__(self, name: str):
        """
        Initialize the base forecaster.
        
        Args:
            name (str): Human-readable name for the forecasting model
                       (e.g., "LSTM", "Random_Forest", "Linear_Regression")
        """
        self.name = name                 # Model identifier for logging/results
        self.model = None               # Will store the actual ML/DL model
        self.is_fitted = False          # Flag to track if model has been trained
        
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        Train the forecasting model on historical time series data.
        
        This method must be implemented by each specific forecaster to handle
        the training process appropriate for that algorithm type.
        
        Args:
            X_train (np.ndarray): Input sequences of shape (n_samples, sequence_length, n_features)
                                 Each sample contains historical values used for prediction
            y_train (np.ndarray): Target values of shape (n_samples, prediction_horizon)
                                 Future values to be predicted
            **kwargs: Additional training parameters specific to each model
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate forecasts for new input sequences.
        
        Uses the trained model to predict future values based on input sequences.
        The model must be fitted before calling this method.
        
        Args:
            X (np.ndarray): Input sequences of shape (n_samples, sequence_length, n_features)
                           Historical data for which to generate predictions
        
        Returns:
            np.ndarray: Predicted values of shape (n_samples, prediction_horizon)
                       Future values forecasted by the model
        
        Raises:
            ValueError: If model has not been fitted yet
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk for later use.
        
        This method can be overridden by subclasses to implement model-specific
        saving logic. The default implementation raises NotImplementedError.
        
        Args:
            filepath (str): Path where the model should be saved
        
        Raises:
            NotImplementedError: Must be implemented by subclasses if needed
        """
        raise NotImplementedError("Subclasses must implement the save_model method")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a previously trained model from disk.
        
        This method can be overridden by subclasses to implement model-specific
        loading logic. The default implementation raises NotImplementedError.
        
        Args:
            filepath (str): Path to the saved model file
        
        Raises:
            NotImplementedError: Must be implemented by subclasses if needed
        """
        raise NotImplementedError("Subclasses must implement the load_model method")


# Import note for users of this module
# ====================================
# 
# Available forecasting models are organized in separate files:
#
# from .lstm_models import LSTMForecaster, BidirectionalLSTMForecaster, ConvLSTMForecaster
# from .transformer_models import TransformerForecaster, TimesFMInspiredForecaster
# from .base_models import get_baseline_forecasters  # For Random Forest, Linear, etc.
#
# Or use the factory function:
# model = create_forecaster('lstm', hidden_units=64)
# baseline_models = get_baseline_forecasters()


def create_forecaster(model_type: str, **kwargs) -> BaseForecaster:
    """
    Factory function to create forecasting models by name.
    
    This utility function provides a convenient way to instantiate forecasting
    models without importing each class individually.
    
    Args:
        model_type (str): Name of the forecasting model to create
                         Options: 'lstm', 'ann', 'random_forest', 'linear', 
                                'polynomial', 'gaussian', 'transformer'
        **kwargs: Parameters to pass to the model constructor
    
    Returns:
        BaseForecaster: Instance of the requested forecasting model
    
    Example:
        >>> lstm_model = create_forecaster('lstm', hidden_units=64, num_layers=2)
        >>> rf_model = create_forecaster('random_forest', n_estimators=200)
    """
    model_type = model_type.lower()
    
    try:
        if model_type in ['lstm', 'bidirectional_lstm', 'conv_lstm']:
            from .lstm_models import LSTMForecaster, BidirectionalLSTMForecaster, ConvLSTMForecaster
            if model_type == 'lstm':
                return LSTMForecaster(**kwargs)
            elif model_type == 'bidirectional_lstm':
                return BidirectionalLSTMForecaster(**kwargs)
            elif model_type == 'conv_lstm':
                return ConvLSTMForecaster(**kwargs)
                
        elif model_type in ['transformer', 'timesfm']:
            from .transformer_models import TransformerForecaster, TimesFMInspiredForecaster
            if model_type == 'transformer':
                return TransformerForecaster(**kwargs)
            elif model_type == 'timesfm':
                return TimesFMInspiredForecaster(**kwargs)
        
        # Se arriviamo qui, il model_type non è riconosciuto
        raise ValueError(f"Unknown model type: {model_type}")
            
    except ImportError as e:
        raise ImportError(f"Could not import {model_type} model. Make sure required dependencies are installed: {e}")
    except Exception as e:
        raise RuntimeError(f"Error creating model {model_type}: {e}")


def get_available_models() -> Dict[str, list]:
    """
    Get a list of all available forecasting models organized by category.
    
    Returns:
        Dict[str, list]: Dictionary mapping model categories to available models
    """
    return {
        'lstm': ['lstm', 'bidirectional_lstm', 'conv_lstm'],
        'transformer': ['transformer', 'timesfm'],
        'baseline': ['random_forest', 'linear_regression', 'polynomial_regression', 'gaussian_process', 'ann']
    }


class SklearnForecaster(BaseForecaster):
    """Wrapper for sklearn models to work with time series forecasting."""
    
    def __init__(self, name: str, model):
        super().__init__(name)
        self.model = model
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, **kwargs) -> None:
        """Train sklearn model on flattened sequences."""
        # Flatten sequences for sklearn models
        X_flat = X_train.reshape(X_train.shape[0], -1)
        y_flat = y_train.flatten() if len(y_train.shape) > 1 else y_train
        
        self.model.fit(X_flat, y_flat)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using sklearn model."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} model must be fitted before prediction")
            
        X_flat = X.reshape(X.shape[0], -1)
        predictions = self.model.predict(X_flat)
        
        # Return as 2D array for consistency
        return predictions.reshape(-1, 1) if len(predictions.shape) == 1 else predictions


def get_baseline_forecasters() -> Dict[str, BaseForecaster]:
    """
    Get collection of baseline forecasting models for comparison.
    
    Returns:
        Dict[str, BaseForecaster]: Dictionary of baseline models
    """
    return {
        'Random_Forest': SklearnForecaster(
            'Random_Forest',
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        ),
        'Polynomial_Regression': SklearnForecaster(
            'Polynomial_Regression',
            Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ])
        ),
        'Gaussian_Process': SklearnForecaster(
            'Gaussian_Process',
            GaussianProcessRegressor(random_state=42)
        ),
        'ANN': SklearnForecaster(
            'ANN',
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        )
    }