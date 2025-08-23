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
# Specific forecasting models have been moved to separate files for better organization:
#
# from .lstm_models import LSTMForecaster, BidirectionalLSTMForecaster, ConvLSTMForecaster
# from .neural_models import ANNForecaster, ResNetForecaster, AutoencoderForecaster  
# from .classical_models import RandomForestForecaster, LinearForecaster, PolynomialForecaster, GaussianForecaster
# from .transformer_models import TransformerForecaster, TimesFMInspiredForecaster
#
# Or use the convenient imports from the main forecasting module:
# from src.forecasting import LSTMForecaster, RandomForestForecaster, etc.


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
                
        elif model_type in ['ann', 'resnet', 'autoencoder']:
            from .neural_models import ANNForecaster, ResNetForecaster, AutoencoderForecaster
            if model_type == 'ann':
                return ANNForecaster(**kwargs)
            elif model_type == 'resnet':
                return ResNetForecaster(**kwargs)
            elif model_type == 'autoencoder':
                return AutoencoderForecaster(**kwargs)
                
        elif model_type in ['random_forest', 'linear', 'polynomial', 'gaussian', 'svr']:
            from .classical_models import (RandomForestForecaster, LinearForecaster, 
                                         PolynomialForecaster, GaussianForecaster, SVRForecaster)
            if model_type == 'random_forest':
                return RandomForestForecaster(**kwargs)
            elif model_type == 'linear':
                return LinearForecaster(**kwargs)
            elif model_type == 'polynomial':
                return PolynomialForecaster(**kwargs)
            elif model_type == 'gaussian':
                return GaussianForecaster(**kwargs)
            elif model_type == 'svr':
                return SVRForecaster(**kwargs)
                
        elif model_type in ['transformer', 'timesfm']:
            from .transformer_models import TransformerForecaster, TimesFMInspiredForecaster
            if model_type == 'transformer':
                return TransformerForecaster(**kwargs)
            elif model_type == 'timesfm':
                return TimesFMInspiredForecaster(**kwargs)
                
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    except ImportError as e:
        raise ImportError(f"Could not import {model_type} model. Make sure required dependencies are installed: {e}")


def get_available_models() -> Dict[str, list]:
    """
    Get a list of all available forecasting models organized by category.
    
    Returns:
        Dict[str, list]: Dictionary mapping model categories to available models
    """
    return {
        'lstm': ['lstm', 'bidirectional_lstm', 'conv_lstm'],
        'neural': ['ann', 'resnet', 'autoencoder'],
        'classical': ['random_forest', 'linear', 'polynomial', 'gaussian', 'svr'],
        'transformer': ['transformer', 'timesfm']
    }