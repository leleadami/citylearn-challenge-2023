"""
CityLearn Challenge Forecasting Module

This package provides a comprehensive suite of forecasting models for building
energy prediction, organized by algorithm type:

- base_models.py: Base classes and legacy implementations
- lstm_models.py: LSTM and recurrent neural networks
- neural_models.py: Feedforward neural networks (ANN, ResNet, etc.)
- classical_models.py: Traditional ML (Random Forest, Linear, GP, etc.)
- transformer_models.py: Transformer and attention-based models
- citylearn_challenge.py: Challenge-specific implementations

Usage:
    from src.forecasting.lstm_models import LSTMForecaster
    from src.forecasting.classical_models import RandomForestForecaster
    from src.forecasting.neural_models import ANNForecaster
    from src.forecasting.transformer_models import TransformerForecaster
"""

# Import all forecasting models for easy access
from .base_models import BaseForecaster

# LSTM and Recurrent Models
try:
    from .lstm_models import LSTMForecaster, BidirectionalLSTMForecaster, ConvLSTMForecaster
except ImportError as e:
    print(f"Warning: Could not import LSTM models: {e}")

# Classical Machine Learning Models
try:
    from .classical_models import (
        RandomForestForecaster, LinearForecaster, PolynomialForecaster,
        GaussianForecaster, SVRForecaster
    )
except ImportError as e:
    print(f"Warning: Could not import classical models: {e}")

# Neural Network Models
try:
    from .neural_models import (
        ANNForecaster, ResNetForecaster, AutoencoderForecaster,
        EnsembleNeuralForecaster
    )
except ImportError as e:
    print(f"Warning: Could not import neural models: {e}")

# Transformer Models
try:
    from .transformer_models import TransformerForecaster, TimesFMInspiredForecaster
except ImportError as e:
    print(f"Warning: Could not import transformer models: {e}")

# Challenge-specific implementations
try:
    from .citylearn_challenge import CityLearnChallengeForecaster
except ImportError as e:
    print(f"Warning: Could not import challenge forecaster: {e}")


# Define what gets exported when using "from forecasting import *"
__all__ = [
    # Base classes
    'BaseForecaster',
    
    # LSTM models
    'LSTMForecaster',
    'BidirectionalLSTMForecaster', 
    'ConvLSTMForecaster',
    
    # Classical models
    'RandomForestForecaster',
    'LinearForecaster',
    'PolynomialForecaster',
    'GaussianForecaster',
    'SVRForecaster',
    
    # Neural models
    'ANNForecaster',
    'ResNetForecaster',
    'AutoencoderForecaster',
    'EnsembleNeuralForecaster',
    
    # Transformer models
    'TransformerForecaster',
    'TimesFMInspiredForecaster',
    
    # Challenge models
    'CityLearnChallengeForecaster'
]