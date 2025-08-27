"""
LSTM (Long Short-Term Memory) Models for CityLearn Challenge

This module implements LSTM-based time series forecasting models specifically
designed for building energy prediction. LSTM networks are particularly
well-suited for energy forecasting because they can capture:

1. Long-term dependencies in energy consumption patterns
2. Complex seasonal and weekly cycles  
3. Non-linear relationships between weather and energy demand
4. Memory of past consumption patterns for future prediction

LSTM Architecture Benefits for Energy Forecasting:
- Selective memory through gates (forget, input, output)
- Gradient flow preservation for long sequences
- Ability to learn complex temporal patterns
- Robust to varying sequence lengths and missing data
"""

import numpy as np
from typing import Tuple, Optional
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from .base_models import BaseForecaster




class LSTMForecaster(BaseForecaster):
    """
    LSTM neural network for building energy time series forecasting.
    
    This implementation provides a flexible LSTM architecture that can adapt
    to different building types, energy variables, and forecasting horizons.
    Key features:
    - Multi-layer LSTM architecture
    - Dropout regularization for overfitting prevention
    - Early stopping and learning rate reduction
    - Automatic input shape adaptation
    """

    def __init__(self, 
                 sequence_length: int = 24,
                 hidden_units: int = 50,
                 num_layers: int = 2,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Initialize LSTM forecaster with architecture parameters.
        
        Args:
            sequence_length: Number of historical time steps to use for prediction
            hidden_units: Number of LSTM units in each layer
            num_layers: Number of stacked LSTM layers (depth of network)
            dropout_rate: Fraction of units to drop for regularization
            learning_rate: Initial learning rate for optimizer
        """
        super().__init__("LSTM")
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Model will be built during fit() to adapt to input dimensions
        self.model = None
        
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Construct the LSTM neural network architecture.
        
        Creates a deep LSTM network with the specified number of layers and
        regularization. The architecture is:
        1. Input layer accepting sequences of shape (sequence_length, features)
        2. Multiple LSTM layers with dropout regularization
        3. Dense output layer for prediction
        
        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)
        """
        
        # Initialize sequential model (layers stacked in order)
        self.model = Sequential()
        
        # First LSTM layer - must specify input shape for Keras
        self.model.add(LSTM(
            self.hidden_units,
            return_sequences=True if self.num_layers > 1 else False,
            input_shape=input_shape,
            name=f"lstm_layer_1"
        ))
        self.model.add(Dropout(self.dropout_rate, name="dropout_1"))
        
        # Add additional LSTM layers if specified
        for i in range(1, self.num_layers):
            return_sequences = i < self.num_layers - 1
            
            self.model.add(LSTM(
                self.hidden_units, 
                return_sequences=return_sequences,
                name=f"lstm_layer_{i+1}"
            ))
            self.model.add(Dropout(self.dropout_rate, name=f"dropout_{i+1}"))
        
        # Output layer - Dense layer for final prediction
        self.model.add(Dense(1, name="prediction_output"))
        
        # Compile model with optimizer, loss function, and metrics
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), # type: ingnore # type: ignore
            loss='mse',
            metrics=['mae']
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: int = 0) -> None:
        """
        Train the LSTM model on building energy time series data.
        
        Args:
            X_train: Training sequences of shape (n_samples, sequence_length, n_features)
            y_train: Training targets of shape (n_samples, 1)
            X_val: Optional validation sequences
            y_val: Optional validation targets
            epochs: Maximum number of training epochs
            batch_size: Number of samples per batch
            verbose: Training verbosity level
        """
        # Ensure correct input shape for LSTM
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
            
        # Build model architecture based on input shape
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self._build_model(input_shape)
            
        assert self.model is not None
        
        # Setup callbacks for training optimization
        callbacks = [
            EarlyStopping(
                monitor='loss',
                patience=15,
                restore_best_weights=True,
                verbose=0  # Clean output
            ),
            ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=0  # Clean output
            )
        ]
        
        # Prepare validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            if len(X_val.shape) == 2:
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            if len(y_val.shape) == 1:
                y_val = y_val.reshape(-1, 1)
            validation_data = (X_val, y_val)
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose, # type: ignore
            shuffle=True
        )
        
        self.is_fitted = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for new input sequences.
        
        Args:
            X: Input sequences of shape (n_samples, sequence_length, n_features)
            
        Returns:
            Predictions of shape (n_samples, 1)
        """
        assert self.model is not None
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Ensure correct input shape
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        predictions = self.model.predict(X, verbose='0')
        return predictions.flatten() if len(predictions.shape) > 1 else predictions


class BidirectionalLSTMForecaster(BaseForecaster):
    """
    Bidirectional LSTM for capturing both forward and backward temporal patterns.
    
    Useful for energy forecasting when future context (e.g., weather forecasts,
    scheduled events) might be available and relevant for current predictions.
    """
    
    def __init__(self, 
                 sequence_length: int = 24,
                 hidden_units: int = 50,
                 num_layers: int = 2,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """Initialize Bidirectional LSTM forecaster."""
        super().__init__("Bidirectional_LSTM")
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """Build bidirectional LSTM architecture."""
        from keras.layers import Bidirectional
        
        self.model = Sequential()
        
        # First bidirectional LSTM layer
        self.model.add(Bidirectional(
            LSTM(self.hidden_units, 
                 return_sequences=True if self.num_layers > 1 else False),
            input_shape=input_shape,
            name="bidirectional_lstm_1"
        ))
        self.model.add(Dropout(self.dropout_rate))
        
        # Additional bidirectional layers
        for i in range(1, self.num_layers):
            return_sequences = i < self.num_layers - 1
            self.model.add(Bidirectional(
                LSTM(self.hidden_units, return_sequences=return_sequences),
                name=f"bidirectional_lstm_{i+1}"
            ))
            self.model.add(Dropout(self.dropout_rate))
        
        # Output layer
        self.model.add(Dense(1, name="prediction_output"))
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), # type: ignore
            loss='mse',
            metrics=['mae']
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: int = 0) -> None:
        """Train the bidirectional LSTM model."""
        # Same implementation as regular LSTM
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
            
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self._build_model(input_shape)
            
        assert self.model is not None
        
        callbacks = [
            EarlyStopping(monitor='loss', patience=15, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, min_lr=1e-7, verbose=0)
        ]
        
        validation_data = None
        if X_val is not None and y_val is not None:
            if len(X_val.shape) == 2:
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            if len(y_val.shape) == 1:
                y_val = y_val.reshape(-1, 1)
            validation_data = (X_val, y_val)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=str(verbose),
            shuffle=True
        )
        
        self.is_fitted = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using bidirectional LSTM."""
        assert self.model is not None
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        predictions = self.model.predict(X, verbose='0')
        return predictions.flatten() if len(predictions.shape) > 1 else predictions


class ConvLSTMForecaster(BaseForecaster):
    """
    Convolutional LSTM for capturing spatial-temporal patterns in building data.
    
    Useful when dealing with multiple buildings or spatial features like
    temperature distributions, solar irradiance patterns, etc.
    """
    
    def __init__(self, 
                 sequence_length: int = 24,
                 filters: int = 32,
                 kernel_size: int = 3,
                 lstm_units: int = 50,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """Initialize ConvLSTM forecaster."""
        super().__init__("ConvLSTM")
        self.sequence_length = sequence_length
        self.filters = filters
        self.kernel_size = kernel_size
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """Build CNN-LSTM hybrid architecture."""
        from keras.layers import Conv1D, MaxPooling1D, Flatten
        
        self.model = Sequential()
        
        # Convolutional layers for feature extraction
        self.model.add(Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation='relu',
            input_shape=input_shape,
            name="conv1d_1"
        ))
        self.model.add(MaxPooling1D(pool_size=2, name="maxpool1d_1"))
        self.model.add(Dropout(self.dropout_rate))
        
        # Additional conv layer
        self.model.add(Conv1D(
            filters=self.filters//2,
            kernel_size=self.kernel_size,
            activation='relu',
            name="conv1d_2"
        ))
        self.model.add(MaxPooling1D(pool_size=2, name="maxpool1d_2"))
        
        # LSTM layer for temporal modeling
        self.model.add(LSTM(
            self.lstm_units,
            return_sequences=False,
            name="lstm_layer"
        ))
        self.model.add(Dropout(self.dropout_rate))
        
        # Output layer
        self.model.add(Dense(1, name="prediction_output"))
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), # type: ignore
            loss='mse',
            metrics=['mae']
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: int = 0) -> None:
        """Train the ConvLSTM model."""
        # Same training logic as other LSTM variants
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
            
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self._build_model(input_shape)
            
        assert self.model is not None
        
        callbacks = [
            EarlyStopping(monitor='loss', patience=15, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, min_lr=1e-7, verbose=0)
        ]
        
        validation_data = None
        if X_val is not None and y_val is not None:
            if len(X_val.shape) == 2:
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            if len(y_val.shape) == 1:
                y_val = y_val.reshape(-1, 1)
            validation_data = (X_val, y_val)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=str(verbose),
            shuffle=True
        )
        
        self.is_fitted = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using ConvLSTM."""
        assert self.model is not None
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        predictions = self.model.predict(X, verbose='0')
        return predictions.flatten() if len(predictions.shape) > 1 else predictions