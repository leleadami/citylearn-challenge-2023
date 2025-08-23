"""
Neural Network Models for CityLearn Challenge

This module implements various neural network architectures for building
energy forecasting beyond LSTM. Neural networks excel at:

1. Non-linear pattern recognition in complex energy data
2. Automatic feature extraction from raw time series
3. Handling high-dimensional input spaces
4. Learning complex interactions between multiple variables

Neural models are particularly effective for:
- Complex multi-building energy patterns
- Weather-energy interaction modeling
- Peak demand prediction with multiple factors
- Transfer learning across different building types
"""

import numpy as np
from typing import Tuple, Optional, List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import warnings
warnings.filterwarnings('ignore')

from .base_models import BaseForecaster


class ANNForecaster(BaseForecaster):
    """
    Artificial Neural Network (Multi-Layer Perceptron) for energy forecasting.
    
    This feedforward neural network is designed for building energy prediction
    with the following architecture benefits:
    - Multiple hidden layers for complex pattern recognition
    - Dropout and batch normalization for regularization
    - Flexible architecture adaptation to different input sizes
    - Fast training compared to recurrent architectures
    
    ANNs are particularly effective when:
    - Historical patterns are more important than sequence order
    - Quick training and deployment is needed
    - Feature engineering has created informative inputs
    - Baseline neural network performance is desired
    """
    
    def __init__(self, 
                 hidden_layers: List[int] = [128, 64, 32],
                 activation: str = 'relu',
                 dropout_rate: float = 0.3,
                 batch_norm: bool = True,
                 l1_reg: float = 0.0,
                 l2_reg: float = 0.001,
                 learning_rate: float = 0.001):
        """
        Initialize ANN forecaster with flexible architecture.
        
        Args:
            hidden_layers: List of units in each hidden layer
            activation: Activation function for hidden layers
            dropout_rate: Dropout rate for regularization
            batch_norm: Whether to use batch normalization
            l1_reg: L1 regularization strength
            l2_reg: L2 regularization strength  
            learning_rate: Initial learning rate for optimizer
        """
        super().__init__("ANN")
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        
        self.model = None
        
    def _build_model(self, input_dim: int) -> None:
        """
        Build the ANN architecture.
        
        Creates a deep feedforward network with:
        1. Input layer matching feature dimensions
        2. Multiple hidden layers with activation, dropout, batch norm
        3. Output layer for regression
        
        Args:
            input_dim: Number of input features
        """
        self.model = Sequential()
        
        # Input layer - first hidden layer needs input dimension
        self.model.add(Dense(
            self.hidden_layers[0], 
            input_dim=input_dim,
            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name="dense_input"
        ))
        
        # Add batch normalization if specified
        if self.batch_norm:
            self.model.add(BatchNormalization(name="batch_norm_input"))
            
        self.model.add(Activation(self.activation, name="activation_input"))
        self.model.add(Dropout(self.dropout_rate, name="dropout_input"))
        
        # Hidden layers
        for i, units in enumerate(self.hidden_layers[1:], 1):
            self.model.add(Dense(
                units,
                kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name=f"dense_hidden_{i}"
            ))
            
            if self.batch_norm:
                self.model.add(BatchNormalization(name=f"batch_norm_hidden_{i}"))
                
            self.model.add(Activation(self.activation, name=f"activation_hidden_{i}"))
            self.model.add(Dropout(self.dropout_rate, name=f"dropout_hidden_{i}"))
        
        # Output layer - single unit for regression
        self.model.add(Dense(1, name="output"))
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
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
        Train the ANN model on energy data.
        
        Args:
            X_train: Training features of shape (n_samples, n_features)
            y_train: Training targets of shape (n_samples,) or (n_samples, 1)
            X_val: Optional validation features
            y_val: Optional validation targets
            epochs: Maximum training epochs
            batch_size: Batch size for training
            verbose: Training verbosity level
        """
        # Flatten inputs for dense layers
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        y_train_flat = y_train.flatten() if len(y_train.shape) > 1 else y_train
        
        # Build model if not already built
        if self.model is None:
            self._build_model(X_train_flat.shape[1])
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='loss',
                patience=15,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=verbose
            )
        ]
        
        # Prepare validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            y_val_flat = y_val.flatten() if len(y_val.shape) > 1 else y_val
            validation_data = (X_val_flat, y_val_flat)
        
        # Train the model
        history = self.model.fit(
            X_train_flat, y_train_flat,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        self.is_fitted = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ANN predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_flat = X.reshape(X.shape[0], -1)
        predictions = self.model.predict(X_flat, verbose=0)
        return predictions.flatten()


class ResNetForecaster(BaseForecaster):
    """
    Residual Neural Network for deep energy forecasting.
    
    ResNet architecture with skip connections enables training of very deep
    networks by solving the vanishing gradient problem. Benefits include:
    - Deeper networks without degradation
    - Better gradient flow during backpropagation
    - Improved representation learning for complex energy patterns
    - State-of-the-art performance on many forecasting tasks
    """
    
    def __init__(self,
                 residual_blocks: int = 3,
                 block_size: int = 64,
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001):
        """Initialize ResNet forecaster."""
        super().__init__("ResNet")
        self.residual_blocks = residual_blocks
        self.block_size = block_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        
    def _residual_block(self, x, block_id: int):
        """Create a residual block with skip connection."""
        from tensorflow.keras.layers import Add
        
        # Store input for skip connection
        shortcut = x
        
        # First dense layer
        x = Dense(
            self.block_size, 
            activation='relu',
            name=f"res_block_{block_id}_dense_1"
        )(x)
        x = Dropout(self.dropout_rate, name=f"res_block_{block_id}_dropout_1")(x)
        
        # Second dense layer
        x = Dense(
            self.block_size,
            name=f"res_block_{block_id}_dense_2"
        )(x)
        
        # Skip connection - add input to output
        x = Add(name=f"res_block_{block_id}_add")([shortcut, x])
        x = Activation('relu', name=f"res_block_{block_id}_relu")(x)
        x = Dropout(self.dropout_rate, name=f"res_block_{block_id}_dropout_2")(x)
        
        return x
    
    def _build_model(self, input_dim: int) -> None:
        """Build ResNet architecture with skip connections."""
        from tensorflow.keras.layers import Input
        from tensorflow.keras.models import Model
        
        # Input layer
        inputs = Input(shape=(input_dim,), name="input")
        
        # Initial dense layer to match residual block size
        x = Dense(self.block_size, activation='relu', name="initial_dense")(inputs)
        x = Dropout(self.dropout_rate, name="initial_dropout")(x)
        
        # Stack residual blocks
        for i in range(self.residual_blocks):
            x = self._residual_block(x, i)
        
        # Output layer
        outputs = Dense(1, name="output")(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name="ResNet_Forecaster")
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: int = 0) -> None:
        """Train ResNet model."""
        # Same training logic as ANN
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        y_train_flat = y_train.flatten() if len(y_train.shape) > 1 else y_train
        
        if self.model is None:
            self._build_model(X_train_flat.shape[1])
        
        callbacks = [
            EarlyStopping(monitor='loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, min_lr=1e-7)
        ]
        
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            y_val_flat = y_val.flatten() if len(y_val.shape) > 1 else y_val
            validation_data = (X_val_flat, y_val_flat)
        
        history = self.model.fit(
            X_train_flat, y_train_flat,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        self.is_fitted = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ResNet predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_flat = X.reshape(X.shape[0], -1)
        predictions = self.model.predict(X_flat, verbose=0)
        return predictions.flatten()


class AutoencoderForecaster(BaseForecaster):
    """
    Autoencoder-based forecasting for dimensionality reduction and denoising.
    
    Uses an encoder-decoder architecture to:
    1. Learn compressed representations of energy patterns
    2. Remove noise from input data
    3. Capture the most important features for prediction
    4. Enable transfer learning across buildings
    """
    
    def __init__(self,
                 encoding_dims: List[int] = [64, 32, 16],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """Initialize autoencoder forecaster."""
        super().__init__("Autoencoder")
        self.encoding_dims = encoding_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        
    def _build_model(self, input_dim: int) -> None:
        """Build autoencoder architecture."""
        from tensorflow.keras.layers import Input
        from tensorflow.keras.models import Model
        
        # Input layer
        inputs = Input(shape=(input_dim,), name="input")
        
        # Encoder
        encoded = inputs
        for i, dim in enumerate(self.encoding_dims):
            encoded = Dense(dim, activation='relu', name=f"encoder_{i}")(encoded)
            encoded = Dropout(self.dropout_rate, name=f"encoder_dropout_{i}")(encoded)
        
        # Decoder (mirror of encoder)
        decoded = encoded
        for i, dim in enumerate(reversed(self.encoding_dims[:-1])):
            decoded = Dense(dim, activation='relu', name=f"decoder_{i}")(decoded)
            decoded = Dropout(self.dropout_rate, name=f"decoder_dropout_{i}")(decoded)
        
        # Final prediction layer
        outputs = Dense(1, name="output")(decoded)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name="Autoencoder_Forecaster")
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: int = 0) -> None:
        """Train autoencoder model."""
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        y_train_flat = y_train.flatten() if len(y_train.shape) > 1 else y_train
        
        if self.model is None:
            self._build_model(X_train_flat.shape[1])
        
        callbacks = [
            EarlyStopping(monitor='loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            y_val_flat = y_val.flatten() if len(y_val.shape) > 1 else y_val
            validation_data = (X_val_flat, y_val_flat)
        
        history = self.model.fit(
            X_train_flat, y_train_flat,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        self.is_fitted = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate autoencoder predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_flat = X.reshape(X.shape[0], -1)
        predictions = self.model.predict(X_flat, verbose=0)
        return predictions.flatten()
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Get encoded representations from the autoencoder."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before encoding")
        
        # Create encoder model (input to encoded layer)
        from tensorflow.keras.models import Model
        encoder_output = self.model.get_layer(f"encoder_{len(self.encoding_dims)-1}").output
        encoder = Model(self.model.input, encoder_output)
        
        X_flat = X.reshape(X.shape[0], -1)
        encoded = encoder.predict(X_flat, verbose=0)
        return encoded


class EnsembleNeuralForecaster(BaseForecaster):
    """
    Ensemble of neural network architectures for robust forecasting.
    
    Combines multiple neural network types to leverage their different strengths:
    - ANN for general non-linear patterns
    - ResNet for deep feature learning
    - Autoencoder for noise reduction
    
    Ensemble benefits:
    - Improved generalization through diversity
    - Reduced overfitting risk
    - Better handling of different energy patterns
    - More robust predictions across building types
    """
    
    def __init__(self,
                 include_ann: bool = True,
                 include_resnet: bool = True,
                 include_autoencoder: bool = True,
                 ensemble_method: str = 'average'):
        """Initialize neural ensemble forecaster."""
        super().__init__("Neural_Ensemble")
        self.include_ann = include_ann
        self.include_resnet = include_resnet
        self.include_autoencoder = include_autoencoder
        self.ensemble_method = ensemble_method
        
        self.models = {}
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 50,
            batch_size: int = 32,
            verbose: int = 0) -> None:
        """Train all models in the ensemble."""
        if self.include_ann:
            print("Training ANN...")
            self.models['ann'] = ANNForecaster()
            self.models['ann'].fit(X_train, y_train, X_val, y_val, epochs, batch_size, verbose)
        
        if self.include_resnet:
            print("Training ResNet...")
            self.models['resnet'] = ResNetForecaster()
            self.models['resnet'].fit(X_train, y_train, X_val, y_val, epochs, batch_size, verbose)
            
        if self.include_autoencoder:
            print("Training Autoencoder...")
            self.models['autoencoder'] = AutoencoderForecaster()
            self.models['autoencoder'].fit(X_train, y_train, X_val, y_val, epochs, batch_size, verbose)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = []
        
        for model_name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
        
        # Combine predictions
        if self.ensemble_method == 'average':
            ensemble_pred = np.mean(predictions, axis=0)
        elif self.ensemble_method == 'median':
            ensemble_pred = np.median(predictions, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return ensemble_pred