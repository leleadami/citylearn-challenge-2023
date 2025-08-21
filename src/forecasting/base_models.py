"""
Base forecasting models for time series prediction.
Includes LSTM, ANN, Gaussian regression, Linear regression, Random Forest, and Polynomial fitting.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from typing import Dict, Tuple, Optional
import joblib
import os


class BaseForecaster:
    """
    Base class for all forecasting models in the CityLearn Challenge.
    
    This abstract class defines the common interface that all forecasting models
    must implement. It provides standardized methods for training, prediction,
    and model persistence across different algorithm types.
    
    The class supports various time series forecasting approaches including:
    - Deep learning models (LSTM, ANN)
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
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the forecasting model on historical time series data.
        
        This method must be implemented by each specific forecaster to handle
        the training process appropriate for that algorithm type.
        
        Args:
            X_train (np.ndarray): Input sequences of shape (n_samples, sequence_length, n_features)
                                 Each sample contains historical values used for prediction
            y_train (np.ndarray): Target values of shape (n_samples, prediction_horizon)
                                 Future values to be predicted
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement the fit method")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate forecasts for new input sequences.
        
        Uses the trained model to predict future values based on input sequences.
        The model must be fitted before calling this method.
        
        Args:
            X (np.ndarray): Input sequences of shape (n_samples, sequence_length, n_features)
                           Historical data for which to generate forecasts
        
        Returns:
            np.ndarray: Predicted values of shape (n_samples, prediction_horizon)
                       Future forecasted values
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement the predict method")
        
    def save_model(self, filepath: str) -> None:
        """
        Persist the trained model to disk for later use.
        
        Saves model parameters and architecture to allow loading and inference
        without retraining. Implementation varies by model type.
        
        Args:
            filepath (str): Path where the model should be saved
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement the save_model method")
        
    def load_model(self, filepath: str) -> None:
        """
        Load a previously trained model from disk.
        
        Restores model parameters and sets is_fitted flag to enable predictions
        without retraining.
        
        Args:
            filepath (str): Path to the saved model file
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement the load_model method")


class LSTMForecaster(BaseForecaster):
    """LSTM neural network for time series forecasting."""
    
    def __init__(self, 
                 sequence_length: int = 24,
                 hidden_units: int = 50,
                 num_layers: int = 2,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        super().__init__("LSTM")
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.num_layers = num_layers             # Depth of LSTM stack
        self.dropout_rate = dropout_rate         # Regularization strength
        self.learning_rate = learning_rate       # Optimizer step size
        
        # Model will be built during fit() to adapt to input dimensions
        self.model = None                        # TensorFlow/Keras Sequential model
        
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Construct the LSTM neural network architecture.
        
        Creates a deep LSTM network with the specified number of layers and
        regularization. The architecture is:
        1. Input layer accepting sequences of shape (sequence_length, features)
        2. Multiple LSTM layers with dropout regularization
        3. Dense output layer for prediction
        
        Args:
            input_shape (Tuple[int, int]): Shape of input sequences (sequence_length, n_features)
                                          Automatically determined from training data
        """
        # Initialize sequential model (layers stacked in order)
        self.model = Sequential()
        
        # First LSTM layer - must specify input shape for Keras
        # return_sequences=True if we have more layers after this one
        self.model.add(LSTM(
            self.hidden_units,                                    # Number of LSTM units
            return_sequences=True if self.num_layers > 1 else False,  # Return full sequence or just last output
            input_shape=input_shape,                              # Expected input dimensions
            name=f"lstm_layer_1"                                  # Layer name for debugging
        ))
        # Add dropout after first LSTM to prevent overfitting
        self.model.add(Dropout(self.dropout_rate, name="dropout_1"))
        
        # Add additional LSTM layers if specified (for deeper networks)
        for i in range(1, self.num_layers):
            # Only return sequences if this isn't the last LSTM layer
            return_sequences = i < self.num_layers - 1
            
            self.model.add(LSTM(
                self.hidden_units, 
                return_sequences=return_sequences,
                name=f"lstm_layer_{i+1}"
            ))
            # Add dropout after each LSTM layer
            self.model.add(Dropout(self.dropout_rate, name=f"dropout_{i+1}"))
        
        # Output layer - Dense layer for final prediction
        # Single output unit for scalar prediction (e.g., energy consumption)
        self.model.add(Dense(1, name="prediction_output"))
        
        # Compile model with optimizer, loss function, and metrics
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),  # Adam: adaptive learning rate optimizer
            loss='mse',                                         # Mean Squared Error for regression
            metrics=['mae']                                     # Mean Absolute Error for monitoring
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: int = 0) -> None:
        """
        Train the LSTM model on building energy time series data.
        
        LSTMs are particularly well-suited for building energy forecasting because:
        1. **Temporal Dependencies**: Building energy consumption has strong temporal patterns
           (daily cycles, weekly patterns, seasonal variations) that LSTMs can capture
        2. **Long Memory**: LSTMs can remember patterns from hours or days ago, crucial for
           predicting energy usage based on historical consumption patterns
        3. **Non-linear Relationships**: Energy consumption has complex non-linear relationships
           with factors like weather, occupancy, and building operations
        4. **Variable-length Dependencies**: Different buildings may have different lag periods
           for energy patterns, which LSTMs can adapt to during training
        
        Training Process:
        - Uses backpropagation through time (BPTT) to learn temporal patterns
        - Adam optimizer adapts learning rate for each parameter individually
        - Validation data enables early stopping and overfitting prevention
        
        Args:
            X_train (np.ndarray): Input sequences of shape (n_samples, sequence_length, n_features)
                                 Each sequence represents historical energy data
            y_train (np.ndarray): Target values of shape (n_samples, 1) or (n_samples, horizon)
                                 Future energy values to predict
            X_val (np.ndarray, optional): Validation input sequences for monitoring overfitting
            y_val (np.ndarray, optional): Validation targets for performance monitoring
            epochs (int): Maximum number of training iterations through the dataset
                         More epochs allow learning complex patterns but risk overfitting
            batch_size (int): Number of sequences processed simultaneously
                             Smaller batches provide more gradient updates but are noisier
            verbose (int): Training output verbosity (0=silent, 1=progress bar, 2=one line per epoch)
        """
        # Build model architecture if not already constructed
        # Model building is deferred until fit() to automatically adapt to input dimensions
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, n_features)
            self._build_model(input_shape)
        
        # Prepare validation data for monitoring training progress
        # Validation helps detect overfitting and enables early stopping
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train the LSTM using gradient descent optimization
        # The model learns to map input sequences to target values
        self.model.fit(
            X_train, y_train,
            validation_data=validation_data,  # Monitor performance on unseen data
            epochs=epochs,                    # Number of complete passes through training data
            batch_size=batch_size,            # Mini-batch size for stochastic gradient descent
            verbose=verbose                   # Control training output verbosity
        )
        
        # Mark model as trained and ready for predictions
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate energy consumption forecasts using the trained LSTM model.
        
        The LSTM uses its learned internal state representations to predict future
        energy consumption based on historical patterns. The prediction process:
        
        1. **Forward Pass**: Input sequences flow through LSTM layers
        2. **State Evolution**: Hidden states capture temporal dependencies
        3. **Memory Integration**: Cell states maintain long-term patterns
        4. **Output Generation**: Final dense layer produces energy predictions
        
        Why LSTM predictions work well for energy forecasting:
        - Captures daily/weekly cycles in building energy consumption
        - Remembers seasonal patterns and weather-dependent variations
        - Adapts to building-specific usage patterns and occupancy schedules
        - Handles irregular events and operational changes
        
        Args:
            X (np.ndarray): Input sequences of shape (n_samples, sequence_length, n_features)
                           Historical energy data for generating forecasts
        
        Returns:
            np.ndarray: Predicted energy values of shape (n_samples, 1) or (n_samples, horizon)
                       Future energy consumption forecasts
        
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Generate predictions using the trained LSTM
        # The model applies learned patterns to new input sequences
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """
        Persist the trained LSTM model to disk for future use.
        
        Saves the complete model including:
        - Network architecture (layer structure, connections)
        - Learned weights and biases from training
        - Optimizer state (for resuming training if needed)
        - Model configuration and hyperparameters
        
        This is crucial for production energy forecasting systems where
        models need to be deployed and reused without retraining.
        
        Args:
            filepath (str): Path where the model should be saved (typically .h5 or .keras format)
        """
        if self.model is not None:
            self.model.save(filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a previously trained LSTM model from disk.
        
        Restores the complete model state including learned parameters,
        enabling immediate predictions without retraining. Essential for
        deploying energy forecasting models in production environments.
        
        Args:
            filepath (str): Path to the saved model file
        """
        self.model = tf.keras.models.load_model(filepath)
        self.is_fitted = True


class ANNForecaster(BaseForecaster):
    """
    Artificial Neural Network (Multi-Layer Perceptron) for time series forecasting.
    
    ANNs are excellent for building energy forecasting because they can:
    1. **Universal Approximation**: Can approximate any continuous function, making them
       suitable for complex energy consumption patterns
    2. **Non-linear Mapping**: Capture non-linear relationships between weather, occupancy,
       time of day, and energy consumption
    3. **Feature Interaction**: Learn interactions between multiple input variables
       (temperature, humidity, day of week, etc.)
    4. **Fast Inference**: Once trained, predictions are very fast, suitable for real-time
       energy management systems
    
    Architecture Design Rationale:
    - Multiple hidden layers enable hierarchical feature learning
    - Decreasing layer sizes (64→32) create a funnel effect for information compression
    - ReLU activation prevents vanishing gradients and enables efficient training
    - Dropout regularization prevents overfitting to specific energy patterns
    
    Mathematical Foundation:
    For each layer: output = ReLU(W × input + b)
    Where W are learned weights and b are learned biases
    Final layer uses linear activation for regression output
    """
    
    def __init__(self,
                 hidden_layers: list = [64, 32],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Initialize ANN forecaster with architecture parameters.
        
        Args:
            hidden_layers (list): Sizes of hidden layers [64, 32] creates two hidden layers
                                 First layer: 64 neurons (captures complex patterns)
                                 Second layer: 32 neurons (refines and compresses features)
            dropout_rate (float): Fraction of neurons to randomly disable during training
                                 0.2 means 20% dropout - prevents overfitting to training data
            learning_rate (float): Step size for gradient descent optimization
                                 0.001 is conservative, ensuring stable convergence
        """
        super().__init__("ANN")
        self.hidden_layers = hidden_layers    # Architecture specification
        self.dropout_rate = dropout_rate      # Regularization strength
        self.learning_rate = learning_rate    # Optimization step size
    
    def _build_model(self, input_dim: int) -> None:
        """
        Construct the ANN architecture optimized for energy forecasting.
        
        Network Design Philosophy:
        1. **Input Layer**: Accepts flattened time series sequences
        2. **Hidden Layers**: Progressively smaller layers create feature hierarchy
        3. **Activation Functions**: ReLU enables non-linear transformations
        4. **Regularization**: Dropout prevents memorization of training patterns
        5. **Output Layer**: Single neuron for scalar energy prediction
        
        Why this architecture works for energy forecasting:
        - First layer (64 units): Captures diverse temporal and seasonal patterns
        - Second layer (32 units): Combines patterns into higher-level features
        - Dropout: Ensures model generalizes to unseen energy consumption patterns
        - Linear output: Appropriate for continuous energy values (kWh)
        
        Args:
            input_dim (int): Flattened input dimension (sequence_length × n_features)
                           For 24-hour sequences with 1 feature: input_dim = 24
        """
        # Initialize sequential model (feed-forward architecture)
        self.model = Sequential()
        
        # First hidden layer - primary pattern detection
        # ReLU activation: f(x) = max(0, x) - prevents vanishing gradients
        self.model.add(Dense(
            self.hidden_layers[0],           # 64 neurons for rich representation
            activation='relu',               # Non-linear activation for complex patterns
            input_dim=input_dim,            # Dimension of flattened input sequences
            name="primary_hidden_layer"
        ))
        # Dropout regularization - randomly disable 20% of neurons during training
        self.model.add(Dropout(self.dropout_rate, name="primary_dropout"))
        
        # Additional hidden layers - hierarchical feature learning
        for i, units in enumerate(self.hidden_layers[1:]):
            self.model.add(Dense(
                units,                       # Decreasing size creates information funnel
                activation='relu',           # Consistent non-linear activation
                name=f"hidden_layer_{i+2}"
            ))
            # Apply dropout after each hidden layer
            self.model.add(Dropout(self.dropout_rate, name=f"dropout_{i+2}"))
        
        # Output layer - energy consumption prediction
        # Linear activation (default) for continuous regression output
        self.model.add(Dense(1, name="energy_prediction_output"))
        
        # Compile model with optimization configuration
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),  # Adaptive learning rate optimizer
            loss='mse',                                         # Mean Squared Error for regression
            metrics=['mae']                                     # Mean Absolute Error for monitoring
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: int = 0) -> None:
        """
        Train the ANN on building energy consumption data.
        
        ANNs excel at energy forecasting because they can:
        1. **Learn Complex Patterns**: Capture non-linear relationships between
           multiple factors affecting energy consumption
        2. **Fast Training**: Converge quickly compared to RNNs, suitable for
           frequent model updates with new energy data
        3. **Robust to Noise**: Multiple layers provide built-in filtering of
           measurement noise common in building sensor data
        4. **Scalable**: Can handle many input features (weather, occupancy, schedules)
        
        Data Preprocessing for ANNs:
        - Flattens time series sequences into feature vectors
        - Treats temporal information as spatial patterns
        - Works well when temporal dependencies are not long-range
        
        Training Process:
        - Backpropagation learns optimal weights for energy prediction
        - Adam optimizer adapts learning rate based on gradient statistics
        - Dropout prevents overfitting to specific building patterns
        
        Args:
            X_train (np.ndarray): Input sequences (flattened for ANN processing)
            y_train (np.ndarray): Target energy values
            X_val (np.ndarray, optional): Validation sequences
            y_val (np.ndarray, optional): Validation targets
            epochs (int): Training iterations (100 usually sufficient for energy data)
            batch_size (int): Samples per gradient update (32 balances speed and stability)
            verbose (int): Training progress display level
        """
        # Reshape sequential data for ANN processing
        # ANNs expect 2D input: (n_samples, n_features) rather than 3D sequences
        if len(X_train.shape) > 2:
            # Flatten sequence dimension: (samples, seq_len, features) → (samples, seq_len*features)
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_flat = X_train
            
        # Ensure target is 1D for regression
        if len(y_train.shape) > 1:
            y_train_flat = y_train.flatten()
        else:
            y_train_flat = y_train
        
        # Build network architecture if not already constructed
        if self.model is None:
            self._build_model(X_train_flat.shape[1])  # Input dimension = flattened sequence length
        
        # Prepare validation data for overfitting monitoring
        validation_data = None
        if X_val is not None and y_val is not None:
            # Apply same flattening to validation data
            X_val_flat = X_val.reshape(X_val.shape[0], -1) if len(X_val.shape) > 2 else X_val
            y_val_flat = y_val.flatten() if len(y_val.shape) > 1 else y_val
            validation_data = (X_val_flat, y_val_flat)
        
        # Train the neural network using gradient descent
        self.model.fit(
            X_train_flat, y_train_flat,      # Training data (flattened)
            validation_data=validation_data,  # Monitor generalization performance
            epochs=epochs,                    # Number of complete training passes
            batch_size=batch_size,            # Mini-batch size for SGD
            verbose=verbose                   # Training progress output
        )
        
        # Mark model as trained and ready for energy predictions
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate energy consumption forecasts using the trained ANN.
        
        ANN Prediction Process:
        1. **Input Processing**: Flatten temporal sequences into feature vectors
        2. **Forward Propagation**: Pass through hidden layers with learned weights
        3. **Feature Transformation**: Each layer extracts higher-level patterns
        4. **Output Generation**: Final layer produces energy consumption prediction
        
        Advantages for Energy Forecasting:
        - Very fast inference (milliseconds) suitable for real-time systems
        - Stable predictions not affected by input sequence variations
        - Can handle missing values through learned feature representations
        - Scales well to multiple buildings or prediction targets
        
        Args:
            X (np.ndarray): Input sequences to forecast (will be flattened)
        
        Returns:
            np.ndarray: Predicted energy consumption values
        
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Flatten sequential input for ANN processing
        # Convert (samples, seq_len, features) → (samples, seq_len*features)
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
        
        # Generate energy consumption predictions
        return self.model.predict(X_flat)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained ANN model for deployment in energy management systems.
        
        Saves complete model state including learned weights that encode
        building-specific energy consumption patterns.
        
        Args:
            filepath (str): Path to save the model file
        """
        if self.model is not None:
            self.model.save(filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a pre-trained ANN model for immediate energy forecasting.
        
        Enables deployment of trained models without retraining,
        essential for production energy management systems.
        
        Args:
            filepath (str): Path to the saved model file
        """
        self.model = tf.keras.models.load_model(filepath)
        self.is_fitted = True


class GaussianForecaster(BaseForecaster):
    """
    Gaussian Process Regressor for probabilistic time series forecasting.
    
    Gaussian Processes (GPs) are exceptionally valuable for building energy forecasting because:
    
    1. **Uncertainty Quantification**: GPs provide confidence intervals with predictions,
       crucial for energy management decisions and risk assessment
    2. **Non-parametric Flexibility**: Can model complex energy consumption patterns
       without assuming specific functional forms
    3. **Automatic Complexity Control**: The Bayesian framework prevents overfitting
       through built-in regularization
    4. **Small Data Performance**: Work well with limited training data, common in
       new building deployments or seasonal data
    5. **Smoothness Assumptions**: Energy consumption is typically smooth over time,
       which GPs naturally encode through kernel functions
    
    Mathematical Foundation:
    GP defines a distribution over functions: f(x) ~ GP(m(x), k(x,x'))
    - m(x): mean function (often assumed zero)
    - k(x,x'): covariance/kernel function encoding smoothness assumptions
    
    For energy forecasting, the RBF kernel captures:
    - Temporal smoothness: nearby time points have similar energy consumption
    - Periodic patterns: daily/weekly cycles through kernel combinations
    - Noise handling: alpha parameter accounts for measurement uncertainty
    
    Kernel Design Rationale:
    - Constant kernel: Models overall energy consumption level
    - RBF kernel: Captures smooth temporal variations and local patterns
    - Hyperparameter bounds: Prevent degenerate solutions while allowing flexibility
    """
    
    def __init__(self, kernel=None, alpha=1e-10):
        """
        Initialize Gaussian Process forecaster with kernel specification.
        
        Args:
            kernel: Covariance function defining smoothness assumptions
                   Default: ConstantKernel * RBF - captures energy baseline + smooth variations
            alpha (float): Noise variance parameter (1e-10)
                          Small value assumes accurate energy measurements
                          Larger values account for sensor noise and model uncertainty
        
        Kernel Components:
        - ConstantKernel(1.0, (1e-3, 1e3)): Models constant energy baseline
          * Initial value: 1.0 (neutral starting point)
          * Bounds: (1e-3, 1e3) allow wide range of energy scales
        - RBF(1.0, (1e-2, 1e2)): Radial Basis Function for smooth patterns
          * Length scale: 1.0 (controls how quickly correlations decay)
          * Bounds: (1e-2, 1e2) from very local to very global smoothness
        """
        super().__init__("Gaussian")
        
        # Define default kernel if none provided
        if kernel is None:
            # Composite kernel for energy forecasting:
            # 1. Constant kernel: captures overall energy consumption level
            # 2. RBF kernel: models smooth temporal variations
            kernel = (
                C(1.0, (1e-3, 1e3)) *    # Constant: energy baseline (wide range for different buildings)
                RBF(1.0, (1e-2, 1e2))    # RBF: smooth patterns (flexible length scale)
            )
        
        # Initialize Gaussian Process Regressor
        self.model = GaussianProcessRegressor(
            kernel=kernel,    # Covariance function encoding energy consumption assumptions
            alpha=alpha       # Noise variance (small for accurate energy meters)
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the Gaussian Process on building energy consumption data.
        
        GP Training Process (Bayesian Learning):
        1. **Kernel Hyperparameter Optimization**: Uses marginal likelihood maximization
           to find optimal kernel parameters for the energy data
        2. **Covariance Matrix Construction**: Builds K(X,X) encoding correlations
           between all training time points
        3. **Cholesky Decomposition**: Efficiently computes matrix inverse for predictions
        4. **Noise Estimation**: Learns appropriate noise level for energy measurements
        
        Why GPs Excel for Energy Forecasting:
        - **Adaptive Smoothness**: Automatically learns appropriate smoothness level
           for different buildings and time periods
        - **Non-linear Patterns**: Can capture complex relationships without manual
           feature engineering (weekday vs weekend, seasonal effects)
        - **Uncertainty Awareness**: Provides confidence bounds, crucial for energy
           procurement and grid management decisions
        - **Interpolation**: Excellent at filling gaps in energy consumption data
        
        Computational Complexity: O(n³) for training, where n is number of samples
        - Suitable for moderate datasets (hundreds to thousands of samples)
        - For larger datasets, consider sparse GP approximations
        
        Args:
            X_train (np.ndarray): Input sequences (flattened for GP)
                                 Each row represents a time point or sequence
            y_train (np.ndarray): Target energy values
                                 Scalar energy consumption for each time point
        """
        # Prepare data for Gaussian Process (requires 2D input)
        # Flatten sequential data: (samples, seq_len, features) → (samples, seq_len*features)
        X_train_flat = X_train.reshape(X_train.shape[0], -1) if len(X_train.shape) > 2 else X_train
        y_train_flat = y_train.flatten() if len(y_train.shape) > 1 else y_train
        
        # Fit GP using maximum likelihood estimation of hyperparameters
        # This optimizes kernel parameters to best explain the energy consumption patterns
        self.model.fit(X_train_flat, y_train_flat)
        
        # Store training completion flag
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate probabilistic energy consumption forecasts with uncertainty estimates.
        
        GP Prediction Process:
        1. **Posterior Computation**: Uses training data to compute posterior distribution
           over functions consistent with observed energy consumption
        2. **Predictive Distribution**: For each test point, computes Gaussian distribution
           over possible energy values
        3. **Mean Prediction**: Returns most likely energy consumption value
        4. **Uncertainty Quantification**: Can also return confidence intervals
           (use predict with return_std=True for uncertainty)
        
        Mathematical Foundation:
        For test input x*, predictive distribution is:
        p(f*|x*, X, y) = N(μ*, σ*²)
        where:
        - μ* = k*ᵀ(K + σ²I)⁻¹y (predictive mean)
        - σ*² = k** - k*ᵀ(K + σ²I)⁻¹k* (predictive variance)
        
        Advantages for Energy Management:
        - **Risk Assessment**: Uncertainty bounds help evaluate forecast reliability
        - **Decision Making**: Confidence intervals inform energy procurement strategies
        - **Outlier Detection**: High uncertainty indicates unusual consumption patterns
        - **Model Validation**: Calibrated uncertainties enable proper model evaluation
        
        Args:
            X (np.ndarray): Input sequences for prediction (will be flattened)
        
        Returns:
            np.ndarray: Predicted energy consumption values (mean of predictive distribution)
        
        Note: For uncertainty estimates, use self.model.predict(X, return_std=True)
              This returns both mean predictions and standard deviations
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Flatten input sequences for GP processing
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
        
        # Generate probabilistic predictions (returns mean of predictive distribution)
        return self.model.predict(X_flat)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained Gaussian Process model including learned hyperparameters.
        
        Saves:
        - Optimized kernel hyperparameters (length scales, noise variance)
        - Training data (required for GP predictions)
        - Precomputed covariance matrices (for efficient inference)
        
        Essential for deploying probabilistic energy forecasting in production
        where uncertainty quantification is crucial for grid management.
        
        Args:
            filepath (str): Path to save the model file
        """
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained Gaussian Process model for probabilistic energy forecasting.
        
        Restores complete GP state including optimized hyperparameters
        and training data, enabling immediate probabilistic predictions.
        
        Args:
            filepath (str): Path to the saved model file
        """
        self.model = joblib.load(filepath)
        self.is_fitted = True


class LinearForecaster(BaseForecaster):
    """
    Linear Regression for time series forecasting with interpretable coefficients.
    
    Linear regression serves as an essential baseline for energy forecasting because:
    
    1. **Interpretability**: Coefficients directly show how each input feature
       (temperature, time of day, etc.) influences energy consumption
    2. **Fast Training and Inference**: Closed-form solution via normal equations
       enables rapid model updates with new energy data
    3. **Stable Predictions**: Linear models are robust and don't suffer from
       gradient vanishing or local minima issues
    4. **Baseline Performance**: Provides minimum acceptable performance threshold
       for more complex models to beat
    5. **Feature Importance**: Reveals which factors most strongly predict energy use
    
    Mathematical Foundation:
    Assumes linear relationship: E(y|x) = β₀ + β₁x₁ + ... + βₙxₙ
    Where:
    - y: energy consumption
    - xᵢ: input features (past energy values, weather, time)
    - βᵢ: learned coefficients showing feature impact
    
    Solution via least squares: β = (XᵀX)⁻¹Xᵀy
    
    Limitations for Energy Forecasting:
    - Cannot capture non-linear relationships (e.g., HVAC efficiency curves)
    - Assumes constant feature effects across all conditions
    - May struggle with complex temporal patterns
    
    When Linear Models Work Well:
    - Well-conditioned buildings with predictable energy patterns
    - Simple HVAC systems with linear response to weather
    - Short-term forecasting where linear approximation is valid
    - Situations requiring explainable predictions
    """
    
    def __init__(self):
        """
        Initialize Linear Regression forecaster.
        
        Uses scikit-learn's LinearRegression with default parameters:
        - fit_intercept=True: Learns bias term for energy baseline
        - normalize=False: Assumes input features are pre-scaled
        - copy_X=True: Preserves original training data
        - n_jobs=None: Single-threaded (fast enough for linear regression)
        """
        super().__init__("Linear")
        # Linear regression model for energy consumption prediction
        self.model = LinearRegression()
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train linear regression model on energy consumption data.
        
        Training Process:
        1. **Normal Equations**: Solves analytically for optimal coefficients
           β = (XᵀX)⁻¹Xᵀy - no iterative optimization needed
        2. **Feature Relationships**: Learns linear weights for each input feature
        3. **Bias Term**: Automatically includes intercept for energy baseline
        
        Advantages for Energy Forecasting:
        - **Instant Training**: No epochs or convergence concerns
        - **Deterministic**: Same training data always produces identical model
        - **No Hyperparameters**: No learning rates, architectures, or regularization to tune
        - **Memory Efficient**: Stores only coefficient vector, not training data
        
        Energy Domain Insights:
        - Positive coefficients indicate energy-increasing factors (cooling degree days)
        - Negative coefficients show energy-reducing factors (natural lighting)
        - Magnitude shows relative importance of different factors
        
        Args:
            X_train (np.ndarray): Input features (flattened time series or engineered features)
                                 Each column represents a predictor variable
            y_train (np.ndarray): Target energy consumption values
                                 Continuous values in kWh or similar units
        """
        # Flatten sequential data for linear regression (requires 2D input)
        X_train_flat = X_train.reshape(X_train.shape[0], -1) if len(X_train.shape) > 2 else X_train
        y_train_flat = y_train.flatten() if len(y_train.shape) > 1 else y_train
        
        # Fit linear model using least squares solution
        # Computes: β = (XᵀX)⁻¹Xᵀy efficiently
        self.model.fit(X_train_flat, y_train_flat)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate energy consumption forecasts using linear relationships.
        
        Prediction Formula: ŷ = Xβ + β₀
        Where:
        - ŷ: predicted energy consumption
        - X: input features (weather, time, past consumption)
        - β: learned feature coefficients
        - β₀: learned intercept (baseline energy)
        
        Advantages:
        - **Fastest Predictions**: Simple matrix multiplication
        - **Deterministic**: No randomness in predictions
        - **Extrapolation**: Can predict beyond training data range
        - **Interpretable**: Easy to understand why specific prediction was made
        
        Limitations:
        - **Linear Assumptions**: May miss non-linear energy relationships
        - **Feature Engineering**: Requires good input features for complex patterns
        
        Args:
            X (np.ndarray): Input features for prediction (flattened if needed)
        
        Returns:
            np.ndarray: Predicted energy consumption values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Flatten input for linear regression
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
        
        # Compute linear prediction: ŷ = Xβ + β₀
        return self.model.predict(X_flat)
    
    def save_model(self, filepath: str) -> None:
        """
        Save linear regression model with learned coefficients.
        
        Saves the coefficient vector and intercept that encode
        linear relationships between features and energy consumption.
        Very lightweight - only stores learned parameters, not training data.
        
        Args:
            filepath (str): Path to save the model file
        """
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load pre-trained linear regression model.
        
        Restores learned coefficients for immediate energy predictions.
        Useful for deploying interpretable forecasting models.
        
        Args:
            filepath (str): Path to the saved model file
        """
        self.model = joblib.load(filepath)
        self.is_fitted = True


class RandomForestForecaster(BaseForecaster):
    """
    Random Forest ensemble method for robust time series forecasting.
    
    Random Forests are exceptionally well-suited for building energy forecasting:
    
    1. **Non-linear Pattern Capture**: Decision trees naturally handle complex,
       non-linear relationships between weather, occupancy, and energy consumption
    2. **Feature Interaction Detection**: Automatically discovers interactions
       between variables (e.g., temperature × humidity effects on HVAC load)
    3. **Outlier Robustness**: Tree-based splits are resistant to extreme values
       common in energy data (equipment failures, unusual weather)
    4. **Missing Value Handling**: Can work with incomplete sensor data
    5. **Feature Importance**: Ranks which variables most influence energy consumption
    6. **No Assumptions**: Doesn't require linearity, normality, or homoscedasticity
    
    Ensemble Advantages:
    - **Bias-Variance Trade-off**: Multiple trees reduce overfitting while maintaining
      low bias through individual tree complexity
    - **Bootstrap Aggregating**: Each tree trains on different data samples,
      improving generalization to new buildings or time periods
    - **Random Feature Selection**: At each split, only considers subset of features,
      preventing dominance by strongly correlated variables
    
    Mathematical Foundation:
    Prediction = (1/B) ∑ᵢᴇ₁ᴮ Treeᵢ(x)
    Where B is number of trees, each trained on bootstrap sample with random features
    
    Why Random Forests Excel for Energy Data:
    - **Seasonal Patterns**: Trees can create rules like "if month=July and temp>80°F"
    - **Building Types**: Different tree paths for different building characteristics
    - **Weather Dependencies**: Complex thresholds for heating/cooling transitions
    - **Occupancy Effects**: Non-linear relationships with business hours, weekends
    
    Hyperparameter Choices:
    - n_estimators=100: Balance between performance and computation time
    - max_depth=None: Allow trees to grow until pure leaves (controlled by min_samples_split)
    - random_state=42: Reproducible results for model comparison
    - n_jobs=-1: Use all CPU cores for parallel tree training
    """
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize Random Forest forecaster with energy-optimized parameters.
        
        Args:
            n_estimators (int): Number of decision trees in the forest
                               100 trees provide good bias-variance balance
                               More trees = better performance but slower training
            max_depth (None): Maximum tree depth (None = grow until pure leaves)
                             Deep trees capture complex energy patterns
                             Controlled by other parameters to prevent overfitting
            random_state (int): Random seed for reproducible results
                               Important for consistent model comparisons
        """
        super().__init__("RandomForest")
        
        # Random Forest Regressor optimized for energy forecasting
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,    # Number of trees in the ensemble
            max_depth=max_depth,          # Tree depth (None allows full growth)
            random_state=random_state,    # Reproducibility seed
            n_jobs=-1,                    # Parallel training using all CPU cores
            # Additional energy-specific defaults:
            min_samples_split=2,          # Minimum samples to split (default, allows fine patterns)
            min_samples_leaf=1,           # Minimum samples per leaf (default, captures rare events)
            max_features='auto',          # Features considered per split (sqrt(n) for regression)
            bootstrap=True,               # Bootstrap sampling for tree diversity
            oob_score=False              # Out-of-bag scoring (disabled for speed)
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train Random Forest ensemble on building energy consumption patterns.
        
        Training Process:
        1. **Bootstrap Sampling**: Each tree trains on random subset of data (63.2% on average)
        2. **Random Feature Selection**: At each split, considers subset of features
        3. **Greedy Tree Growth**: Recursively splits nodes to minimize MSE
        4. **Ensemble Assembly**: Combines multiple diverse trees
        
        Why Random Forests Work Well for Energy Data:
        
        **Complex Pattern Learning**:
        - Trees can learn rules like: "IF temperature > 75°F AND weekday THEN high_cooling"
        - Captures threshold effects in HVAC operation
        - Models seasonal and diurnal patterns naturally
        
        **Robustness Benefits**:
        - Bootstrap sampling reduces sensitivity to outliers
        - Random feature selection prevents over-reliance on dominant variables
        - Ensemble averaging smooths individual tree predictions
        
        **Building-Specific Adaptation**:
        - Different trees can specialize in different operating conditions
        - Handles heterogeneous building types within same dataset
        - Adapts to unusual energy consumption patterns
        
        **Computational Efficiency**:
        - Parallel tree training (n_jobs=-1) utilizes multiple CPU cores
        - Training time scales linearly with number of trees
        - No hyperparameter tuning required for basic performance
        
        Args:
            X_train (np.ndarray): Input features (flattened sequences or engineered features)
                                 Can include weather, time, past consumption, building metadata
            y_train (np.ndarray): Target energy consumption values
                                 Continuous energy values (kWh, kW, etc.)
        """
        # Flatten sequential data for tree-based learning
        # Random forests expect tabular data: (samples, features)
        X_train_flat = X_train.reshape(X_train.shape[0], -1) if len(X_train.shape) > 2 else X_train
        y_train_flat = y_train.flatten() if len(y_train.shape) > 1 else y_train
        
        # Train ensemble of decision trees with bootstrap sampling and random features
        self.model.fit(X_train_flat, y_train_flat)
        
        # After training, can access feature importance:
        # self.model.feature_importances_ shows which variables most influence energy
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate energy consumption forecasts using ensemble of decision trees.
        
        Prediction Process:
        1. **Individual Tree Predictions**: Each tree generates prediction based on
           learned splitting rules and energy consumption patterns
        2. **Ensemble Averaging**: Combines predictions from all trees
           Final prediction = (1/n_trees) ∑ tree_predictions
        3. **Variance Reduction**: Averaging reduces prediction noise from individual trees
        
        Tree-Based Decision Making for Energy:
        - Each tree follows learned rules: "IF temp > threshold AND hour < 9 THEN low_energy"
        - Different trees capture different aspects of energy consumption
        - Robust to feature noise and missing values
        
        Advantages for Energy Forecasting:
        **Non-linear Relationships**: Naturally handles HVAC efficiency curves,
        thermal lag effects, and occupancy thresholds
        
        **Interpretability**: Can extract decision rules showing how predictions are made
        (e.g., "High energy consumption when temp > 80°F and weekday = True")
        
        **Confidence Estimation**: Can compute prediction intervals using individual
        tree predictions (though this requires custom implementation)
        
        **Feature Robustness**: Predictions remain stable even with some missing
        input features (sensors fail, weather data unavailable)
        
        Args:
            X (np.ndarray): Input features for prediction (flattened if sequential)
        
        Returns:
            np.ndarray: Predicted energy consumption values (ensemble average)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Flatten input for tree-based processing
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
        
        # Generate ensemble predictions (average of all trees)
        return self.model.predict(X_flat)
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained Random Forest ensemble with all decision trees.
        
        Saves complete ensemble including:
        - All individual decision trees with learned splitting rules
        - Feature importance rankings for energy consumption factors
        - Bootstrap sampling information
        - Model hyperparameters
        
        Important for energy management systems requiring model persistence
        and interpretability through feature importance analysis.
        
        Args:
            filepath (str): Path to save the model file
        """
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load pre-trained Random Forest ensemble for energy forecasting.
        
        Restores complete ensemble state including all decision trees
        and learned energy consumption patterns. Enables immediate
        predictions and feature importance analysis.
        
        Args:
            filepath (str): Path to the saved model file
        """
        self.model = joblib.load(filepath)
        self.is_fitted = True


class PolynomialForecaster(BaseForecaster):
    """
    Polynomial Regression for non-linear time series forecasting with interpretable curves.
    
    Polynomial regression extends linear models to capture non-linear energy patterns
    while maintaining interpretability and analytical tractability:
    
    1. **Non-linear Energy Relationships**: Captures curved relationships between
       temperature and HVAC energy consumption (heating/cooling curves)
    2. **Interaction Effects**: Automatically includes feature interactions
       (temperature × humidity effects on energy)
    3. **Seasonal Curves**: Models non-linear seasonal patterns in energy consumption
    4. **Threshold Effects**: Can approximate step functions in building operations
    5. **Analytical Solutions**: Maintains closed-form solution like linear regression
    
    Mathematical Foundation:
    Extends linear model: E(y|x) = β₀ + β₁x₁ + β₂x₁² + β₃x₁x₂ + ...
    
    For degree=2 with 2 features [x₁, x₂], creates:
    [1, x₁, x₂, x₁², x₁x₂, x₂²]
    
    Energy Applications:
    - **Temperature Curves**: Quadratic relationships model U-shaped energy consumption
      (heating at low temps, cooling at high temps, minimum at comfort zone)
    - **Time Interactions**: Captures how temperature effects vary by time of day
    - **Equipment Efficiency**: Models non-linear efficiency curves of HVAC systems
    - **Occupancy Effects**: Non-linear relationship between occupancy and energy
    
    Advantages:
    - **Interpretable Coefficients**: Each term has clear physical meaning
    - **Fast Training**: Closed-form solution like linear regression
    - **Smooth Predictions**: Continuous curves rather than step functions
    - **Feature Engineering**: Automatically creates interaction terms
    
    Limitations:
    - **Overfitting Risk**: High-degree polynomials can oscillate wildly
    - **Extrapolation Issues**: Poor behavior outside training data range
    - **Feature Explosion**: Number of terms grows exponentially with degree
    - **Multicollinearity**: High correlation between polynomial terms
    
    Best Practices for Energy Forecasting:
    - Use degree=2 or 3 to balance flexibility and stability
    - Apply feature scaling to prevent numerical issues
    - Consider regularization (Ridge/Lasso) for high-degree polynomials
    - Validate extrapolation performance on out-of-sample data
    """
    
    def __init__(self, degree=2):
        """
        Initialize Polynomial Regression forecaster.
        
        Args:
            degree (int): Polynomial degree for feature expansion
                         degree=1: Linear regression (no benefit over LinearForecaster)
                         degree=2: Quadratic terms + interactions (recommended for energy)
                         degree=3: Cubic terms (use carefully, risk of overfitting)
                         degree>3: Generally not recommended for energy data
        
        Degree=2 Rationale for Energy Forecasting:
        - Captures U-shaped temperature-energy relationships
        - Includes pairwise feature interactions
        - Balances flexibility with interpretability
        - Avoids extreme oscillations of higher-degree polynomials
        """
        super().__init__("Polynomial")
        self.degree = degree
        
        # Polynomial feature transformer
        # Creates all polynomial combinations up to specified degree
        self.poly_features = PolynomialFeatures(
            degree=degree,
            include_bias=True,     # Include constant term (energy baseline)
            interaction_only=False # Include both interactions and pure powers
        )
        
        # Linear regression model applied to polynomial features
        self.model = LinearRegression()
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train polynomial regression model on energy consumption data.
        
        Training Process:
        1. **Feature Expansion**: Transform input features into polynomial basis
           Original: [temp, humidity] → Polynomial: [1, temp, humidity, temp², temp*humidity, humidity²]
        2. **Linear Fitting**: Apply linear regression to expanded feature space
        3. **Coefficient Learning**: Learn weights for each polynomial term
        
        Polynomial Feature Engineering for Energy:
        
        **Degree=2 Example** (Temperature and Hour of Day):
        - Constant: 1 (baseline energy consumption)
        - Linear: temp, hour (direct effects)
        - Quadratic: temp², hour² (U-shaped curves)
        - Interaction: temp*hour (temperature effect varies by time)
        
        **Physical Interpretation**:
        - temp² coefficient: Captures U-shaped energy curve (heating + cooling)
        - temp*hour coefficient: Shows how temperature sensitivity changes throughout day
        - hour² coefficient: Models non-linear daily energy patterns
        
        **Mathematical Benefits**:
        - Closed-form solution: β = (XᵀX)⁻¹Xᵀy (no iterative optimization)
        - Global optimum guaranteed (convex optimization)
        - No hyperparameter tuning required
        
        **Computational Considerations**:
        - Feature space grows as C(n+d, d) where n=features, d=degree
        - For 10 features, degree=3: creates 286 polynomial terms
        - May need regularization for high-dimensional polynomial spaces
        
        Args:
            X_train (np.ndarray): Input features (flattened if sequential)
                                 Should be scaled to prevent numerical issues
            y_train (np.ndarray): Target energy consumption values
        """
        # Flatten sequential data for polynomial regression
        X_train_flat = X_train.reshape(X_train.shape[0], -1) if len(X_train.shape) > 2 else X_train
        y_train_flat = y_train.flatten() if len(y_train.shape) > 1 else y_train
        
        # Transform features into polynomial basis
        # Creates all combinations of features up to specified degree
        X_poly = self.poly_features.fit_transform(X_train_flat)
        
        # Fit linear regression on polynomial features
        # This learns coefficients for each polynomial term
        self.model.fit(X_poly, y_train_flat)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate energy consumption forecasts using polynomial relationships.
        
        Prediction Process:
        1. **Feature Expansion**: Transform input into same polynomial basis as training
        2. **Linear Combination**: Multiply polynomial features by learned coefficients
        3. **Curve Generation**: Produces smooth, non-linear energy predictions
        
        Polynomial Prediction Formula:
        ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₁² + β₄x₁x₂ + β₅x₂² + ...
        
        Energy Forecasting Benefits:
        
        **Non-linear Energy Curves**:
        - Models realistic HVAC response to temperature changes
        - Captures comfort zone (minimum energy) and extreme weather (high energy)
        - Represents equipment efficiency curves and thermal lag effects
        
        **Interaction Effects**:
        - Temperature-humidity interactions affect perceived comfort and HVAC load
        - Time-weather interactions show how building thermal mass affects energy
        - Occupancy-temperature interactions model adaptive comfort behaviors
        
        **Smooth Extrapolation**:
        - Provides continuous curves rather than step functions
        - Better than linear models for moderate extrapolation
        - More stable than neural networks for out-of-sample conditions
        
        **Interpretable Predictions**:
        - Each coefficient shows marginal effect of polynomial terms
        - Can analyze which non-linear effects dominate energy consumption
        - Enables physical validation of learned relationships
        
        Cautions:
        - High-degree polynomials may oscillate outside training range
        - Sensitive to input feature scaling
        - May require regularization for high-dimensional polynomial spaces
        
        Args:
            X (np.ndarray): Input features for prediction (same scale as training)
        
        Returns:
            np.ndarray: Predicted energy consumption with polynomial relationships
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Flatten input features
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
        
        # Transform to polynomial feature space (same transformation as training)
        X_poly = self.poly_features.transform(X_flat)
        
        # Generate polynomial predictions
        return self.model.predict(X_poly)
    
    def save_model(self, filepath: str) -> None:
        """
        Save polynomial regression model with feature transformer and coefficients.
        
        Saves both:
        - Polynomial feature transformer (for consistent feature expansion)
        - Linear regression model (with polynomial coefficients)
        
        Essential for maintaining exact polynomial feature mapping
        when deploying energy forecasting models.
        
        Args:
            filepath (str): Path to save the model file
        """
        model_data = {
            'poly_features': self.poly_features,  # Feature transformation parameters
            'model': self.model                   # Learned polynomial coefficients
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load pre-trained polynomial regression model.
        
        Restores both feature transformer and learned coefficients,
        enabling immediate polynomial energy predictions.
        
        Args:
            filepath (str): Path to the saved model file
        """
        model_data = joblib.load(filepath)
        self.poly_features = model_data['poly_features']  # Restore feature transformer
        self.model = model_data['model']                   # Restore polynomial coefficients
        self.is_fitted = True


def get_all_forecasters() -> Dict[str, BaseForecaster]:
    """
    Get dictionary of all available forecasting models for energy prediction comparison.
    
    This function provides a comprehensive suite of forecasting algorithms,
    each with different strengths for building energy consumption prediction:
    
    **Deep Learning Models** (Best for complex temporal patterns):
    - LSTM: Captures long-term dependencies and temporal sequences
    - ANN: Fast non-linear mapping for feature-rich datasets
    
    **Probabilistic Models** (Best for uncertainty quantification):
    - Gaussian: Provides prediction confidence intervals
    
    **Tree-Based Models** (Best for interpretability and robustness):
    - RandomForest: Handles non-linear patterns and feature interactions
    
    **Linear Models** (Best for baselines and interpretability):
    - Linear: Fast, interpretable baseline with feature coefficients
    - Polynomial: Non-linear extension with interaction terms
    
    **Advanced Models** (If available):
    - Transformer: Attention-based models for complex time series
    - TimesFM: Foundation model approach for time series
    
    Model Selection Guidelines for Energy Forecasting:
    
    1. **Start with Linear**: Establishes baseline performance and feature importance
    2. **Try RandomForest**: Often best overall performance with minimal tuning
    3. **Use LSTM**: When temporal dependencies are crucial (long sequences)
    4. **Apply Gaussian**: When uncertainty quantification is needed
    5. **Consider Polynomial**: For interpretable non-linear relationships
    6. **Experiment with ANN**: For fast inference with good performance
    7. **Test Transformers**: For state-of-the-art performance (if available)
    
    Returns:
        Dict[str, BaseForecaster]: Dictionary mapping model names to instances
                                  Ready for training on energy consumption data
    
    Example Usage:
        ```python
        models = get_all_forecasters()
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
        ```
    """
    # Initialize base forecasting models with energy-optimized defaults
    base_models = {
        'LSTM': LSTMForecaster(),           # Recurrent neural network for sequences
        'ANN': ANNForecaster(),             # Multi-layer perceptron for features
        'Gaussian': GaussianForecaster(),   # Probabilistic regression with uncertainty
        'Linear': LinearForecaster(),       # Linear baseline with interpretable coefficients
        'RandomForest': RandomForestForecaster(),  # Ensemble of decision trees
        'Polynomial': PolynomialForecaster()       # Non-linear regression with interactions
    }
    
    # Attempt to add transformer models if dependencies are available
    try:
        from .transformer_models import get_transformer_forecasters
        transformer_models = get_transformer_forecasters()
        base_models.update(transformer_models)
        print(f"✅ Loaded {len(transformer_models)} transformer models")
    except ImportError:
        print("⚠️  Transformer models not available (TensorFlow required)")
    
    print(f"📊 Total available forecasting models: {len(base_models)}")
    return base_models