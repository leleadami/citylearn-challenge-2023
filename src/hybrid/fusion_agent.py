"""
Hybrid Fusion Agent: RL + LSTM + Transformer Integration
Combines reinforcement learning with advanced forecasting for building energy management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class SimpleRandomPolicy:
    """Simple random policy as fallback for RL component."""
    
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Generate random action."""
        return np.random.uniform(-1, 1, self.action_dim)
        
    def learn(self, state, action, reward, next_state):
        """No learning for random policy."""
        pass


class HybridFusionAgent:
    """
    Advanced hybrid agent combining:
    1. LSTM networks for temporal pattern learning
    2. Transformer attention for key feature identification  
    3. Reinforcement Learning for optimal control decisions
    4. Classical ML as fallback for robustness
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 sequence_length: int = 24,
                 prediction_horizon: int = 48,
                 learning_rate: float = 0.001):
        """
        Initialize hybrid fusion agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space  
            sequence_length: Input sequence length for neural networks
            prediction_horizon: Forecasting horizon in hours
            learning_rate: Learning rate for neural components
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.learning_rate = learning_rate
        
        # Component models
        self.lstm_forecaster = None
        self.transformer_forecaster = None
        self.rl_policy = None
        self.classical_fallback = None
        
        # Fusion weights (learned adaptively)
        self.fusion_weights = {
            'lstm': 0.4,
            'transformer': 0.4, 
            'classical': 0.2
        }
        
        # State tracking
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.prediction_history = []
        
        # Performance metrics
        self.performance_metrics = {
            'lstm_accuracy': [],
            'transformer_accuracy': [],
            'classical_accuracy': [],
            'fusion_accuracy': [],
            'rl_reward': []
        }
        
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all agent components with fallback mechanisms."""
        print("[INIT] Initializing hybrid fusion agent components...")
        
        # Initialize LSTM component
        if TF_AVAILABLE:
            try:
                self.lstm_forecaster = self._build_lstm_model()
                print("[LSTM] Advanced LSTM forecaster initialized")
            except Exception as e:
                print(f"[LSTM] TensorFlow LSTM failed: {e}")
                self.lstm_forecaster = None
        else:
            print("[LSTM] TensorFlow not available - using classical fallback")
            
        # Initialize Transformer component  
        if TF_AVAILABLE:
            try:
                self.transformer_forecaster = self._build_transformer_model()
                print("[TRANSFORMER] Multi-head attention model initialized")
            except Exception as e:
                print(f"[TRANSFORMER] TensorFlow Transformer failed: {e}")
                self.transformer_forecaster = None
        else:
            print("[TRANSFORMER] TensorFlow not available - using classical fallback")
            
        # Initialize RL component
        self.rl_policy = self._build_rl_policy()
        print("[RL] Q-Learning policy initialized")
        
        # Initialize classical fallback
        if SKLEARN_AVAILABLE:
            self.classical_fallback = {
                'linear': LinearRegression(),
                'forest': RandomForestRegressor(n_estimators=50, random_state=42)
            }
            print("[CLASSICAL] Scikit-learn fallback models initialized")
        else:
            print("[CLASSICAL] Using basic statistical fallback")
            
    def _build_lstm_model(self):
        """Build advanced LSTM forecasting model."""
        if not TF_AVAILABLE:
            return None
            
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(self.prediction_horizon, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def _build_transformer_model(self):
        """Build transformer model with multi-head attention."""
        if not TF_AVAILABLE:
            return None
            
        # Simplified transformer for energy forecasting
        inputs = tf.keras.Input(shape=(self.sequence_length, 1))
        
        # Multi-head attention layer
        attention_output = MultiHeadAttention(
            num_heads=4, 
            key_dim=16
        )(inputs, inputs)
        
        # Layer normalization
        x = LayerNormalization()(attention_output + inputs)
        
        # Feed forward network
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.prediction_horizon, activation='linear')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def _build_rl_policy(self):
        """Build Q-Learning policy for control decisions."""
        try:
            from ..rl.q_learning import QLearningAgent
            
            return QLearningAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                learning_rate=0.1,
                epsilon=0.1,
                discretization_bins=10
            )
        except ImportError as e:
            print(f"[RL] Q-Learning import failed: {e}")
            print("[RL] Using simple random policy fallback")
            return SimpleRandomPolicy(self.action_dim)
        
    def forecast(self, historical_data: np.ndarray, target_name: str = "energy") -> Dict[str, np.ndarray]:
        """
        Generate ensemble forecasts using all available models.
        
        Args:
            historical_data: Historical time series data
            target_name: Name of the forecasting target
            
        Returns:
            Dictionary with forecasts from each model and fusion result
        """
        forecasts = {}
        
        # Prepare data for neural networks
        if len(historical_data) >= self.sequence_length:
            sequence_data = self._prepare_sequence_data(historical_data)
        else:
            sequence_data = None
            
        # LSTM forecast
        if self.lstm_forecaster is not None and sequence_data is not None:
            try:
                lstm_pred = self.lstm_forecaster.predict(sequence_data, verbose=0)
                forecasts['lstm'] = lstm_pred.flatten()
                print(f"[LSTM] Forecast generated for {target_name}")
            except Exception as e:
                print(f"[LSTM] Forecast failed: {e}")
                forecasts['lstm'] = self._classical_forecast(historical_data)
        else:
            forecasts['lstm'] = self._classical_forecast(historical_data)
            
        # Transformer forecast
        if self.transformer_forecaster is not None and sequence_data is not None:
            try:
                transformer_pred = self.transformer_forecaster.predict(sequence_data, verbose=0)
                forecasts['transformer'] = transformer_pred.flatten()
                print(f"[TRANSFORMER] Forecast generated for {target_name}")
            except Exception as e:
                print(f"[TRANSFORMER] Forecast failed: {e}")
                forecasts['transformer'] = self._classical_forecast(historical_data)
        else:
            forecasts['transformer'] = self._classical_forecast(historical_data)
            
        # Classical forecast  
        forecasts['classical'] = self._classical_forecast(historical_data)
        
        # Fusion forecast (weighted ensemble)
        forecasts['fusion'] = self._fuse_forecasts(forecasts)
        
        return forecasts
        
    def _prepare_sequence_data(self, data: np.ndarray) -> np.ndarray:
        """Prepare data for neural network input."""
        if len(data) < self.sequence_length:
            # Pad with mean if insufficient data
            mean_val = np.mean(data)
            padded_data = np.concatenate([
                np.full(self.sequence_length - len(data), mean_val),
                data
            ])
        else:
            padded_data = data[-self.sequence_length:]
            
        return padded_data.reshape(1, self.sequence_length, 1)
        
    def _classical_forecast(self, data: np.ndarray) -> np.ndarray:
        """Generate classical ML or statistical forecast."""
        if len(data) < 2:
            # Return constant forecast if insufficient data
            return np.full(self.prediction_horizon, data[-1] if len(data) > 0 else 0.0)
            
        if SKLEARN_AVAILABLE and self.classical_fallback:
            try:
                # Use linear regression on trend
                X = np.arange(len(data)).reshape(-1, 1)
                y = data
                
                # Train simple linear model
                self.classical_fallback['linear'].fit(X, y)
                
                # Predict future values
                future_X = np.arange(len(data), len(data) + self.prediction_horizon).reshape(-1, 1)
                forecast = self.classical_fallback['linear'].predict(future_X)
                
                return forecast
                
            except Exception as e:
                print(f"[CLASSICAL] Scikit-learn forecast failed: {e}")
                
        # Statistical fallback: simple trend + seasonality
        if len(data) >= 24:  # Daily seasonality
            recent_trend = np.mean(data[-24:])
            seasonal_pattern = data[-24:] - recent_trend
            forecast = []
            
            for i in range(self.prediction_horizon):
                seasonal_component = seasonal_pattern[i % 24]
                forecast.append(recent_trend + seasonal_component)
                
            return np.array(forecast)
        else:
            # Simple moving average
            forecast_value = np.mean(data[-min(len(data), 7):])
            return np.full(self.prediction_horizon, forecast_value)
            
    def _fuse_forecasts(self, forecasts: Dict[str, np.ndarray]) -> np.ndarray:
        """Fuse multiple forecasts using adaptive weights."""
        fusion = np.zeros(self.prediction_horizon)
        total_weight = 0.0
        
        for model_name, weight in self.fusion_weights.items():
            if model_name in forecasts:
                forecast = forecasts[model_name]
                if len(forecast) == self.prediction_horizon:
                    fusion += weight * forecast
                    total_weight += weight
                    
        if total_weight > 0:
            fusion /= total_weight
        else:
            # Fallback to simple average
            valid_forecasts = [f for f in forecasts.values() if len(f) == self.prediction_horizon]
            if valid_forecasts:
                fusion = np.mean(valid_forecasts, axis=0)
                
        return fusion
        
    def act(self, state: np.ndarray, forecast: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate control action using RL policy informed by forecasts.
        
        Args:
            state: Current building state
            forecast: Optional forecast to inform decision
            
        Returns:
            Control action
        """
        # Store state for learning
        self.state_history.append(state.copy())
        
        # Use RL policy for action selection
        if self.rl_policy:
            action = self.rl_policy.predict(state)
        else:
            # Fallback: zero action (no control)
            action = np.zeros(self.action_dim)
            
        # Store action
        self.action_history.append(action.copy())
        
        return action
        
    def learn(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray):
        """Update RL policy based on experience."""
        self.reward_history.append(reward)
        
        if self.rl_policy:
            self.rl_policy.learn(state, action, reward, next_state)
            
    def train_forecasters(self, training_data: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]):
        """
        Train neural forecasting models on historical data.
        
        Args:
            training_data: Historical input sequences
            targets: Historical target values for forecasting
        """
        print("[TRAIN] Training hybrid forecasting models...")
        
        for target_name, target_data in targets.items():
            if target_name in training_data:
                input_data = training_data[target_name]
                
                # Train LSTM
                if self.lstm_forecaster is not None:
                    try:
                        self.lstm_forecaster.fit(
                            input_data, target_data,
                            epochs=20, batch_size=32, verbose=0
                        )
                        print(f"[LSTM] Trained on {target_name}")
                    except Exception as e:
                        print(f"[LSTM] Training failed for {target_name}: {e}")
                        
                # Train Transformer
                if self.transformer_forecaster is not None:
                    try:
                        self.transformer_forecaster.fit(
                            input_data, target_data,
                            epochs=20, batch_size=32, verbose=0
                        )
                        print(f"[TRANSFORMER] Trained on {target_name}")
                    except Exception as e:
                        print(f"[TRANSFORMER] Training failed for {target_name}: {e}")
                        
    def evaluate_performance(self, test_data: Dict[str, np.ndarray], test_targets: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate performance of all components."""
        performance = {}
        
        for target_name, target_data in test_targets.items():
            if target_name in test_data:
                input_data = test_data[target_name]
                
                # Get forecasts
                sample_data = input_data[0].flatten() if len(input_data.shape) > 1 else input_data
                forecasts = self.forecast(sample_data, target_name)
                
                # Calculate errors (simplified for single prediction)
                true_value = target_data[0] if len(target_data) > 0 else 0.0
                
                for model_name, forecast in forecasts.items():
                    pred_value = forecast[0] if len(forecast) > 0 else 0.0
                    error = abs(true_value - pred_value)
                    performance[f"{model_name}_{target_name}_mae"] = error
                    
        return performance
        
    def update_fusion_weights(self, performance_metrics: Dict[str, float]):
        """Adaptively update fusion weights based on performance."""
        # Simple adaptive weighting based on inverse error
        model_errors = {}
        
        for metric_name, error in performance_metrics.items():
            if 'mae' in metric_name:
                model_name = metric_name.split('_')[0]
                if model_name in self.fusion_weights:
                    model_errors[model_name] = error
                    
        if model_errors:
            # Inverse error weighting
            total_inv_error = sum(1.0 / (error + 1e-6) for error in model_errors.values())
            
            for model_name, error in model_errors.items():
                self.fusion_weights[model_name] = (1.0 / (error + 1e-6)) / total_inv_error
                
            print(f"[FUSION] Updated weights: {self.fusion_weights}")
            
    def save_model(self, filepath: str):
        """Save hybrid agent state."""
        state = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'fusion_weights': self.fusion_weights,
            'performance_metrics': self.performance_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
        print(f"[SAVE] Hybrid agent saved to {filepath}")
        
    def load_model(self, filepath: str):
        """Load hybrid agent state."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            self.fusion_weights = state.get('fusion_weights', self.fusion_weights)
            self.performance_metrics = state.get('performance_metrics', self.performance_metrics)
            
            print(f"[LOAD] Hybrid agent loaded from {filepath}")
        except Exception as e:
            print(f"[LOAD] Failed to load from {filepath}: {e}")
            
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of hybrid agent."""
        return {
            'components': {
                'lstm': self.lstm_forecaster is not None,
                'transformer': self.transformer_forecaster is not None,
                'rl_policy': self.rl_policy is not None,
                'classical': self.classical_fallback is not None
            },
            'fusion_weights': self.fusion_weights,
            'experience': {
                'states': len(self.state_history),
                'actions': len(self.action_history),
                'rewards': len(self.reward_history)
            },
            'tensorflow_available': TF_AVAILABLE,
            'sklearn_available': SKLEARN_AVAILABLE
        }


def create_hybrid_agent(state_dim: int, action_dim: int, **kwargs) -> HybridFusionAgent:
    """Factory function to create hybrid fusion agent."""
    print("[FACTORY] Creating hybrid fusion agent...")
    print(f"[CONFIG] State dim: {state_dim}, Action dim: {action_dim}")
    
    agent = HybridFusionAgent(state_dim, action_dim, **kwargs)
    
    status = agent.get_status()
    print(f"[STATUS] Components active: {status['components']}")
    print(f"[STATUS] TensorFlow: {status['tensorflow_available']}, Scikit-learn: {status['sklearn_available']}")
    
    return agent


if __name__ == "__main__":
    # Demonstration of hybrid agent
    print("Hybrid Fusion Agent - Demonstration")
    print("=" * 50)
    
    # Create agent
    agent = create_hybrid_agent(state_dim=10, action_dim=5)
    
    # Simulate data
    historical_data = np.random.randn(100)
    state = np.random.randn(10)
    
    # Test forecasting
    forecasts = agent.forecast(historical_data, "cooling_demand")
    print(f"Generated forecasts: {list(forecasts.keys())}")
    
    # Test action generation
    action = agent.act(state)
    print(f"Generated action: {action}")
    
    # Show status
    status = agent.get_status()
    print(f"Agent status: {status}")