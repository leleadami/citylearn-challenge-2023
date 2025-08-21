"""
Transformer-based models for advanced time series forecasting.

This module implements state-of-the-art attention-based architectures specifically
designed for building energy consumption prediction:

1. **TransformerForecaster**: Standard transformer encoder with multi-head attention
   for capturing complex temporal patterns in energy consumption

2. **TimesFMInspiredForecaster**: Implementation inspired by Google's TimesFM
   foundation model, featuring patching and advanced positional encoding

Key Innovations for Energy Forecasting:
- **Attention Mechanisms**: Identify which historical time points most influence
  future energy consumption (e.g., same hour previous day, weather transitions)
- **Positional Encoding**: Captures cyclical patterns (daily, weekly, seasonal)
- **Multi-Head Attention**: Learns multiple types of temporal relationships
  simultaneously (short-term trends, long-term patterns, weather effects)
- **Patch-Based Processing**: TimesFM-style approach for efficient sequence modeling

Why Transformers Excel for Building Energy Forecasting:
1. **Long-Range Dependencies**: Attention can connect energy usage to events
   hours or days in the past (thermal mass effects, scheduling patterns)
2. **Multiple Time Scales**: Different attention heads can focus on different
   temporal patterns (hourly cycles, daily schedules, weekly variations)
3. **Weather Integration**: Can learn complex relationships between weather
   sequences and energy consumption patterns
4. **Building-Specific Adaptation**: Self-attention adapts to unique building
   characteristics and operational patterns
5. **Uncertainty Handling**: Attention weights provide interpretability about
   prediction confidence and influential factors
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import math
from typing import Tuple, Optional
import joblib
from .base_models import BaseForecaster


class MultiHeadAttention(layers.Layer):
    """
    Multi-head attention mechanism optimized for time series energy forecasting.
    
    Multi-head attention is crucial for energy forecasting because it allows the model
    to simultaneously attend to different types of temporal patterns:
    
    **Head Specialization for Energy Data**:
    - Head 1: Daily cycles (24-hour HVAC patterns)
    - Head 2: Weekly patterns (weekday vs weekend consumption)
    - Head 3: Weather sensitivity (temperature-energy relationships)
    - Head 4: Seasonal trends (heating/cooling season transitions)
    - Head 5-8: Building-specific patterns (occupancy, equipment schedules)
    
    **Mathematical Foundation**:
    Attention(Q,K,V) = softmax(QKᵀ/√(d_k))V
    
    Where:
    - Q (Query): "What energy pattern am I trying to predict?"
    - K (Key): "What historical patterns are available?"
    - V (Value): "What are the actual energy values for those patterns?"
    - d_k: Scaling factor to prevent vanishing gradients
    
    **Energy Forecasting Benefits**:
    1. **Adaptive Attention**: Automatically focuses on most relevant time points
    2. **Pattern Discovery**: Learns complex temporal dependencies without manual feature engineering
    3. **Weather Integration**: Connects weather sequences to energy responses
    4. **Interpretability**: Attention weights show which historical periods influence predictions
    5. **Robust to Missing Data**: Can attend to available information when sensors fail
    
    **Architecture Details**:
    - Multiple heads enable parallel attention to different pattern types
    - Linear projections create query, key, value representations
    - Scaled dot-product attention prevents gradient explosion
    - Final dense layer combines multi-head outputs
    """
    
    def __init__(self, d_model: int, num_heads: int, **kwargs):
        """
        Initialize multi-head attention for energy time series.
        
        Args:
            d_model (int): Model dimension (must be divisible by num_heads)
                          64-128 typical for energy forecasting
            num_heads (int): Number of parallel attention heads
                            8 heads allow specialization for different energy patterns
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        
        # Ensure model dimension is evenly divisible by number of heads
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        # Dimension per attention head
        self.depth = d_model // num_heads
        
        # Linear projections for query, key, value
        # These learn to transform input into attention representations
        self.wq = layers.Dense(d_model, name="query_projection")    # "What to predict?"
        self.wk = layers.Dense(d_model, name="key_projection")      # "What patterns exist?"
        self.wv = layers.Dense(d_model, name="value_projection")    # "What are the values?"
        
        # Final linear transformation after multi-head concatenation
        self.dense = layers.Dense(d_model, name="output_projection")
    
    def split_heads(self, x, batch_size):
        """
        Split model dimension into multiple attention heads for parallel processing.
        
        This reshaping enables each head to focus on different aspects of energy patterns:
        - Original: (batch, sequence, d_model)
        - Split: (batch, num_heads, sequence, depth)
        
        Each head operates on depth=d_model/num_heads dimensions, allowing
        specialization for different types of temporal relationships in energy data.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            batch_size: Current batch size
            
        Returns:
            Reshaped tensor: (batch_size, num_heads, seq_len, depth)
        """
        # Reshape to separate heads: (batch, seq_len, num_heads, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        # Transpose to put heads dimension first: (batch, num_heads, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        """
        Compute multi-head attention for energy time series.
        
        Attention Process for Energy Forecasting:
        1. **Linear Projections**: Transform inputs into query, key, value spaces
        2. **Head Splitting**: Divide into parallel attention computations
        3. **Scaled Attention**: Compute attention weights showing temporal importance
        4. **Value Aggregation**: Weighted combination of historical energy values
        5. **Head Concatenation**: Combine insights from all attention heads
        6. **Output Projection**: Final transformation for next layer
        
        Energy Attention Interpretation:
        - High attention weights indicate historical time points that strongly
          influence current energy prediction
        - Different heads may attend to different patterns (daily, weekly, weather)
        - Attention weights provide explainability for energy forecasting decisions
        
        Args:
            v, k, q: Value, key, query tensors (typically same input for self-attention)
            mask: Optional attention mask (for causal/future masking)
            
        Returns:
            output: Attended representation for energy forecasting
            attention_weights: Interpretable attention patterns
        """
        batch_size = tf.shape(q)[0]
        
        # Apply linear projections to create query, key, value representations
        q = self.wq(q)  # Query: "What energy pattern to predict?"
        k = self.wk(k)  # Key: "What historical patterns available?"
        v = self.wv(v)  # Value: "What are the actual energy values?"
        
        # Split into multiple heads for parallel attention
        q = self.split_heads(q, batch_size)  # (batch, heads, seq_len, depth)
        k = self.split_heads(k, batch_size)  # (batch, heads, seq_len, depth)
        v = self.split_heads(v, batch_size)  # (batch, heads, seq_len, depth)
        
        # Compute scaled dot-product attention for each head
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Transpose back to combine heads: (batch, seq_len, heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        # Concatenate all heads: (batch, seq_len, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        # Final linear transformation
        output = self.dense(concat_attention)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, q, k, v, mask):
        """
        Core attention mechanism for discovering energy consumption patterns.
        
        Scaled Dot-Product Attention Formula:
        Attention(Q,K,V) = softmax(QKᵀ/√(d_k))V
        
        **Energy Forecasting Interpretation**:
        1. **QKᵀ**: Measures similarity between current query (energy prediction target)
           and historical keys (past energy patterns)
        2. **Scaling**: Prevents attention collapse when model dimension is large
        3. **Softmax**: Creates probability distribution over historical time points
        4. **Weighted V**: Combines historical energy values based on relevance
        
        **Physical Meaning for Energy**:
        - High QKᵀ scores: Historical periods similar to current prediction context
        - Attention weights: Importance of each historical hour for current prediction
        - Weighted values: Energy consumption influenced by relevant historical patterns
        
        **Examples of Learned Patterns**:
        - High attention to same hour previous day (daily cycles)
        - Strong attention during weather transitions (thermal lag effects)
        - Focus on similar occupancy patterns (behavioral consistency)
        - Attention to equipment startup/shutdown times
        
        Args:
            q: Query tensor (what to predict)
            k: Key tensor (historical patterns)
            v: Value tensor (historical energy values)
            mask: Optional mask for causal attention or padding
            
        Returns:
            output: Attention-weighted energy representations
            attention_weights: Temporal importance scores for interpretability
        """
        # Compute attention scores: similarity between queries and keys
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        
        # Scale by square root of key dimension to prevent vanishing gradients
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Apply mask if provided (e.g., causal masking for autoregressive prediction)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  # Mask with large negative values
        
        # Convert to probabilities: each query attends to historical time points
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Apply attention weights to values: weighted sum of historical energy patterns
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights


class PositionalEncoding(layers.Layer):
    """
    Sinusoidal positional encoding for time series transformers.
    
    Positional encoding is crucial for energy forecasting because transformers
    lack inherent understanding of temporal order. This encoding injects
    time-aware information into the model:
    
    **Why Positional Encoding Matters for Energy**:
    1. **Time Awareness**: Distinguishes between "8 AM energy consumption" vs "8 PM"
    2. **Cyclical Patterns**: Sinusoidal functions naturally encode daily/weekly cycles
    3. **Relative Position**: Allows model to understand "2 hours ago" vs "24 hours ago"
    4. **Scale Invariance**: Works for different sequence lengths (hourly, daily data)
    
    **Mathematical Foundation**:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Where:
    - pos: Position in sequence (time step)
    - i: Dimension index
    - d_model: Model dimension
    
    **Energy Forecasting Benefits**:
    - **Daily Cycles**: Sinusoidal patterns naturally align with 24-hour energy cycles
    - **Multi-Scale Patterns**: Different frequencies capture hourly, daily, weekly patterns
    - **Continuous Representation**: Smooth encoding for interpolation between time points
    - **Generalization**: Same encoding works for different buildings and time periods
    
    **Frequency Design**:
    - High frequency components: Capture fine-grained patterns (hourly variations)
    - Low frequency components: Capture coarse patterns (seasonal trends)
    - Mixed frequencies: Enable multi-scale temporal understanding
    """
    
    def __init__(self, position: int, d_model: int, **kwargs):
        """
        Initialize positional encoding for energy time series.
        
        Args:
            position (int): Maximum sequence length (e.g., 168 for week of hourly data)
            d_model (int): Model dimension for encoding
        """
        super().__init__(**kwargs)
        # Pre-compute positional encodings for efficiency
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def get_angles(self, pos, i, d_model):
        """
        Compute angle rates for sinusoidal positional encoding.
        
        Creates different frequencies for different dimensions:
        - Lower dimensions: High frequency (capture fine temporal details)
        - Higher dimensions: Low frequency (capture long-term patterns)
        
        This multi-frequency approach enables the transformer to attend to
        both short-term energy fluctuations and long-term seasonal patterns.
        
        Args:
            pos: Position indices (time steps)
            i: Dimension indices
            d_model: Model dimension
            
        Returns:
            Angle rates for sinusoidal encoding
        """
        # Compute frequency rates: higher dimensions have lower frequencies
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def positional_encoding(self, position, d_model):
        """
        Generate sinusoidal positional encodings for energy time series.
        
        Creates unique encoding for each time position using sine and cosine
        functions at different frequencies. This provides the transformer with
        rich temporal information essential for energy pattern recognition.
        
        **Encoding Strategy**:
        - Even dimensions (0, 2, 4, ...): Use sine functions
        - Odd dimensions (1, 3, 5, ...): Use cosine functions
        - Different frequencies for different dimension pairs
        
        **Energy Application Example**:
        For hourly energy data over 24 hours:
        - High frequency encodings distinguish each hour
        - Medium frequency encodings capture morning/afternoon/evening patterns
        - Low frequency encodings represent daily cycles
        
        Args:
            position: Maximum sequence length
            d_model: Model dimension
            
        Returns:
            Positional encoding matrix of shape (1, position, d_model)
        """
        # Compute angle rates for all position-dimension combinations
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],     # Position indices (time steps)
            np.arange(d_model)[np.newaxis, :],     # Dimension indices
            d_model
        )
        
        # Apply sinusoidal functions with alternating pattern:
        # Even dimensions: sine functions for temporal patterns
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Odd dimensions: cosine functions for complementary temporal information
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        # Add batch dimension for broadcasting
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, x):
        """
        Add positional information to energy time series embeddings.
        
        Combines learned feature representations with positional encodings
        to give the transformer temporal awareness essential for energy forecasting.
        
        **Why Addition Works**:
        - Preserves original energy feature information
        - Injects temporal context without information loss
        - Enables model to learn when patterns occur, not just what patterns exist
        
        **Energy Forecasting Impact**:
        - Model can distinguish "high consumption at 6 PM" from "high consumption at 6 AM"
        - Enables learning of time-dependent energy relationships
        - Supports prediction of cyclical patterns (daily, weekly, seasonal)
        
        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)
            
        Returns:
            Position-aware embeddings for energy pattern recognition
        """
        # Add positional encoding to input embeddings
        # Slice encoding to match current sequence length
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


class TransformerBlock(layers.Layer):
    """
    Complete transformer block for energy time series processing.
    
    A transformer block combines multiple components essential for energy forecasting:
    
    **Architecture Components**:
    1. **Multi-Head Attention**: Discovers temporal relationships in energy consumption
    2. **Feed-Forward Network**: Applies non-linear transformations to attended features
    3. **Layer Normalization**: Stabilizes training and improves convergence
    4. **Residual Connections**: Preserves information flow and enables deep networks
    5. **Dropout Regularization**: Prevents overfitting to specific energy patterns
    
    **Energy Forecasting Workflow**:
    1. **Attention Phase**: "Which historical energy patterns are relevant?"
       - Model attends to similar weather conditions, time periods, occupancy patterns
    2. **Feed-Forward Phase**: "How should I transform these attended patterns?"
       - Non-linear processing to extract complex energy relationships
    3. **Normalization**: Ensures stable gradients for effective learning
    4. **Residual Connection**: Preserves original energy information
    
    **Why This Architecture Works for Energy**:
    - **Information Preservation**: Residual connections maintain energy signal integrity
    - **Stable Training**: Layer normalization prevents gradient problems
    - **Pattern Integration**: Feed-forward networks combine attended temporal patterns
    - **Regularization**: Dropout prevents memorization of specific building patterns
    
    **Mathematical Structure**:
    output = LayerNorm(x + MultiHeadAttention(x))
    output = LayerNorm(output + FeedForward(output))
    """
    
    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1, **kwargs):
        """
        Initialize transformer block for energy forecasting.
        
        Args:
            d_model (int): Model dimension (64-128 typical for energy data)
            num_heads (int): Number of attention heads (8 enables diverse pattern detection)
            dff (int): Feed-forward network dimension (256-512 for complex transformations)
            rate (float): Dropout rate (0.1 for regularization without underfitting)
        """
        super().__init__(**kwargs)
        
        # Multi-head attention for temporal pattern discovery
        self.att = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network for non-linear feature transformation
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)
        
        # Layer normalization for training stability
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, name="attention_norm")
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, name="ffn_norm")
        
        # Dropout for regularization
        self.dropout1 = layers.Dropout(rate, name="attention_dropout")
        self.dropout2 = layers.Dropout(rate, name="ffn_dropout")
    
    def point_wise_feed_forward_network(self, d_model, dff):
        """
        Create feed-forward network for non-linear energy pattern transformation.
        
        Two-layer MLP applied to each time step independently:
        1. **Expansion Layer**: Projects to higher dimension (dff) with ReLU activation
           - Captures complex non-linear relationships in energy data
           - ReLU enables efficient gradient flow and sparse activations
        2. **Compression Layer**: Projects back to model dimension (d_model)
           - Integrates learned non-linear features for next layer
        
        **Energy Domain Applications**:
        - **Weather Processing**: Non-linear temperature-energy relationships
        - **Occupancy Modeling**: Complex human behavior patterns
        - **Equipment Dynamics**: HVAC efficiency curves and thermal lag
        - **Interaction Effects**: Combined impacts of multiple factors
        
        **Architecture Rationale**:
        - dff > d_model: Expansion enables richer feature representations
        - Point-wise: Same transformation applied to each time step
        - ReLU activation: Handles non-negative energy relationships well
        
        Args:
            d_model: Input/output dimension
            dff: Hidden dimension (typically 2-4x d_model)
            
        Returns:
            Sequential feed-forward network
        """
        return tf.keras.Sequential([
            layers.Dense(dff, activation='relu', name="ffn_expansion"),   # Expand and activate
            layers.Dense(d_model, name="ffn_compression")                 # Compress back
        ], name="feed_forward_network")
    
    def call(self, x, training=None, mask=None):
        """
        Process energy time series through transformer block.
        
        **Two-Stage Processing**:
        
        **Stage 1 - Attention and Integration**:
        1. Self-attention discovers temporal relationships in energy data
        2. Dropout regularization prevents overfitting
        3. Residual connection preserves original energy information
        4. Layer normalization stabilizes learning
        
        **Stage 2 - Non-linear Transformation**:
        1. Feed-forward network applies complex transformations
        2. Dropout prevents memorization of specific patterns
        3. Residual connection maintains information flow
        4. Layer normalization ensures stable gradients
        
        **Energy Forecasting Benefits**:
        - **Temporal Awareness**: Attention connects relevant historical periods
        - **Non-linear Processing**: FFN captures complex energy relationships
        - **Information Preservation**: Residuals maintain energy signal integrity
        - **Training Stability**: Layer norms enable deep network training
        
        Args:
            x: Input energy sequence representations
            training: Whether in training mode (affects dropout)
            mask: Optional attention mask
            
        Returns:
            Processed energy representations with enhanced temporal understanding
        """
        # Stage 1: Multi-head attention for temporal pattern discovery
        attn_output, _ = self.att(x, x, x, mask)  # Self-attention on energy sequence
        attn_output = self.dropout1(attn_output, training=training)  # Regularization
        out1 = self.layernorm1(x + attn_output)  # Residual connection + normalization
        
        # Stage 2: Feed-forward transformation for non-linear energy modeling
        ffn_output = self.ffn(out1)  # Non-linear feature transformation
        ffn_output = self.dropout2(ffn_output, training=training)  # Regularization
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection + normalization
        
        return out2


class TimeSeriesTransformer(layers.Layer):
    """
    Complete transformer encoder for energy consumption time series forecasting.
    
    This transformer encoder is specifically designed for building energy forecasting,
    combining multiple transformer blocks to create deep temporal understanding:
    
    **Architecture Overview**:
    1. **Input Embedding**: Projects raw energy features to model dimension
    2. **Positional Encoding**: Adds temporal awareness (daily, weekly cycles)
    3. **Stacked Transformer Blocks**: Progressive feature refinement and pattern discovery
    4. **Dropout Regularization**: Prevents overfitting to specific buildings
    
    **Why Deep Transformers Work for Energy**:
    - **Layer 1**: Basic temporal patterns (hour-to-hour variations)
    - **Layer 2**: Daily patterns (morning startup, evening shutdown)
    - **Layer 3**: Weekly patterns (weekday vs weekend differences)
    - **Layer 4**: Seasonal patterns (heating vs cooling seasons)
    - **Layer 5+**: Complex interactions (weather-occupancy-equipment)
    
    **Energy-Specific Design Choices**:
    - **Encoder-Only**: Focus on understanding patterns rather than generation
    - **Self-Attention**: Discovers relationships within energy consumption history
    - **Deep Architecture**: Captures hierarchical temporal patterns
    - **Residual Connections**: Preserves energy signal through deep network
    
    **Computational Benefits**:
    - **Parallel Processing**: All time steps processed simultaneously
    - **Long-Range Dependencies**: Attention connects distant time points
    - **Interpretability**: Attention weights show important historical periods
    - **Transfer Learning**: Pre-trained patterns can adapt to new buildings
    """
    
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int,
                 maximum_position_encoding: int, rate: float = 0.1, **kwargs):
        """
        Initialize transformer encoder for energy time series.
        
        Args:
            num_layers (int): Number of transformer blocks (4-6 typical for energy)
                             More layers = deeper temporal understanding
            d_model (int): Model dimension (64-128 for energy forecasting)
                          Balances representation power with computational cost
            num_heads (int): Attention heads per block (8 enables diverse pattern types)
            dff (int): Feed-forward dimension (256-512 for complex transformations)
            maximum_position_encoding (int): Max sequence length (168 for weekly data)
            rate (float): Dropout rate (0.1 for energy forecasting regularization)
        """
        super().__init__(**kwargs)
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Input projection layer
        # Transforms raw energy features (consumption, weather, etc.) to model dimension
        self.embedding = layers.Dense(d_model, name="input_embedding")
        
        # Positional encoding for temporal awareness
        # Critical for energy forecasting to understand when patterns occur
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, d_model)
        
        # Stack of transformer blocks for progressive pattern discovery
        self.enc_layers = [
            TransformerBlock(d_model, num_heads, dff, rate, name=f"transformer_block_{i}") 
            for i in range(num_layers)
        ]
        
        # Input dropout for regularization
        self.dropout = layers.Dropout(rate, name="input_dropout")
    
    def call(self, x, training=None, mask=None):
        """
        Process energy time series through complete transformer encoder.
        
        **Processing Pipeline**:
        
        **Step 1 - Input Preparation**:
        - Embed raw energy features into rich representation space
        - Scale embeddings for stable training (standard transformer practice)
        - Add positional encoding for temporal awareness
        - Apply dropout for regularization
        
        **Step 2 - Deep Processing**:
        - Pass through multiple transformer blocks
        - Each layer refines temporal understanding
        - Progressive abstraction from raw energy to high-level patterns
        
        **Energy Forecasting Interpretation**:
        - **Input**: Raw energy consumption + weather + temporal features
        - **Layer 1**: "What are the basic energy patterns?"
        - **Layer 2**: "How do daily cycles work?"
        - **Layer 3**: "What about weekly patterns?"
        - **Layer 4**: "How do seasons and weather interact?"
        - **Output**: Rich temporal representations for forecasting
        
        **Attention Progression**:
        - Early layers: Focus on local temporal patterns
        - Middle layers: Discover daily and weekly cycles
        - Deep layers: Capture long-term seasonal relationships
        
        Args:
            x: Input energy sequence (batch_size, seq_len, features)
            training: Training mode flag (affects dropout and batch norm)
            mask: Optional attention mask for sequence padding
            
        Returns:
            Encoded representations (batch_size, seq_len, d_model)
            Ready for forecasting head or downstream tasks
        """
        seq_len = tf.shape(x)[1]
        
        # Step 1: Input embedding and positional encoding
        x = self.embedding(x)  # Project to model dimension
        
        # Scale embeddings (standard transformer practice for training stability)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        # Add positional information for temporal awareness
        x = self.pos_encoding(x)
        
        # Apply input dropout for regularization
        x = self.dropout(x, training=training)
        
        # Step 2: Deep transformer processing
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        # Return encoded energy sequence representations
        return x  # (batch_size, input_seq_len, d_model)


class TransformerForecaster(BaseForecaster):
    """
    Transformer-based forecaster for building energy consumption prediction.
    
    This implementation adapts the transformer architecture specifically for energy
    forecasting, combining attention mechanisms with domain-specific design choices:
    
    **Why Transformers Excel for Energy Forecasting**:
    
    1. **Long-Range Dependencies**: Energy consumption often depends on events
       hours or days in the past (thermal mass, occupancy patterns, weather lag)
    
    2. **Multi-Scale Patterns**: Different attention heads can simultaneously focus on:
       - Hourly cycles (HVAC operation schedules)
       - Daily patterns (occupancy and lighting schedules)
       - Weekly cycles (weekday vs weekend energy usage)
       - Seasonal trends (heating vs cooling dominant periods)
    
    3. **Variable Importance**: Attention weights reveal which historical periods
       most influence current energy consumption, providing interpretability
    
    4. **Weather Integration**: Can learn complex relationships between weather
       sequences and building energy responses
    
    5. **Building Adaptation**: Self-attention automatically adapts to building-specific
       patterns without manual feature engineering
    
    **Architecture Design for Energy**:
    - **Encoder-Only**: Focuses on understanding patterns rather than autoregressive generation
    - **Global Average Pooling**: Summarizes sequence information for point prediction
    - **Multi-Horizon Output**: Single model predicts multiple future time steps
    - **Residual Connections**: Preserves energy signal through deep network
    
    **Hyperparameter Rationale**:
    - sequence_length=24: Full daily cycle for capturing diurnal patterns
    - d_model=64: Sufficient for energy pattern complexity without overfitting
    - num_heads=8: Enables diverse attention pattern specialization
    - num_layers=4: Balances representation power with computational efficiency
    - dropout_rate=0.1: Conservative regularization for energy data
    """
    
    def __init__(self, 
                 sequence_length: int = 24,
                 prediction_horizon: int = 48,
                 d_model: int = 64,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dff: int = 256,
                 dropout_rate: float = 0.1,
                 learning_rate: float = 0.001):
        """
        Initialize transformer forecaster for energy prediction.
        
        Args:
            sequence_length (int): Input sequence length (24 = daily cycle)
                                  Should match dominant energy pattern cycle
            prediction_horizon (int): Number of future steps to predict
                                    48 = 48-hour ahead forecasting for energy planning
            d_model (int): Model dimension (64 balances capacity with efficiency)
            num_heads (int): Attention heads (8 enables diverse pattern detection)
            num_layers (int): Transformer blocks (4 layers capture multi-scale patterns)
            dff (int): Feed-forward dimension (256 for complex transformations)
            dropout_rate (float): Regularization strength (0.1 prevents overfitting)
            learning_rate (float): Optimizer step size (0.001 for stable training)
        """
        super().__init__("Transformer")
        
        # Sequence and prediction parameters
        self.sequence_length = sequence_length      # Historical window for prediction
        self.prediction_horizon = prediction_horizon # Future steps to forecast
        
        # Model architecture parameters
        self.d_model = d_model                      # Model dimension
        self.num_heads = num_heads                  # Attention heads per layer
        self.num_layers = num_layers                # Transformer blocks
        self.dff = dff                              # Feed-forward dimension
        
        # Training parameters
        self.dropout_rate = dropout_rate            # Regularization strength
        self.learning_rate = learning_rate          # Optimization step size
        
        # Model will be built during first fit() call
        self.model = None
    
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build transformer architecture optimized for energy consumption forecasting.
        
        **Model Architecture Design**:
        
        **Stage 1 - Input Processing**:
        - Input layer accepts energy time series (consumption, weather, temporal features)
        - Shape: (batch_size, sequence_length, n_features)
        
        **Stage 2 - Transformer Encoder**:
        - Multi-layer transformer processes temporal patterns
        - Attention mechanisms discover energy consumption relationships
        - Positional encoding provides temporal awareness
        
        **Stage 3 - Sequence Aggregation**:
        - Global average pooling summarizes sequence information
        - Alternative: Use last token or learnable CLS token
        - Reduces (batch_size, seq_len, d_model) to (batch_size, d_model)
        
        **Stage 4 - Forecasting Head**:
        - Dense layer with ReLU for non-linear energy relationships
        - Dropout for regularization
        - Final linear layer outputs multi-horizon predictions
        
        **Energy-Specific Design Choices**:
        - **MSE Loss**: Appropriate for continuous energy values
        - **MAE Metric**: Interpretable error in energy units (kWh)
        - **Adam Optimizer**: Adaptive learning rates for complex energy patterns
        - **Multi-Step Output**: Single model predicts entire forecast horizon
        
        Args:
            input_shape: (sequence_length, n_features) for energy time series
        """
        # Stage 1: Input definition
        inputs = Input(shape=input_shape, name='energy_sequence_input')
        
        # Stage 2: Transformer encoder for temporal pattern discovery
        transformer = TimeSeriesTransformer(
            num_layers=self.num_layers,                    # Deep representation learning
            d_model=self.d_model,                          # Model dimension
            num_heads=self.num_heads,                      # Parallel attention patterns
            dff=self.dff,                                  # Feed-forward capacity
            maximum_position_encoding=self.sequence_length, # Temporal encoding length
            rate=self.dropout_rate                         # Regularization strength
        )
        
        # Apply transformer encoding
        transformer_output = transformer(inputs)  # (batch_size, seq_len, d_model)
        
        # Stage 3: Sequence aggregation for point prediction
        # Global average pooling summarizes entire sequence information
        pooled = layers.GlobalAveragePooling1D(name='sequence_aggregation')(transformer_output)
        
        # Stage 4: Forecasting head for energy prediction
        # Non-linear transformation for complex energy relationships
        dense = layers.Dense(
            self.dff, 
            activation='relu', 
            name='forecasting_transformation'
        )(pooled)
        dense = layers.Dropout(self.dropout_rate, name='forecasting_dropout')(dense)
        
        # Multi-horizon energy consumption prediction
        outputs = layers.Dense(
            self.prediction_horizon, 
            name='energy_forecast_output',
            activation='linear'  # Linear for regression (energy consumption values)
        )(dense)
        
        # Create complete model
        self.model = Model(
            inputs=inputs, 
            outputs=outputs, 
            name='EnergyTransformerForecaster'
        )
        
        # Compile with energy forecasting objectives
        self.model.compile(
            optimizer=Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,      # Momentum for smooth convergence
                beta_2=0.999,    # Second moment estimation
                epsilon=1e-7     # Numerical stability
            ),
            loss='mse',          # Mean Squared Error for energy regression
            metrics=['mae']      # Mean Absolute Error for interpretable monitoring
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: int = 0) -> None:
        """
        Train transformer model on building energy consumption patterns.
        
        **Transformer Training for Energy Forecasting**:
        
        **Data Preparation**:
        - Automatically handles 2D input by adding feature dimension
        - Ensures consistent 3D format: (samples, sequence_length, features)
        - Critical for proper attention mechanism operation
        
        **Training Strategy**:
        - **Early Stopping**: Prevents overfitting to specific building patterns
        - **Learning Rate Scheduling**: Adapts learning rate based on validation performance
        - **Validation Monitoring**: Uses validation loss to guide training decisions
        
        **Why This Training Approach Works for Energy**:
        
        1. **Patience=10**: Energy patterns are complex, need time to learn
        2. **LR Reduction**: Helps fine-tune attention patterns in later epochs
        3. **Best Weight Restoration**: Ensures optimal energy forecasting performance
        4. **Batch Processing**: Enables efficient parallel attention computation
        
        **Attention Learning Process**:
        - Early epochs: Learn basic temporal patterns
        - Middle epochs: Discover daily and weekly cycles
        - Late epochs: Refine complex weather-energy relationships
        
        **Energy-Specific Considerations**:
        - Transformers excel with sufficient training data (weeks/months of hourly data)
        - Attention patterns become more interpretable with longer training
        - Validation prevents memorization of specific building characteristics
        
        Args:
            X_train: Energy sequences (samples, seq_len, features)
                    Features can include: consumption history, weather, time encoding
            y_train: Target energy values (samples, prediction_horizon)
            X_val: Validation sequences for overfitting prevention
            y_val: Validation targets for performance monitoring
            epochs: Maximum training iterations (100 typically sufficient)
            batch_size: Parallel sequences processed (32 balances memory and gradients)
            verbose: Training progress display level
        """
        
        # Ensure proper 3D input format for transformer
        if len(X_train.shape) == 2:
            # Add feature dimension: (samples, seq_len) -> (samples, seq_len, 1)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        # Build model architecture if not already constructed
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])  # (seq_len, features)
            self._build_model(input_shape)
        
        # Prepare validation data with same format
        validation_data = None
        if X_val is not None and y_val is not None:
            if len(X_val.shape) == 2:
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            validation_data = (X_val, y_val)
        
        # Energy forecasting training callbacks
        callbacks = [
            EarlyStopping(
                patience=10,                    # Allow time for complex pattern learning
                restore_best_weights=True,      # Keep best energy forecasting performance
                monitor='val_loss',             # Use validation loss for decisions
                verbose=1 if verbose > 0 else 0
            ),
            ReduceLROnPlateau(
                factor=0.5,                     # Halve learning rate when stuck
                patience=5,                     # Wait 5 epochs before reduction
                min_lr=1e-7,                    # Minimum learning rate threshold
                monitor='val_loss',             # Monitor validation performance
                verbose=1 if verbose > 0 else 0
            )
        ]
        
        # Train transformer on energy consumption patterns
        self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks
        )
        
        # Mark as trained and ready for energy predictions
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate energy consumption forecasts using transformer attention patterns.
        
        **Transformer Prediction Process**:
        
        **Stage 1 - Input Processing**:
        - Ensures proper 3D format for attention mechanisms
        - Maintains consistency with training data format
        
        **Stage 2 - Attention-Based Encoding**:
        - Multi-head attention discovers relevant historical patterns
        - Different heads focus on different energy consumption patterns:
          * Daily cycles, weekly patterns, weather relationships
        - Positional encoding provides temporal context
        
        **Stage 3 - Sequence Aggregation**:
        - Global average pooling summarizes attended patterns
        - Captures overall energy consumption trends
        
        **Stage 4 - Multi-Horizon Forecasting**:
        - Dense layers transform aggregated features
        - Outputs predictions for entire forecast horizon
        
        **Energy Forecasting Advantages**:
        
        1. **Interpretable Attention**: Can analyze which historical periods
           influence predictions (e.g., same hour yesterday, weather events)
        
        2. **Long-Range Dependencies**: Connects energy consumption to events
           hours or days in the past (thermal mass, occupancy patterns)
        
        3. **Multi-Scale Patterns**: Simultaneously considers short-term
           fluctuations and long-term seasonal trends
        
        4. **Robust Predictions**: Attention mechanisms handle missing data
           and irregular patterns better than recurrent models
        
        Args:
            X: Input energy sequences (samples, seq_len, features)
               Should match training data format and scaling
               
        Returns:
            Multi-horizon energy consumption predictions
            Shape: (samples, prediction_horizon)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Ensure proper input format for transformer attention
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Generate energy forecasts using learned attention patterns
        predictions = self.model.predict(X)
        
        # Handle single-step prediction case
        if self.prediction_horizon == 1:
            predictions = predictions[:, 0:1]
        
        return predictions
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained transformer model with learned attention patterns.
        
        Preserves complete model state including:
        - Transformer architecture and weights
        - Learned attention patterns for energy forecasting
        - Position encodings for temporal awareness
        - Forecasting head parameters
        
        Essential for deploying energy forecasting models in production
        where attention-based predictions provide interpretability.
        
        Args:
            filepath: Path to save the complete model
        """
        if self.model is not None:
            self.model.save(filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load pre-trained transformer model for energy forecasting.
        
        Restores complete transformer state including custom components
        necessary for attention-based energy prediction.
        
        Args:
            filepath: Path to the saved transformer model
        """
        self.model = tf.keras.models.load_model(filepath, custom_objects={
            'TimeSeriesTransformer': TimeSeriesTransformer,
            'TransformerBlock': TransformerBlock,
            'MultiHeadAttention': MultiHeadAttention,
            'PositionalEncoding': PositionalEncoding
        })
        self.is_fitted = True


class TimesFMInspiredForecaster(BaseForecaster):
    """
    TimesFM-Inspired Foundation Model for Building Energy Forecasting.
    
    This implementation adapts concepts from Google's TimesFM (Time Series Foundation Model)
    for building energy consumption prediction, incorporating foundation model principles:
    
    **TimesFM Foundation Model Concepts**:
    
    1. **Patch-Based Processing**: Divides time series into patches (like vision transformers)
       - Reduces sequence length for computational efficiency
       - Enables processing of very long energy consumption histories
       - Natural aggregation of sub-hourly measurements into hourly patches
    
    2. **Pre-Training Capability**: Architecture designed for large-scale pre-training
       - Can learn from multiple buildings before fine-tuning
       - Captures universal energy consumption patterns
       - Enables few-shot learning for new buildings
    
    3. **Zero-Shot Generalization**: Foundation model can work on unseen buildings
       - Learned patterns transfer across different building types
       - Reduces need for building-specific training data
       - Enables rapid deployment to new facilities
    
    4. **Flexible Input Lengths**: Can handle variable sequence lengths
       - Adapts to different data availability scenarios
       - Works with incomplete historical data
       - Scales from short-term to long-term forecasting
    
    **Energy-Specific Adaptations**:
    
    **Patch Design for Energy Data**:
    - **4-Hour Patches**: Natural aggregation of hourly energy measurements
    - **Seasonal Patching**: Can group daily patterns for seasonal analysis
    - **Multi-Resolution**: Different patch sizes for different time scales
    
    **Foundation Model Training**:
    - **Multi-Building Pre-training**: Learn universal energy patterns
    - **Masked Language Modeling**: Predict missing energy values
    - **Contrastive Learning**: Distinguish similar vs different energy patterns
    
    **Architecture Enhancements**:
    - **Deeper Network**: 6 layers for complex pattern hierarchy
    - **Larger Model**: 128 dimensions for richer representations
    - **Advanced Forecasting Head**: Multi-layer prediction with GELU activation
    - **Robust Loss**: Huber loss for outlier resilience
    
    **Advantages for Energy Forecasting**:
    1. **Scalability**: Processes long sequences efficiently via patching
    2. **Transferability**: Pre-trained patterns work across buildings
    3. **Robustness**: Handles missing data and irregular patterns
    4. **Interpretability**: Patch-level attention shows energy event importance
    5. **Efficiency**: Reduced computational cost through patch aggregation
    """
    
    def __init__(self, 
                 sequence_length: int = 24,
                 prediction_horizon: int = 48,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 patch_size: int = 4,
                 learning_rate: float = 0.0001):
        """
        Initialize TimesFM-inspired forecaster for energy prediction.
        
        Args:
            sequence_length (int): Input sequence length (24 hours for daily patterns)
            prediction_horizon (int): Forecast horizon (48 hours for long-term planning)
            d_model (int): Model dimension (128 for rich foundation model representations)
            num_heads (int): Attention heads (8 for diverse pattern specialization)
            num_layers (int): Transformer layers (6 for deep hierarchical learning)
            patch_size (int): Time series patch size (4 = 4-hour energy aggregation)
                             Reduces sequence length by factor of patch_size
            learning_rate (float): Learning rate (0.0001 for stable foundation model training)
        
        **Patch Size Rationale**:
        - patch_size=4: Groups hourly data into 4-hour energy periods
        - Captures natural energy patterns (morning startup, midday peak, evening shutdown)
        - Reduces 24-hour sequence to 6 patches for efficient processing
        - Enables modeling of longer historical contexts
        """
        super().__init__("TimesFM")
        
        # Sequence and forecasting parameters
        self.sequence_length = sequence_length      # Original sequence length
        self.prediction_horizon = prediction_horizon # Future steps to predict
        
        # Foundation model architecture parameters
        self.d_model = d_model                      # Larger model for foundation capabilities
        self.num_heads = num_heads                  # Multi-head attention
        self.num_layers = num_layers                # Deeper network for complex patterns
        
        # TimesFM-specific parameters
        self.patch_size = patch_size                # Time series patching for efficiency
        self.learning_rate = learning_rate          # Conservative LR for stable training
        
        # Model will be built during training
        self.model = None
    
    def _create_patches(self, x):
        """
        Create TimesFM-style patches from energy time series.
        
        **Patching Strategy for Energy Data**:
        
        Converts sequential hourly energy data into patches representing
        multi-hour energy consumption periods:
        
        **Input**: [hour1, hour2, hour3, hour4, hour5, hour6, ...]
        **Output**: [[hour1-4], [hour5-8], [hour9-12], ...]
        
        **Energy Patch Interpretation**:
        - **Patch 1**: Morning energy pattern (6 AM - 10 AM)
        - **Patch 2**: Midday energy pattern (10 AM - 2 PM)
        - **Patch 3**: Afternoon energy pattern (2 PM - 6 PM)
        - **Patch 4**: Evening energy pattern (6 PM - 10 PM)
        - **Patch 5**: Night energy pattern (10 PM - 2 AM)
        - **Patch 6**: Late night pattern (2 AM - 6 AM)
        
        **Benefits for Energy Forecasting**:
        1. **Computational Efficiency**: Reduces sequence length by patch_size factor
        2. **Natural Aggregation**: Aligns with energy management time periods
        3. **Pattern Recognition**: Each patch represents distinct operational phase
        4. **Long Context**: Enables processing of longer energy histories
        5. **Multi-Resolution**: Can use different patch sizes for different applications
        
        **Padding Strategy**:
        - Zero-padding ensures consistent patch dimensions
        - Minimal impact on energy patterns (padding typically < 10% of sequence)
        - Maintains temporal order and relationships
        
        Args:
            x: Input energy sequences (batch_size, seq_len, features)
            
        Returns:
            Patched sequences (batch_size, num_patches, patch_size * features)
        """
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        features = tf.shape(x)[2]
        
        # Calculate required padding for even patch division
        padded_len = ((seq_len + self.patch_size - 1) // self.patch_size) * self.patch_size
        
        # Apply zero-padding if sequence length not divisible by patch size
        if seq_len < padded_len:
            padding = tf.zeros([batch_size, padded_len - seq_len, features])
            x = tf.concat([x, padding], axis=1)
        
        # Reshape into patches: group consecutive time steps
        num_patches = padded_len // self.patch_size
        patches = tf.reshape(x, [batch_size, num_patches, self.patch_size * features])
        
        return patches
    
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build TimesFM-inspired foundation model architecture.
        
        **Foundation Model Architecture Pipeline**:
        
        **Stage 1 - Patch Creation**:
        - Transform sequential energy data into patches
        - Reduces computational complexity while preserving patterns
        - Natural alignment with energy management time periods
        
        **Stage 2 - Patch Embedding**:
        - Project patches to foundation model dimension (128-512)
        - Larger dimension enables richer pattern representations
        - Foundation models benefit from high-capacity embeddings
        
        **Stage 3 - Positional Encoding**:
        - Add temporal awareness to patch sequences
        - Scaled for patch-level rather than time-step level
        - Enables understanding of energy pattern sequences
        
        **Stage 4 - Deep Transformer Processing**:
        - 6+ layers for hierarchical pattern learning
        - Each layer refines understanding of energy consumption patterns
        - Attention patterns become increasingly sophisticated
        
        **Stage 5 - Advanced Forecasting Head**:
        - Multi-layer dense network with GELU activation
        - GELU provides smoother gradients than ReLU
        - Progressive dimension reduction for focused predictions
        
        **TimesFM Design Principles**:
        - **Scale**: Larger model for better generalization
        - **Robustness**: Huber loss handles energy data outliers
        - **Efficiency**: Patch processing reduces computational cost
        - **Transfer Learning**: Architecture supports pre-training
        
        Args:
            input_shape: (sequence_length, features) for energy time series
        """
        # Stage 1: Input definition
        inputs = Input(shape=input_shape, name='energy_timesfm_input')
        
        # Stage 2: TimesFM-style patch creation
        patches = layers.Lambda(
            self._create_patches, 
            name='energy_patching'
        )(inputs)
        
        # Stage 3: Patch embedding to foundation model dimension
        patch_embedding = layers.Dense(
            self.d_model, 
            name='patch_embedding',
            activation='linear'  # Linear projection maintains patch information
        )(patches)
        
        # Stage 4: Positional encoding for patch sequences
        # Note: Position count is num_patches, not original sequence length
        pos_encoding = PositionalEncoding(
            self.sequence_length // self.patch_size,  # Number of patches
            self.d_model
        )
        encoded = pos_encoding(patch_embedding)
        
        # Stage 5: Deep transformer processing (foundation model depth)
        x = encoded
        for i in range(self.num_layers):
            transformer_block = TransformerBlock(
                self.d_model,                    # Foundation model dimension
                self.num_heads,                  # Multi-head attention
                self.d_model * 4,                # Large FFN for complex patterns
                rate=0.1,                       # Conservative dropout
                name=f'timesfm_block_{i}'
            )
            x = transformer_block(x)
        
        # Stage 6: Global context aggregation
        global_context = layers.GlobalAveragePooling1D(
            name='global_energy_context'
        )(x)
        
        # Stage 7: Advanced forecasting head (TimesFM-style)
        # First forecasting layer with GELU activation
        forecast_head = layers.Dense(
            512, 
            activation='gelu',               # Smoother activation for foundation models
            name='forecast_layer_1'
        )(global_context)
        forecast_head = layers.Dropout(0.1, name='forecast_dropout_1')(forecast_head)
        
        # Second forecasting layer for pattern refinement
        forecast_head = layers.Dense(
            256, 
            activation='gelu',
            name='forecast_layer_2'
        )(forecast_head)
        forecast_head = layers.Dropout(0.1, name='forecast_dropout_2')(forecast_head)
        
        # Final prediction layer
        outputs = layers.Dense(
            self.prediction_horizon, 
            name='energy_forecast_output',
            activation='linear'              # Linear for energy regression
        )(forecast_head)
        
        # Create TimesFM-inspired model
        self.model = Model(
            inputs=inputs, 
            outputs=outputs, 
            name='TimesFMEnergyForecaster'
        )
        
        # Foundation model optimizer configuration
        optimizer = Adam(
            learning_rate=self.learning_rate,   # Conservative for foundation model
            beta_1=0.9,                        # Standard momentum
            beta_2=0.999,                      # Standard second moment
            epsilon=1e-8                       # Numerical stability
        )
        
        # Compile with robust loss function
        self.model.compile(
            optimizer=optimizer,
            loss='huber',                      # Robust to energy data outliers
            metrics=['mae']                    # Interpretable energy error metric
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 150,
            batch_size: int = 64,
            verbose: int = 0) -> None:
        """Fit the TimesFM-inspired model."""
        
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self._build_model(input_shape)
        
        validation_data = None
        if X_val is not None and y_val is not None:
            if len(X_val.shape) == 2:
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            validation_data = (X_val, y_val)
        
        # TimesFM-style training with longer patience
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.3, patience=10, min_lr=1e-8, monitor='val_loss')
        ]
        
        self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks
        )
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with TimesFM-inspired model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        predictions = self.model.predict(X)
        
        if self.prediction_horizon == 1:
            predictions = predictions[:, 0:1]
        
        return predictions
    
    def save_model(self, filepath: str) -> None:
        """Save TimesFM model."""
        if self.model is not None:
            self.model.save(filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load TimesFM model."""
        self.model = tf.keras.models.load_model(filepath, custom_objects={
            'TimeSeriesTransformer': TimeSeriesTransformer,
            'TransformerBlock': TransformerBlock,
            'MultiHeadAttention': MultiHeadAttention,
            'PositionalEncoding': PositionalEncoding
        })
        self.is_fitted = True


def get_transformer_forecasters() -> dict:
    """
    Get dictionary of transformer-based forecasting models for energy prediction.
    
    Provides state-of-the-art attention-based models specifically configured
    for building energy consumption forecasting:
    
    **Available Models**:
    
    1. **TransformerForecaster**: Standard transformer encoder architecture
       - Multi-head attention for temporal pattern discovery
       - Ideal for: Medium-term forecasting with clear temporal patterns
       - Strengths: Interpretable attention, robust performance
       - Use cases: Daily/weekly energy planning, HVAC optimization
    
    2. **TimesFMInspiredForecaster**: Foundation model approach
       - Patch-based processing for computational efficiency
       - Deeper architecture for complex pattern hierarchies
       - Ideal for: Long-term forecasting and transfer learning
       - Strengths: Scalability, pre-training capability
       - Use cases: Monthly/seasonal planning, multi-building deployment
    
    **Model Selection Guidelines**:
    
    **Choose TransformerForecaster when**:
    - Need interpretable attention patterns
    - Working with moderate sequence lengths (24-168 hours)
    - Require fast training and inference
    - Building-specific model deployment
    
    **Choose TimesFMInspiredForecaster when**:
    - Processing very long sequences (weeks/months)
    - Planning to pre-train on multiple buildings
    - Need maximum forecasting performance
    - Deploying to many similar buildings
    
    Returns:
        Dict[str, BaseForecaster]: Dictionary mapping model names to instances
                                  Ready for training on energy consumption data
    
    Example Usage:
        ```python
        transformers = get_transformer_forecasters()
        
        # Standard transformer for interpretable forecasting
        standard_model = transformers['Transformer']
        standard_model.fit(X_train, y_train)
        
        # Foundation model for maximum performance
        foundation_model = transformers['TimesFM']
        foundation_model.fit(X_train, y_train)
        ```
    """
    return {
        'Transformer': TransformerForecaster(),       # Standard attention-based forecasting
        'TimesFM': TimesFMInspiredForecaster()        # Foundation model approach
    }