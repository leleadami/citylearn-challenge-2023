"""
Data utility functions for CityLearn project.

This module provides comprehensive data processing utilities specifically designed
for the CityLearn Challenge 2023, focusing on building energy forecasting and
optimization. The CityLearn Challenge involves predicting and controlling energy
consumption, storage, and generation across multiple buildings in a smart city
neighborhood.

Key functionalities:
1. Data loading from CityLearn CSV format with standardized building schemas
2. Time series sequence creation for forecasting models (LSTM, Transformers)
3. Cross-building generalization testing for model robustness
4. Neighborhood-level aggregation for district energy management
5. Standardized preprocessing pipeline with proper scaling and normalization

The CityLearn data structure includes:
- Building-specific data: electrical loads, solar generation, battery states
- Environmental data: weather conditions, carbon intensity, pricing
- Time series format: hourly measurements with seasonal patterns

Design principles:
- Maintain temporal order in splits to avoid data leakage
- Preserve building-specific characteristics for personalized models
- Enable cross-building evaluation to test generalization
- Support both individual building and neighborhood-level analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Optional
import os


def load_building_data(data_path: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Load building data from CityLearn Challenge CSV files.
    
    This function handles the specific data structure of the CityLearn Challenge,
    which organizes data into multiple phases with standardized building schemas.
    Each building CSV contains time series data for energy consumption, generation,
    and storage systems.
    
    CityLearn data characteristics:
    - Hourly time series data (8760 hours per year)
    - Multiple buildings with different energy profiles
    - Standardized column names across buildings
    - Additional environmental data (weather, carbon intensity, pricing)
    
    Data quality considerations:
    - Handles missing files gracefully with fallback mechanisms
    - Ensures consistent data loading across different phases
    - Maintains temporal order crucial for time series analysis
    
    Why this loading approach:
    - Supports both Phase 1 and Phase 2 data structures
    - Automatically detects CityLearn challenge format
    - Loads all relevant environmental variables
    - Returns organized dictionary for easy building-specific access
    
    Args:
        data_path: Path to data directory containing CityLearn challenge data
        
    Returns:
        Dictionary with building names as keys and dataframes as values.
        Also includes environmental data (weather, carbon_intensity, pricing)
        
    Example:
        >>> data = load_building_data("data/citylearn_challenge_2023_phase_1")
        >>> print(data.keys())  # ['Building_1', 'Building_2', 'Building_3', 'weather', ...]
        >>> print(data['Building_1'].columns)  # Energy-related columns
    """
    building_data = {}
    
    # Look for CityLearn challenge data structure
    # The CityLearn Challenge organizes data in phase-specific directories
    # This approach ensures compatibility with different challenge phases
    if os.path.exists(os.path.join(data_path, "citylearn_challenge_2023_phase_1")):
        phase1_path = os.path.join(data_path, "citylearn_challenge_2023_phase_1")
        
        # Load building-specific data files
        # Building files follow the naming convention 'Building_X.csv'
        # Each building has the same schema but different energy patterns
        building_files = [f for f in os.listdir(phase1_path) if f.startswith('Building_') and f.endswith('.csv')]
        
        for file in building_files:
            building_name = file.replace('.csv', '')
            file_path = os.path.join(phase1_path, file)
            # Load with pandas, preserving temporal order
            building_data[building_name] = pd.read_csv(file_path)
        
        # Load additional environmental data files
        # These provide context for building energy patterns:
        # - carbon_intensity: Grid carbon emissions (affects optimization objectives)
        # - pricing: Electricity prices (influences cost-based decisions)
        # - weather: Temperature, humidity, solar radiation (key forecasting features)
        additional_files = ['carbon_intensity.csv', 'pricing.csv', 'weather.csv']
        for file in additional_files:
            file_path = os.path.join(phase1_path, file)
            if os.path.exists(file_path):
                file_name = file.replace('.csv', '')
                building_data[file_name] = pd.read_csv(file_path)
    else:
        # Fallback mechanism for processed or custom data formats
        # Handles cases where data has been preprocessed or stored differently
        for file in os.listdir(data_path):
            if file.endswith('_raw.csv'):
                building_name = file.replace('_raw.csv', '')
                file_path = os.path.join(data_path, file)
                building_data[building_name] = pd.read_csv(file_path)
    
    return building_data


def create_sequences(data: np.ndarray, 
                    sequence_length: int, 
                    prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.
    
    This function implements the sliding window approach essential for time series
    forecasting with deep learning models (LSTM, GRU, Transformers). It transforms
    continuous time series data into supervised learning format.
    
    Why sequence creation is critical for CityLearn:
    - Building energy consumption exhibits temporal dependencies
    - Historical patterns inform future energy needs
    - Enables models to learn seasonal and daily cycles
    - Supports multi-step ahead forecasting (48-hour horizon in CityLearn)
    
    Sequence design considerations:
    - Sequence length should capture relevant temporal patterns
      * 24 hours: Daily cycles (typical choice for energy data)
      * 168 hours: Weekly cycles including weekends
      * Longer sequences: Seasonal patterns but higher computational cost
    
    - Prediction horizon affects model complexity:
      * Single-step (1 hour): Easier to learn, higher accuracy
      * Multi-step (48 hours): CityLearn requirement, more challenging
      * Longer horizons: Useful for planning but lower accuracy
    
    Mathematical formulation:
    For time series [x1, x2, ..., xT], creates:
    X = [[x1, x2, ..., x_seq_len], [x2, x3, ..., x_seq_len+1], ...]
    y = [[x_seq_len+1, ..., x_seq_len+horizon], [x_seq_len+2, ..., x_seq_len+horizon+1], ...]
    
    Args:
        data: Input time series data (1D array of energy measurements)
        sequence_length: Length of input sequences (lookback window)
        prediction_horizon: Number of future steps to predict (CityLearn uses 48)
        
    Returns:
        Tuple of (X, y) arrays where:
        - X: Input sequences of shape (n_samples, sequence_length)
        - y: Target sequences of shape (n_samples, prediction_horizon)
        
    Example:
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> X, y = create_sequences(data, sequence_length=3, prediction_horizon=2)
        >>> print(X[0])  # [1, 2, 3]
        >>> print(y[0])  # [4, 5]
    """
    X, y = [], []
    
    # Sliding window approach to create training sequences
    # Loop ensures no future data leakage into input sequences
    # The range calculation ensures we have complete sequences for both input and target
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        # Input sequence: historical data points
        # Used by the model to learn temporal patterns
        X.append(data[i:i+sequence_length])
        
        # Target sequence: future data points to predict
        # Starts immediately after input sequence ends
        y.append(data[i+sequence_length:i+sequence_length+prediction_horizon])
        
    return np.array(X), np.array(y)


def prepare_forecasting_data(building_data: Dict[str, pd.DataFrame],
                           target_column: str,
                           sequence_length: int = 24,
                           prediction_horizon: int = 48,  # CityLearn Challenge requirement
                           train_ratio: float = 0.6,
                           val_ratio: float = 0.2,
                           normalize: bool = True) -> Dict[str, Dict]:
    """
    Prepare data for time series forecasting.
    
    This function implements the complete preprocessing pipeline for CityLearn
    forecasting tasks. It handles the transformation from raw building data
    to model-ready sequences with proper temporal splits and normalization.
    
    Key preprocessing decisions and rationales:
    
    1. Temporal data splitting (not random):
       - Maintains chronological order essential for time series
       - Avoids data leakage from future to past
       - Reflects real-world deployment scenario
       - Split ratios: 60% train, 20% validation, 20% test (typical for time series)
    
    2. Sequence length selection (default 24 hours):
       - Captures daily energy consumption patterns
       - Includes peak and off-peak periods
       - Balances model complexity vs. computational efficiency
       - Can be adjusted for weekly (168h) or shorter patterns
    
    3. Prediction horizon (48 hours for CityLearn):
       - Matches CityLearn Challenge requirements
       - Enables 2-day ahead planning for energy systems
       - Sufficient for battery optimization and demand response
    
    4. Normalization strategy:
       - StandardScaler for Gaussian-like distributions
       - Prevents gradient vanishing in deep networks
       - Fitted only on training data to avoid data leakage
       - Applied to both input sequences and targets
    
    5. Building-specific processing:
       - Each building processed independently
       - Preserves building-specific energy patterns
       - Enables personalized model training
       - Supports cross-building generalization testing
    
    Args:
        building_data: Dictionary of building dataframes from load_building_data()
        target_column: Column to predict (e.g., 'Electrical_Demand', 'Solar_Generation')
        sequence_length: Length of input sequences (hours of historical data)
        prediction_horizon: Number of future steps to predict (CityLearn requires 48)
        train_ratio: Ratio of training data (chronologically first portion)
        val_ratio: Ratio of validation data (middle portion)
        normalize: Whether to normalize data (recommended for neural networks)
        
    Returns:
        Dictionary with prepared data for each building containing:
        - Normalized sequences: X_train, X_val, X_test, y_train, y_val, y_test
        - Scalers: scaler_X, scaler_y (for inverse transformation)
        - Raw data: Original sequences before normalization
        
    Example:
        >>> data = load_building_data()
        >>> prepared = prepare_forecasting_data(data, 'Electrical_Demand')
        >>> print(prepared['Building_1']['X_train'].shape)  # (n_samples, 24)
        >>> print(prepared['Building_1']['y_train'].shape)  # (n_samples, 48)
    """
    prepared_data = {}
    
    # Process each building independently to preserve building-specific patterns
    for building_name, df in building_data.items():
        # Skip non-building data (weather, pricing, etc.)
        if not building_name.startswith('Building_'):
            continue
            
        # Extract target variable (e.g., electrical demand, solar generation)
        # Ensure the target column exists to avoid KeyError
        if target_column not in df.columns:
            print(f"Warning: {target_column} not found in {building_name}, skipping...")
            continue
            
        target_data = df[target_column].values
        
        # Create sequences using sliding window approach
        # This transforms continuous time series into supervised learning format
        X, y = create_sequences(target_data, sequence_length, prediction_horizon)
        
        # Temporal data splitting (crucial for time series)
        # Unlike random splits, this preserves chronological order
        # Training on past, validating on recent past, testing on most recent
        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # Chronological splits ensure realistic evaluation
        X_train = X[:train_end]        # Earliest data for training
        X_val = X[train_end:val_end]   # Middle data for hyperparameter tuning
        X_test = X[val_end:]           # Latest data for final evaluation
        
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
        
        # Normalization strategy for neural network training
        # StandardScaler: zero mean, unit variance (assumes Gaussian distribution)
        # Critical: fit only on training data to prevent data leakage
        scaler_X = StandardScaler() if normalize else None
        scaler_y = StandardScaler() if normalize else None
        
        if normalize:
            # Fit scalers ONLY on training data to avoid data leakage
            # Reshape for sklearn: (n_samples * sequence_length, n_features)
            # For univariate time series, n_features = 1
            X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            
            # Fit target scaler on training targets
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, y_train.shape[-1]))
            y_train_scaled = y_train_scaled.reshape(y_train.shape)
            
            # Transform (not fit) validation and test data using training statistics
            # This ensures validation/test data uses same scaling as training
            X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1]))
            X_val_scaled = X_val_scaled.reshape(X_val.shape)
            
            X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1]))
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
            
            y_val_scaled = scaler_y.transform(y_val.reshape(-1, y_val.shape[-1]))
            y_val_scaled = y_val_scaled.reshape(y_val.shape)
            
            y_test_scaled = scaler_y.transform(y_test.reshape(-1, y_test.shape[-1]))
            y_test_scaled = y_test_scaled.reshape(y_test.shape)
        else:
            # Use original data without normalization
            # Some models (tree-based) work well without normalization
            X_train_scaled = X_train
            X_val_scaled = X_val
            X_test_scaled = X_test
            y_train_scaled = y_train
            y_val_scaled = y_val
            y_test_scaled = y_test
        
        # Store both normalized and raw data for flexibility
        # Normalized data: ready for neural network training
        # Raw data: needed for evaluation metrics and visualization
        # Scalers: required for inverse transformation of predictions
        prepared_data[building_name] = {
            # Normalized sequences ready for model training
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_scaled,
            'y_val': y_val_scaled,
            'y_test': y_test_scaled,
            
            # Scalers for inverse transformation
            # Essential for converting predictions back to original scale
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            
            # Raw data for evaluation and analysis
            # Metrics should be calculated on original scale
            'raw_data': {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }
        }
    
    return prepared_data


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics for energy forecasting evaluation.
    
    This function computes standard metrics used in energy forecasting research
    and the CityLearn Challenge. Each metric provides different insights into
    model performance and forecasting quality.
    
    Metrics explanation and interpretation:
    
    1. MAE (Mean Absolute Error):
       - Average absolute difference between predictions and actual values
       - Units: Same as target variable (e.g., kWh)
       - Interpretation: Average forecasting error magnitude
       - Robust to outliers, easy to interpret
       - Lower is better
    
    2. MSE (Mean Squared Error):
       - Average squared difference between predictions and actual values
       - Units: Square of target variable (e.g., kWh²)
       - Interpretation: Emphasizes larger errors more than MAE
       - Sensitive to outliers, used in optimization
       - Lower is better
    
    3. RMSE (Root Mean Squared Error):
       - Square root of MSE, returns to original units
       - Units: Same as target variable (e.g., kWh)
       - Interpretation: Standard deviation of forecasting errors
       - More interpretable than MSE, commonly reported
       - Lower is better
    
    4. R² (R-squared, Coefficient of Determination):
       - Proportion of variance in target explained by predictions
       - Range: (-∞, 1], perfect prediction = 1
       - Interpretation: Model's explanatory power
       - Scale-independent, useful for comparing across different targets
       - Higher is better
    
    5. MAPE (Mean Absolute Percentage Error):
       - Average absolute percentage error
       - Units: Percentage (%)
       - Interpretation: Relative forecasting error
       - Scale-independent, easy to communicate to stakeholders
       - Problematic when true values are near zero
       - Lower is better
    
    For CityLearn energy forecasting:
    - RMSE is often the primary metric (matches energy units)
    - MAPE provides intuitive percentage-based interpretation
    - R² indicates how well the model captures energy patterns
    - MAE gives robust error estimate less affected by extreme values
    
    Args:
        y_true: True values (ground truth energy measurements)
        y_pred: Predicted values (model forecasts)
        
    Returns:
        Dictionary containing all calculated metrics:
        - mae: Mean Absolute Error
        - mse: Mean Squared Error  
        - rmse: Root Mean Squared Error
        - r2: R-squared coefficient
        - mape: Mean Absolute Percentage Error
        
    Example:
        >>> y_true = np.array([10, 20, 30, 40])
        >>> y_pred = np.array([12, 18, 32, 38])
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>> print(f"RMSE: {metrics['rmse']:.2f} kWh")
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Flatten arrays if multidimensional (e.g., multi-step forecasts)
    # This ensures metrics are calculated across all predictions
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Calculate comprehensive metrics for energy forecasting evaluation
    metrics = {
        # Mean Absolute Error: Average magnitude of errors
        # Robust to outliers, same units as target variable
        'mae': mean_absolute_error(y_true_flat, y_pred_flat),
        
        # Mean Squared Error: Average squared errors
        # Emphasizes larger errors, used in loss functions
        'mse': mean_squared_error(y_true_flat, y_pred_flat),
        
        # Root Mean Squared Error: Standard deviation of errors
        # Most commonly reported metric in energy forecasting
        'rmse': np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)),
        
        # R-squared: Proportion of variance explained
        # Indicates model's explanatory power (higher is better)
        'r2': r2_score(y_true_flat, y_pred_flat),
        
        # Mean Absolute Percentage Error: Relative error metric
        # Small epsilon (1e-8) prevents division by zero for zero true values
        # Multiplied by 100 to express as percentage
        'mape': np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100
    }
    
    return metrics


def cross_building_split(building_data: Dict[str, pd.DataFrame],
                        target_column: str,
                        sequence_length: int = 24) -> Dict[str, Dict]:
    """
    Prepare data for cross-building generalization tests.
    
    This function implements a critical evaluation strategy for building energy
    forecasting: cross-building generalization. It tests whether models trained
    on one building can generalize to predict energy patterns of other buildings.
    
    Why cross-building evaluation is important:
    
    1. Real-world deployment scenarios:
       - New buildings without historical data
       - Limited labeled data for some buildings
       - Transfer learning opportunities
       - Scalability to large building portfolios
    
    2. Model robustness assessment:
       - Tests if models learn generalizable energy patterns
       - Identifies overfitting to building-specific characteristics
       - Evaluates feature importance across different building types
       - Validates model architecture choices
    
    3. Building-agnostic vs. building-specific models:
       - Building-agnostic: Single model for all buildings
       - Building-specific: Individual models per building
       - Transfer learning: Pre-train on one, fine-tune on another
       - Ensemble approaches: Combine multiple building models
    
    4. Practical considerations:
       - Different building sizes, occupancy patterns, equipment
       - Varying climate conditions and seasonal effects
       - Different operational schedules and usage patterns
       - HVAC system differences and control strategies
    
    Expected results:
    - Lower performance than within-building evaluation
    - Performance varies by building similarity
    - Some features (weather, time) transfer better than others
    - Identifies most generalizable building for training
    
    Implementation strategy:
    - Train separate models for each building as source
    - Test each model on all other buildings as targets
    - Create comprehensive cross-building performance matrix
    - Enables analysis of which buildings are most similar
    
    Args:
        building_data: Dictionary of building dataframes from load_building_data()
        target_column: Column to predict (e.g., 'Electrical_Demand')
        sequence_length: Length of input sequences (default 24 hours)
        
    Returns:
        Dictionary with structure:
        {
            'Building_1': {
                'X_train': training sequences from Building_1,
                'y_train': training targets from Building_1,
                'test_buildings': {
                    'Building_2': {'X_test': ..., 'y_test': ...},
                    'Building_3': {'X_test': ..., 'y_test': ...}
                }
            },
            ...
        }
        
    Example usage:
        >>> cross_data = cross_building_split(building_data, 'Electrical_Demand')
        >>> # Train model on Building_1, test on Building_2
        >>> train_data = cross_data['Building_1']
        >>> test_data = cross_data['Building_1']['test_buildings']['Building_2']
    """
    # Filter to only building data (exclude weather, pricing, etc.)
    building_names = [name for name in building_data.keys() if name.startswith('Building_')]
    cross_building_data = {}
    
    # Create cross-building evaluation setup
    # Each building serves as training source for all others as test targets
    for train_building in building_names:
        # Prepare training data from source building
        # Use entire building dataset for training (no temporal split)
        # This maximizes training data for cross-building transfer
        train_df = building_data[train_building]
        
        if target_column not in train_df.columns:
            print(f"Warning: {target_column} not found in {train_building}, skipping...")
            continue
            
        train_target = train_df[target_column].values
        X_train, y_train = create_sequences(train_target, sequence_length)
        
        # Prepare test data from all other buildings (target buildings)
        # Each target building provides independent test set
        test_data = {}
        for test_building in building_names:
            if test_building != train_building:  # Skip same building
                test_df = building_data[test_building]
                
                if target_column not in test_df.columns:
                    print(f"Warning: {target_column} not found in {test_building}, skipping...")
                    continue
                    
                test_target = test_df[target_column].values
                X_test, y_test = create_sequences(test_target, sequence_length)
                
                # Store test data for this target building
                test_data[test_building] = {
                    'X_test': X_test, 
                    'y_test': y_test
                }
        
        # Store complete cross-building setup for this source building
        cross_building_data[train_building] = {
            'X_train': X_train,           # Training data from source building
            'y_train': y_train,           # Training targets from source building  
            'test_buildings': test_data   # Test data from all target buildings
        }
    
    return cross_building_data


def aggregate_neighborhood_data(building_data: Dict[str, pd.DataFrame],
                              columns: List[str]) -> pd.DataFrame:
    """
    Aggregate data across all buildings for neighborhood-level analysis.
    
    This function implements neighborhood-level aggregation, a key component
    of district energy management and the CityLearn Challenge. It combines
    individual building data to analyze community-wide energy patterns.
    
    Benefits of neighborhood aggregation:
    
    1. District energy management:
       - Total energy consumption and generation
       - Peak demand analysis for grid planning
       - Load balancing opportunities
       - Distributed energy resource coordination
    
    2. Demand smoothing effects:
       - Individual building variability cancels out
       - More predictable aggregate patterns
       - Reduced peak-to-average ratios
       - Better renewable energy integration
    
    3. Economic and environmental benefits:
       - Bulk energy purchasing power
       - Shared infrastructure costs
       - Community-wide carbon footprint
       - Collective demand response participation
    
    4. Forecasting advantages:
       - Aggregate patterns are easier to predict
       - Reduced impact of individual building anomalies
       - More stable training data for models
       - Better signal-to-noise ratio
    
    5. Grid interaction benefits:
       - Single point of contact with utility
       - Coordinated energy export/import
       - Improved grid stability contribution
       - Enhanced resilience through diversity
    
    Aggregation methodology:
    - Simple summation across buildings (appropriate for extensive quantities)
    - Ensures temporal alignment across all buildings
    - Handles missing data gracefully
    - Maintains original time series structure
    
    Typical columns to aggregate:
    - 'Electrical_Demand': Total neighborhood electricity consumption
    - 'Solar_Generation': Total neighborhood solar production
    - 'Battery_SOC': Combined battery state of charge
    - 'Net_Electrical_Demand': Net grid interaction
    
    Args:
        building_data: Dictionary of building dataframes from load_building_data()
        columns: List of columns to aggregate (must exist in all buildings)
        
    Returns:
        DataFrame with aggregated neighborhood data:
        - Columns prefixed with 'total_' to indicate aggregation
        - Same time index as original building data
        - Summed values across all buildings for each timestamp
        
    Example:
        >>> data = load_building_data()
        >>> neighborhood = aggregate_neighborhood_data(data, ['Electrical_Demand', 'Solar_Generation'])
        >>> print(neighborhood.columns)  # ['total_Electrical_Demand', 'total_Solar_Generation']
        >>> # Analyze peak neighborhood demand
        >>> peak_demand = neighborhood['total_Electrical_Demand'].max()
    """
    neighborhood_data = {}
    
    # Filter to only building data (exclude weather, pricing, etc.)
    building_dfs = {name: df for name, df in building_data.items() 
                   if name.startswith('Building_')}
    
    if not building_dfs:
        print("Warning: No building data found for aggregation")
        return pd.DataFrame()
    
    # Process each column for neighborhood aggregation
    for column in columns:
        # Verify column exists in all buildings
        missing_buildings = [name for name, df in building_dfs.items() 
                           if column not in df.columns]
        if missing_buildings:
            print(f"Warning: Column '{column}' missing in buildings: {missing_buildings}")
            continue
            
        # Sum across all buildings for each timestep
        # This creates neighborhood-level totals
        aggregated_values = []
        
        # Ensure all buildings have the same temporal length
        # Use minimum length to avoid index errors
        min_length = min(len(df) for df in building_dfs.values())
        
        if min_length == 0:
            print(f"Warning: Empty building data found, skipping column '{column}'")
            continue
        
        # Aggregate data timestep by timestep
        # This preserves temporal structure while summing across buildings
        for i in range(min_length):
            # Sum the same timestep across all buildings
            # This creates the neighborhood total for that time
            total = sum(df[column].iloc[i] for df in building_dfs.values() 
                       if not pd.isna(df[column].iloc[i]))
            aggregated_values.append(total)
            
        # Store with descriptive prefix to indicate aggregation
        neighborhood_data[f'total_{column}'] = aggregated_values
    
    # Create DataFrame with aggregated neighborhood data
    # Maintains same temporal structure as original building data
    return pd.DataFrame(neighborhood_data)