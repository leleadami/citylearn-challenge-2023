"""
CityLearn Challenge 2023 Forecasting Track Implementation.

This module provides a complete implementation for participating in the CityLearn Challenge 2023
forecasting track, focusing on building energy consumption prediction at multiple scales:

**CityLearn Challenge Overview**:
The CityLearn Challenge is an international competition for developing AI agents that can
optimize energy consumption and reduce carbon emissions in smart cities. The forecasting
track specifically focuses on predicting building energy consumption patterns.

**Competition Structure**:
- **Phase 1**: Training on provided historical data
- **Phase 2**: Testing on unseen data with periodic evaluation
- **Evaluation**: Normalized RMSE across all forecasting targets

**Forecasting Targets** (What the competition requires predicting):

**Building-Level Predictions** (per individual building):
1. **cooling_demand (kWh)**: Electrical energy consumed by cooling systems (HVAC)
   - Critical for: Summer peak load management, cooling system optimization
   - Challenges: Weather dependency, occupancy variations, thermal lag

2. **dhw_demand (kWh)**: Domestic Hot Water energy consumption
   - Critical for: Hot water system scheduling, energy storage optimization
   - Challenges: Occupancy patterns, seasonal variations

3. **non_shiftable_load (kWh)**: Equipment electrical power (lighting, computers, etc.)
   - Critical for: Base load forecasting, equipment scheduling
   - Challenges: Occupancy-dependent, business schedule variations

**Neighborhood-Level Predictions** (aggregated across buildings):
1. **carbon_intensity (kgCO2e/kWh)**: Grid carbon emissions per unit energy
   - Critical for: Carbon-aware energy scheduling, emissions reduction
   - Challenges: Grid composition changes, renewable energy variability

2. **solar_generation (W/kW)**: Aggregated solar power generation
   - Critical for: Renewable energy integration, grid stability
   - Challenges: Weather dependency, seasonal and diurnal variations

**Competition Evaluation**:
- **Metric**: Normalized Root Mean Square Error (NRMSE)
- **Horizon**: 48-hour ahead forecasting
- **Frequency**: Hourly predictions
- **Normalization**: By target variable range to ensure fair comparison

**Why This Challenge Matters**:
1. **Energy Efficiency**: Better predictions enable optimal building operations
2. **Grid Stability**: Accurate forecasts support smart grid management
3. **Carbon Reduction**: Predictive control can minimize emissions
4. **Cost Savings**: Optimal scheduling reduces energy costs
5. **Research Advancement**: Drives innovation in energy forecasting methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import os
import json
from sklearn.metrics import mean_squared_error
from ..utils.data_utils import load_building_data, create_sequences
from .base_models import get_all_forecasters


class CityLearnChallengeForecaster:
    """
    Complete implementation for CityLearn Challenge 2023 Forecasting Track.
    
    This class provides end-to-end functionality for participating in the CityLearn Challenge,
    handling data loading, preprocessing, model training, evaluation, and results analysis.
    
    **Challenge Forecasting Architecture**:
    
    **Multi-Scale Prediction Framework**:
    1. **Building-Level Models**: Individual models for each building and target
       - Captures building-specific patterns and characteristics
       - Enables tailored optimization for different building types
    
    2. **Neighborhood-Level Models**: Aggregated prediction across buildings
       - Models community-wide patterns and grid interactions
       - Supports district-level energy management decisions
    
    **Forecasting Targets Explained**:
    
    **Building-Level Targets** (High-Resolution Predictions):
    - **cooling_demand**: HVAC cooling energy consumption
      * Weather-dependent, occupancy-sensitive
      * Critical for peak demand management
    - **dhw_demand**: Domestic hot water energy usage
      * Occupancy-driven, schedule-dependent
      * Important for thermal energy storage
    - **non_shiftable_load**: Essential electrical equipment
      * Lighting, computers, elevators, ventilation
      * Base load component with occupancy patterns
    
    **Neighborhood-Level Targets** (District-Wide Patterns):
    - **carbon_intensity**: Grid emissions factor
      * Varies with energy source mix (fossil vs renewable)
      * Critical for carbon-aware scheduling
    - **solar_generation**: Distributed solar production
      * Weather-dependent, aggregated across buildings
      * Key for renewable energy integration
    
    **Competition Parameters**:
    - **Prediction Horizon**: 48 hours (supports day-ahead + real-time planning)
    - **Input Window**: 24 hours (captures daily operational cycles)
    - **Frequency**: Hourly predictions (aligned with energy market intervals)
    - **Evaluation**: Normalized RMSE (fair comparison across different scales)
    
    **Data Split Strategy**:
    - **Training (60%)**: Model learning and pattern discovery
    - **Validation (20%)**: Hyperparameter tuning and overfitting prevention
    - **Testing (20%)**: Final performance evaluation (simulates competition)
    
    **Multi-Model Approach**:
    Tests multiple forecasting algorithms to identify best performers:
    - Traditional ML (Linear, Random Forest, Gaussian Process)
    - Deep Learning (LSTM, ANN)
    - Advanced Architectures (Transformers, TimesFM-inspired)
    """
    
    def __init__(self, data_path: str = "data"):
        """
        Initialize CityLearn Challenge forecaster.
        
        Args:
            data_path (str): Path to CityLearn Challenge datasets
                           Should contain phase_1, phase_2_* subdirectories
        """
        self.data_path = data_path
        
        # Challenge forecasting targets (as specified in competition)
        self.building_targets = ['cooling_demand', 'dhw_demand', 'non_shiftable_load']
        self.neighborhood_targets = ['carbon_intensity', 'solar_generation']
        
        # Competition temporal parameters
        self.prediction_horizon = 48    # 48-hour ahead forecasting (competition requirement)
        self.sequence_length = 24       # 24-hour input window (daily cycle capture)
        
        # Competition data split ratios (industry standard for forecasting competitions)
        self.train_ratio = 0.6          # 60% training data
        self.val_ratio = 0.2            # 20% validation data
        self.test_ratio = 0.2           # 20% test data (competition simulation)
        
        # Data storage and model tracking
        self.data = None                # Loaded challenge datasets
        self.models = {}                # Trained forecasting models
        self.results = {}               # Competition performance results
    
    def load_challenge_data(self, phase: str = "phase_1"):
        """
        Load CityLearn Challenge datasets for competition phases.
        
        **Challenge Data Structure**:
        The CityLearn Challenge provides structured datasets representing smart city
        energy consumption and environmental conditions:
        
        **Building Files** (Building_1.csv, Building_2.csv, etc.):
        Each building dataset contains hourly time series with columns:
        - **cooling_demand**: HVAC cooling energy consumption (kWh)
        - **dhw_demand**: Domestic hot water energy consumption (kWh) 
        - **non_shiftable_load**: Essential electrical equipment consumption (kWh)
        - **solar_generation**: Building-level solar power production (W/kW)
        - **Timestamp columns**: For temporal alignment
        
        **Neighborhood Files**:
        - **carbon_intensity.csv**: Grid carbon emissions factor (kgCO2e/kWh)
        - **weather.csv**: Meteorological data (temperature, humidity, solar radiation)
        - **pricing.csv**: Energy pricing signals (if available)
        
        **Competition Phases**:
        - **Phase 1**: Historical training data with full ground truth
        - **Phase 2**: Evaluation data with limited ground truth
        - **Phase 2 Variants**: Multiple test scenarios (local, online evaluation)
        
        **Data Quality Considerations**:
        - **Temporal Alignment**: All datasets share common timestamps
        - **Missing Values**: May require interpolation or forward filling
        - **Units**: Standardized across buildings for fair comparison
        - **Frequency**: Hourly resolution for operational-level forecasting
        
        **Why This Data Loading Matters**:
        1. **Competition Compliance**: Ensures proper data format for submission
        2. **Fair Evaluation**: Consistent loading across all participants
        3. **Reproducibility**: Standardized data access for research
        4. **Scalability**: Can handle multiple buildings and phases
        
        Args:
            phase (str): Competition phase identifier
                        "phase_1" = training data
                        "phase_2_*" = evaluation scenarios
        
        Raises:
            FileNotFoundError: If phase data directory doesn't exist
        """
        # Construct phase-specific data path
        phase_path = os.path.join(self.data_path, f"citylearn_challenge_2023_{phase}")
        
        # Validate data availability
        if not os.path.exists(phase_path):
            raise FileNotFoundError(
                f"Competition data not found: {phase_path}\n"
                f"Please ensure CityLearn Challenge data is properly downloaded and extracted."
            )
        
        # Initialize data storage
        self.data = {}
        
        # Load building-level energy consumption data
        building_files = [
            f for f in os.listdir(phase_path) 
            if f.startswith('Building_') and f.endswith('.csv')
        ]
        
        for file in building_files:
            building_name = file.replace('.csv', '')  # Extract building identifier
            file_path = os.path.join(phase_path, file)
            
            # Load building energy consumption time series
            self.data[building_name] = pd.read_csv(file_path)
            
            # Basic data validation
            required_columns = set(self.building_targets)
            available_columns = set(self.data[building_name].columns)
            missing_columns = required_columns - available_columns
            
            if missing_columns:
                print(f"WARNING  Warning: {building_name} missing columns: {missing_columns}")
        
        # Load neighborhood-level and environmental data
        auxiliary_files = ['carbon_intensity', 'weather', 'pricing']
        for target in auxiliary_files:
            file_path = os.path.join(phase_path, f"{target}.csv")
            if os.path.exists(file_path):
                self.data[target] = pd.read_csv(file_path)
            else:
                print(f"ℹ️  Optional file not found: {target}.csv")
        
        # Summary statistics
        building_count = len([k for k in self.data.keys() if k.startswith('Building_')])
        auxiliary_count = len([k for k in self.data.keys() if not k.startswith('Building_')])
        
        print(f"SUCCESS Successfully loaded CityLearn Challenge {phase} data:")
        print(f"   [DATA] Buildings: {building_count} ({[k for k in self.data.keys() if k.startswith('Building_')]})") 
        print(f"   🌍 Auxiliary datasets: {auxiliary_count} ({[k for k in self.data.keys() if not k.startswith('Building_')]})")        
        print(f"   [TARGET] Ready for forecasting target preparation")
    
    def prepare_building_forecasting_data(self):
        """
        Prepare building-level forecasting data for all targets.
        
        Returns:
            Dictionary with prepared data for each building and target
        """
        prepared_data = {}
        
        building_names = [k for k in self.data.keys() if k.startswith('Building_')]
        
        for building_name in building_names:
            building_df = self.data[building_name]
            prepared_data[building_name] = {}
            
            for target in self.building_targets:
                if target not in building_df.columns:
                    print(f"WARNING  Target {target} not found in {building_name}")
                    continue
                
                # Extract target data
                target_data = building_df[target].values
                
                # Create sequences
                X, y = create_sequences(
                    target_data, 
                    self.sequence_length, 
                    self.prediction_horizon
                )
                
                if len(X) == 0:
                    print(f"WARNING  No sequences created for {building_name}.{target}")
                    continue
                
                # Split data (60/20/20)
                n_samples = len(X)
                train_end = int(n_samples * self.train_ratio)
                val_end = int(n_samples * (self.train_ratio + self.val_ratio))
                
                prepared_data[building_name][target] = {
                    'X_train': X[:train_end],
                    'X_val': X[train_end:val_end],
                    'X_test': X[val_end:],
                    'y_train': y[:train_end],
                    'y_val': y[train_end:val_end],
                    'y_test': y[val_end:],
                    'raw_data': target_data
                }
                
                print(f"   {building_name}.{target}: {len(X)} sequences -> "
                      f"Train: {len(X[:train_end])}, Val: {len(X[train_end:val_end])}, "
                      f"Test: {len(X[val_end:])}")
        
        return prepared_data
    
    def prepare_neighborhood_forecasting_data(self):
        """
        Prepare neighborhood-level forecasting data.
        
        Returns:
            Dictionary with prepared neighborhood data
        """
        prepared_data = {}
        
        # Carbon intensity (already aggregated)
        if 'carbon_intensity' in self.data:
            carbon_data = self.data['carbon_intensity']['carbon_intensity'].values
            X, y = create_sequences(carbon_data, self.sequence_length, self.prediction_horizon)
            
            if len(X) > 0:
                n_samples = len(X)
                train_end = int(n_samples * self.train_ratio)
                val_end = int(n_samples * (self.train_ratio + self.val_ratio))
                
                prepared_data['carbon_intensity'] = {
                    'X_train': X[:train_end],
                    'X_val': X[train_end:val_end], 
                    'X_test': X[val_end:],
                    'y_train': y[:train_end],
                    'y_val': y[train_end:val_end],
                    'y_test': y[val_end:],
                    'raw_data': carbon_data
                }
                print(f"   carbon_intensity: {len(X)} sequences")
        
        # Solar generation (aggregate across buildings)
        building_names = [k for k in self.data.keys() if k.startswith('Building_')]
        if building_names and 'solar_generation' in self.data[building_names[0]].columns:
            # Aggregate solar generation across all buildings
            solar_data = np.zeros(len(self.data[building_names[0]]))
            
            for building_name in building_names:
                building_solar = self.data[building_name]['solar_generation'].values
                solar_data += building_solar
            
            X, y = create_sequences(solar_data, self.sequence_length, self.prediction_horizon)
            
            if len(X) > 0:
                n_samples = len(X)
                train_end = int(n_samples * self.train_ratio)
                val_end = int(n_samples * (self.train_ratio + self.val_ratio))
                
                prepared_data['solar_generation_neighborhood'] = {
                    'X_train': X[:train_end],
                    'X_val': X[train_end:val_end],
                    'X_test': X[val_end:],
                    'y_train': y[:train_end],
                    'y_val': y[train_end:val_end],
                    'y_test': y[val_end:],
                    'raw_data': solar_data
                }
                print(f"   solar_generation_neighborhood: {len(X)} sequences")
        
        return prepared_data
    
    def calculate_normalized_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Normalized Root Mean Square Error - the official CityLearn Challenge metric.
        
        **Why Normalized RMSE for Energy Forecasting Competitions?**
        
        The CityLearn Challenge uses normalized RMSE to ensure fair comparison across
        forecasting targets with vastly different scales:
        
        **Scale Diversity Problem**:
        - cooling_demand: 0-50 kWh (building HVAC)
        - carbon_intensity: 0.1-0.8 kgCO2e/kWh (grid emissions)
        - solar_generation: 0-800 W/kW (solar power)
        
        **NRMSE Solution**: NRMSE = RMSE / (max(y_true) - min(y_true))
        
        **Benefits of Normalization**:
        1. **Scale Invariance**: Targets with different units get equal weight
        2. **Percentage Interpretation**: NRMSE ≈ percentage error relative to range
        3. **Competition Fairness**: No target dominates due to scale
        4. **Interpretability**: 0.1 NRMSE = 10% of the target's total range
        
        **Mathematical Properties**:
        - **Range**: [0, ∞) where 0 = perfect prediction
        - **Baseline**: NRMSE = 1.0 equivalent to predicting mean always
        - **Units**: Dimensionless (enables cross-target comparison)
        
        **Energy Forecasting Interpretation**:
        - NRMSE < 0.1: Excellent forecasting (< 10% of range)
        - NRMSE < 0.2: Good forecasting (< 20% of range)
        - NRMSE < 0.5: Acceptable forecasting (< 50% of range)
        - NRMSE > 1.0: Poor forecasting (worse than mean prediction)
        
        **Edge Cases Handled**:
        - Zero range (constant target): Returns 0.0 (perfect prediction of constant)
        - NaN values: Should be handled by upstream data validation
        
        Args:
            y_true (np.ndarray): Ground truth energy values (any shape)
            y_pred (np.ndarray): Predicted energy values (matching shape)
            
        Returns:
            float: Normalized RMSE score (0 = perfect, higher = worse)
                  Values around 0.1-0.3 typical for good energy forecasting
        
        Example:
            >>> y_true = np.array([10, 20, 30, 40, 50])  # Energy consumption
            >>> y_pred = np.array([12, 18, 32, 38, 52])  # Predictions
            >>> nrmse = calculate_normalized_rmse(y_true, y_pred)
            >>> print(f"NRMSE: {nrmse:.3f}")  # Outputs: NRMSE: 0.100
        """
        # Flatten arrays to handle multi-dimensional predictions (e.g., multi-horizon)
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Validate input dimensions
        if len(y_true_flat) != len(y_pred_flat):
            raise ValueError(
                f"Prediction length mismatch: {len(y_pred_flat)} vs {len(y_true_flat)}"
            )
        
        # Calculate Root Mean Square Error
        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        
        # Calculate normalization factor (target range)
        y_range = np.max(y_true_flat) - np.min(y_true_flat)
        
        # Handle edge case: constant target values
        if y_range == 0:
            # If target is constant and RMSE is 0, prediction is perfect
            # If target is constant but RMSE > 0, prediction is imperfect
            return 0.0 if rmse == 0 else float('inf')
        
        # Compute normalized RMSE
        normalized_rmse = rmse / y_range
        
        return normalized_rmse
    
    def run_building_forecasting_experiments(self, building_data: Dict):
        """
        Run forecasting experiments for all building-level targets.
        
        Args:
            building_data: Prepared building data
            
        Returns:
            Results dictionary
        """
        print("🏢 Running building-level forecasting experiments...")
        
        forecasters = get_all_forecasters()
        results = {}
        
        for building_name, building_targets in building_data.items():
            print(f"\n  Building: {building_name}")
            results[building_name] = {}
            
            for target, data in building_targets.items():
                print(f"    Target: {target}")
                results[building_name][target] = {}
                
                for model_name, forecaster in forecasters.items():
                    print(f"      Model: {model_name}")
                    
                    try:
                        # Train model
                        if model_name in ['LSTM', 'ANN']:
                            # Reshape for neural networks if needed
                            X_train = data['X_train']
                            if len(X_train.shape) == 2:
                                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                            
                            X_val = data['X_val'] 
                            if len(X_val.shape) == 2:
                                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
                            
                            forecaster.fit(X_train, data['y_train'], X_val, data['y_val'], 
                                         epochs=50, verbose=0)
                        else:
                            forecaster.fit(data['X_train'], data['y_train'])
                        
                        # Make predictions
                        X_test = data['X_test']
                        if model_name in ['LSTM', 'ANN'] and len(X_test.shape) == 2:
                            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                        
                        y_pred = forecaster.predict(X_test)
                        y_true = data['y_test']
                        
                        # Calculate metrics
                        normalized_rmse = self.calculate_normalized_rmse(y_true, y_pred)
                        
                        # Standard metrics for comparison
                        from ..utils.data_utils import calculate_metrics
                        standard_metrics = calculate_metrics(y_true, y_pred)
                        
                        results[building_name][target][model_name] = {
                            'normalized_rmse': normalized_rmse,
                            **standard_metrics
                        }
                        
                        print(f"        Normalized RMSE: {normalized_rmse:.4f}")
                        
                    except Exception as e:
                        print(f"        Error: {str(e)}")
                        results[building_name][target][model_name] = {'error': str(e)}
        
        return results
    
    def run_neighborhood_forecasting_experiments(self, neighborhood_data: Dict):
        """
        Run forecasting experiments for neighborhood-level targets.
        
        Args:
            neighborhood_data: Prepared neighborhood data
            
        Returns:
            Results dictionary
        """
        print("🏘️ Running neighborhood-level forecasting experiments...")
        
        forecasters = get_all_forecasters()
        results = {}
        
        for target, data in neighborhood_data.items():
            print(f"\n  Target: {target}")
            results[target] = {}
            
            for model_name, forecaster in forecasters.items():
                print(f"    Model: {model_name}")
                
                try:
                    # Train model
                    if model_name in ['LSTM', 'ANN']:
                        X_train = data['X_train']
                        if len(X_train.shape) == 2:
                            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                        
                        X_val = data['X_val']
                        if len(X_val.shape) == 2:
                            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
                        
                        forecaster.fit(X_train, data['y_train'], X_val, data['y_val'],
                                     epochs=50, verbose=0)
                    else:
                        forecaster.fit(data['X_train'], data['y_train'])
                    
                    # Make predictions
                    X_test = data['X_test']
                    if model_name in ['LSTM', 'ANN'] and len(X_test.shape) == 2:
                        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                    
                    y_pred = forecaster.predict(X_test)
                    y_true = data['y_test']
                    
                    # Calculate metrics
                    normalized_rmse = self.calculate_normalized_rmse(y_true, y_pred)
                    
                    from ..utils.data_utils import calculate_metrics
                    standard_metrics = calculate_metrics(y_true, y_pred)
                    
                    results[target][model_name] = {
                        'normalized_rmse': normalized_rmse,
                        **standard_metrics
                    }
                    
                    print(f"      Normalized RMSE: {normalized_rmse:.4f}")
                    
                except Exception as e:
                    print(f"      Error: {str(e)}")
                    results[target][model_name] = {'error': str(e)}
        
        return results
    
    def calculate_challenge_score(self, building_results: Dict, neighborhood_results: Dict) -> float:
        """
        Calculate overall CityLearn Challenge score.
        
        The competition score is based on normalized RMSE across all forecasted variables.
        
        Args:
            building_results: Building-level results
            neighborhood_results: Neighborhood-level results
            
        Returns:
            Overall challenge score (lower is better)
        """
        all_scores = []
        
        # Collect all normalized RMSE scores
        for building_name, building_targets in building_results.items():
            for target, models in building_targets.items():
                for model_name, metrics in models.items():
                    if 'normalized_rmse' in metrics:
                        all_scores.append(metrics['normalized_rmse'])
        
        for target, models in neighborhood_results.items():
            for model_name, metrics in models.items():
                if 'normalized_rmse' in metrics:
                    all_scores.append(metrics['normalized_rmse'])
        
        if not all_scores:
            return float('inf')
        
        # Overall score is the mean of all normalized RMSE values
        overall_score = np.mean(all_scores)
        return overall_score
    
    def run_full_challenge_experiment(self, phase: str = "phase_1"):
        """
        Execute complete CityLearn Challenge forecasting experiment.
        
        This method orchestrates the entire competition workflow, from data loading
        through model evaluation, simulating the complete challenge experience:
        
        **Experiment Pipeline**:
        
        **Stage 1 - Data Preparation**:
        1. Load challenge datasets (buildings, weather, carbon intensity)
        2. Validate data integrity and completeness
        3. Prepare time series sequences for forecasting models
        4. Split into training/validation/test sets (60/20/20)
        
        **Stage 2 - Building-Level Forecasting**:
        1. Train models on individual building energy consumption patterns
        2. Evaluate on cooling_demand, dhw_demand, non_shiftable_load
        3. Test multiple algorithms (LSTM, Random Forest, Transformers, etc.)
        4. Calculate normalized RMSE for each building-target combination
        
        **Stage 3 - Neighborhood-Level Forecasting**:
        1. Train models on district-wide patterns (carbon intensity, solar)
        2. Handle aggregated data across multiple buildings
        3. Account for grid-level dynamics and weather dependencies
        4. Evaluate community-scale prediction accuracy
        
        **Stage 4 - Competition Scoring**:
        1. Aggregate normalized RMSE across all targets
        2. Calculate overall challenge score (lower = better)
        3. Rank models by competition performance
        4. Generate comprehensive results for analysis
        
        **Why This Comprehensive Approach?**:
        
        1. **Competition Realism**: Simulates actual challenge evaluation process
        2. **Model Comparison**: Tests multiple algorithms under same conditions
        3. **Multi-Scale Evaluation**: Assesses both building and district performance
        4. **Reproducibility**: Standardized workflow enables fair comparison
        5. **Research Value**: Generates rich dataset for forecasting research
        
        **Expected Performance Ranges** (Normalized RMSE):
        - **Excellent Models**: 0.05-0.15 (5-15% of target range)
        - **Good Models**: 0.15-0.30 (15-30% of target range)
        - **Acceptable Models**: 0.30-0.50 (30-50% of target range)
        - **Poor Models**: >0.50 (worse than simple baselines)
        
        **Competition Strategy Tips**:
        1. Focus on robust models that work across all targets
        2. Pay attention to both building and neighborhood scales
        3. Consider ensemble methods for improved performance
        4. Validate thoroughly to avoid overfitting
        
        Args:
            phase (str): Competition phase to evaluate
                        "phase_1" = training/validation phase
                        "phase_2_*" = evaluation phases
                        
        Returns:
            dict: Comprehensive results including:
                 - building_results: Per-building, per-target performance
                 - neighborhood_results: District-level performance  
                 - overall_score: Competition ranking metric
                 - metadata: Experimental configuration
                 
        Example:
            ```python
            challenger = CityLearnChallengeForecaster()
            results = challenger.run_full_challenge_experiment("phase_1")
            print(f"Challenge Score: {results['overall_score']:.4f}")
            ```
        """
        print(f"🏆 CityLearn Challenge 2023 - {phase.upper()} Forecasting Experiment")
        print("=" * 70)
        print("[TARGET] Multi-scale building energy consumption forecasting")
        print("📅 48-hour ahead predictions with hourly resolution")
        print("="* 70)
        
        # Stage 1: Data Loading and Validation
        print("\n[STAGE] Stage 1: Loading CityLearn Challenge data...")
        self.load_challenge_data(phase)
        
        # Stage 2: Building-Level Data Preparation
        print("\n[DATA] Stage 2: Preparing building-level forecasting data...")
        print("   Targets: cooling_demand, dhw_demand, non_shiftable_load")
        building_data = self.prepare_building_forecasting_data()
        
        # Stage 3: Neighborhood-Level Data Preparation  
        print("\n🌍 Stage 3: Preparing neighborhood-level forecasting data...")
        print("   Targets: carbon_intensity, solar_generation")
        neighborhood_data = self.prepare_neighborhood_forecasting_data()
        
        # Stage 4: Building-Level Model Training and Evaluation
        print("\n🏢 Stage 4: Building-level forecasting experiments...")
        building_results = self.run_building_forecasting_experiments(building_data)
        
        # Stage 5: Neighborhood-Level Model Training and Evaluation
        print("\n🎆 Stage 5: Neighborhood-level forecasting experiments...")
        neighborhood_results = self.run_neighborhood_forecasting_experiments(neighborhood_data)
        
        # Stage 6: Competition Score Calculation
        print("\n🏅 Stage 6: Calculating CityLearn Challenge score...")
        overall_score = self.calculate_challenge_score(building_results, neighborhood_results)
        
        # Compile comprehensive results
        results = {
            'phase': phase,
            'building_results': building_results,
            'neighborhood_results': neighborhood_results, 
            'overall_score': overall_score,
            'competition_metadata': {
                'prediction_horizon_hours': self.prediction_horizon,
                'input_sequence_hours': self.sequence_length,
                'data_split_ratios': {
                    'train': self.train_ratio,
                    'validation': self.val_ratio,
                    'test': self.test_ratio
                },
                'evaluation_metric': 'normalized_rmse',
                'forecasting_frequency': 'hourly',
                'challenge_year': 2023
            }
        }
        
        # Results Summary
        print("\n" + "=" * 70)
        print("🏁 CITYLEARN CHALLENGE RESULTS SUMMARY")
        print("=" * 70)
        print(f"[TARGET] Overall Challenge Score (NRMSE): {overall_score:.4f}")
        
        # Performance categorization
        if overall_score < 0.15:
            performance_level = "🎆 EXCELLENT"
        elif overall_score < 0.30:
            performance_level = "💫 GOOD"
        elif overall_score < 0.50:
            performance_level = "📈 ACCEPTABLE"
        else:
            performance_level = "WARNING NEEDS IMPROVEMENT"
            
        print(f"📉 Performance Level: {performance_level}")
        print(f"🕰️ Experiment completed for {phase}")
        print("SUCCESS Results ready for analysis and submission preparation")
        print("=" * 70)
        
        return results
    
    def save_results(self, results: Dict, filepath: str):
        """Save challenge results to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to {filepath}")


def run_citylearn_challenge_experiment():
    """
    Convenience function for complete CityLearn Challenge 2023 participation.
    
    This function provides a one-click solution for researchers and practitioners
    to evaluate forecasting models on the CityLearn Challenge datasets:
    
    **Full Competition Simulation**:
    
    **Phase 1 - Training and Validation**:
    1. Loads historical building energy consumption data
    2. Trains multiple forecasting models on all targets
    3. Evaluates performance using competition metrics
    4. Saves comprehensive results for analysis
    
    **Phase 2 - Evaluation (if available)**:
    1. Tests models on unseen evaluation data
    2. Simulates actual competition submission process
    3. Provides final performance rankings
    4. Generates submission-ready results
    
    **Multi-Model Evaluation**:
    Tests complete suite of forecasting algorithms:
    - **Traditional ML**: Linear Regression, Random Forest, Gaussian Process
    - **Deep Learning**: LSTM, Artificial Neural Networks
    - **Advanced**: Transformer, TimesFM-inspired models
    
    **Automated Workflow**:
    1. **Data Loading**: Handles all CityLearn data formats automatically
    2. **Preprocessing**: Creates proper sequences and splits for each target
    3. **Model Training**: Trains all models with optimized hyperparameters
    4. **Evaluation**: Computes normalized RMSE for fair comparison
    5. **Results Export**: Saves detailed results in JSON format
    6. **Error Handling**: Gracefully handles missing data or failed models
    
    **Research Applications**:
    - **Benchmarking**: Compare new algorithms against established baselines
    - **Ablation Studies**: Analyze impact of different model components
    - **Transfer Learning**: Test model performance across building types
    - **Ensemble Methods**: Combine multiple models for improved performance
    
    **Industry Applications**:
    - **Building Optimization**: Deploy best models for energy management
    - **Grid Planning**: Use forecasts for demand response programs
    - **Carbon Reduction**: Schedule operations to minimize emissions
    - **Cost Optimization**: Reduce energy costs through predictive control
    
    **Expected Outputs**:
    - **Phase 1 Results**: Training performance and model rankings
    - **Phase 2 Results**: Final competition-style evaluation (if data available)
    - **JSON Files**: Detailed results for further analysis
    - **Performance Metrics**: Normalized RMSE for each model and target
    
    **Usage Examples**:
    
    ```python
    # Run complete challenge evaluation
    results = run_citylearn_challenge_experiment()
    
    # Analyze best performing models
    best_score = results['overall_score']
    print(f"Best Challenge Score: {best_score:.4f}")
    
    # Access detailed results
    building_results = results['building_results']
    for building, targets in building_results.items():
        for target, models in targets.items():
            print(f"{building}.{target} best model: {min(models, key=lambda x: models[x].get('normalized_rmse', float('inf')))}")
    ```
    
    Returns:
        dict: Phase 1 results with comprehensive forecasting performance
              Contains model rankings, scores, and detailed analysis
              
    Raises:
        FileNotFoundError: If CityLearn Challenge data not properly installed
        ValueError: If data format doesn't match competition specification
        
    Note:
        Requires CityLearn Challenge 2023 datasets downloaded and extracted
        to 'data/' directory with proper folder structure.
    """
    print("🏆 CITYLEARN CHALLENGE 2023 - COMPLETE FORECASTING EVALUATION")
    print("=" * 80)
    print("[TARGET] Evaluating all forecasting models on building energy consumption")
    print("🕰️ This may take several minutes to complete...")
    print("=" * 80)
    
    # Initialize challenge forecaster
    challenger = CityLearnChallengeForecaster()
    
    # Phase 1: Training and validation experiment
    print("\n🚀 Starting Phase 1 (Training/Validation)...")
    phase1_results = challenger.run_full_challenge_experiment("phase_1")
    
    # Save Phase 1 results
    phase1_path = "results/citylearn_challenge_phase1_results.json"
    challenger.save_results(phase1_results, phase1_path)
    print(f"💾 Phase 1 results saved to: {phase1_path}")
    
    # Phase 2: Evaluation experiment (if data available)
    print("\n🚀 Attempting Phase 2 (Evaluation)...")
    try:
        phase2_results = challenger.run_full_challenge_experiment("phase_2_local_evaluation")
        phase2_path = "results/citylearn_challenge_phase2_results.json"
        challenger.save_results(phase2_results, phase2_path)
        print(f"💾 Phase 2 results saved to: {phase2_path}")
        print("🎆 Both phases completed successfully!")
    except FileNotFoundError:
        print("ℹ️  Phase 2 evaluation data not available - skipping")
        print("📁 Download Phase 2 data for complete competition simulation")
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUCCESS CITYLEARN CHALLENGE EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"🏅 Phase 1 Score: {phase1_results['overall_score']:.4f}")
    print("📈 Check results/ directory for detailed analysis")
    print("🚀 Ready for model deployment or further research!")
    print("=" * 80)
    
    return phase1_results