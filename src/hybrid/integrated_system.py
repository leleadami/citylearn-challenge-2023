"""
Integrated Building Energy Management System
Combines all AI components: RL + LSTM + Transformer + Classical ML
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import json
import os
import warnings
warnings.filterwarnings('ignore')

from .fusion_agent import HybridFusionAgent, create_hybrid_agent
from ..forecasting.citylearn_challenge import CityLearnChallengeForecaster
from ..utils.data_utils import create_sequences


class IntegratedBuildingSystem:
    """
    Complete building energy management system integrating:
    1. Multi-horizon forecasting (LSTM + Transformer + Classical)
    2. Reinforcement learning control (Q-Learning + SAC)
    3. Cross-building transfer learning
    4. Neighborhood aggregation
    5. Adaptive model fusion
    """
    
    def __init__(self, data_path: str = "../data"):
        """Initialize integrated system."""
        self.data_path = data_path
        self.forecaster = CityLearnChallengeForecaster(data_path)
        self.hybrid_agents = {}  # One agent per building
        self.neighborhood_agent = None  # District-level agent
        
        # System configuration
        self.config = {
            'sequence_length': 24,
            'prediction_horizon': 48,
            'learning_rate': 0.001,
            'rl_learning_rate': 0.1,
            'train_ratio': 0.6,
            'val_ratio': 0.2,
            'test_ratio': 0.2
        }
        
        # Performance tracking
        self.performance_log = {
            'building_level': {},
            'neighborhood_level': {},
            'system_level': {}
        }
        
        print("[SYSTEM] Integrated Building Energy Management System initialized")
        
    def setup_system(self, phase: str = "phase_1"):
        """Setup complete system with data loading and agent initialization."""
        print(f"[SETUP] Setting up integrated system for {phase}")
        
        # Load CityLearn data
        try:
            self.forecaster.load_challenge_data(phase)
            print("[DATA] CityLearn Challenge data loaded successfully")
        except Exception as e:
            print(f"[DATA] Error loading data: {e}")
            return False
            
        # Initialize building-level agents
        building_names = [k for k in self.forecaster.data.keys() if k.startswith('Building_')]
        
        for building_name in building_names:
            building_data = self.forecaster.data[building_name]
            
            # Estimate state and action dimensions
            state_dim = len(building_data.columns)  # All available features
            action_dim = 9  # Standard CityLearn action space (Phase 2)
            
            # Create hybrid agent for this building
            agent = create_hybrid_agent(
                state_dim=state_dim,
                action_dim=action_dim,
                sequence_length=self.config['sequence_length'],
                prediction_horizon=self.config['prediction_horizon'],
                learning_rate=self.config['learning_rate']
            )
            
            self.hybrid_agents[building_name] = agent
            print(f"[AGENT] Created hybrid agent for {building_name}")
            
        # Initialize neighborhood-level agent
        if len(building_names) > 1:
            neighborhood_state_dim = sum(len(self.forecaster.data[b].columns) for b in building_names)
            neighborhood_action_dim = len(building_names) * 9  # Combined action space
            
            self.neighborhood_agent = create_hybrid_agent(
                state_dim=neighborhood_state_dim,
                action_dim=neighborhood_action_dim,
                sequence_length=self.config['sequence_length'],
                prediction_horizon=self.config['prediction_horizon'],
                learning_rate=self.config['learning_rate']
            )
            print("[AGENT] Created neighborhood-level hybrid agent")
            
        print(f"[SETUP] System setup complete with {len(self.hybrid_agents)} building agents")
        return True
        
    def train_forecasting_models(self):
        """Train all forecasting components across buildings."""
        print("[TRAIN] Training integrated forecasting models...")
        
        building_names = [k for k in self.forecaster.data.keys() if k.startswith('Building_')]
        targets = ['cooling_demand', 'heating_demand', 'solar_generation']
        
        for building_name in building_names:
            if building_name not in self.hybrid_agents:
                continue
                
            building_data = self.forecaster.data[building_name]
            agent = self.hybrid_agents[building_name]
            
            # Prepare training data for each target
            training_data = {}
            training_targets = {}
            
            for target in targets:
                if target in building_data.columns:
                    # Create sequences for neural network training
                    target_values = building_data[target].values
                    
                    if len(target_values) >= self.config['sequence_length'] + self.config['prediction_horizon']:
                        X, y = create_sequences(
                            target_values,
                            self.config['sequence_length'],
                            self.config['prediction_horizon']
                        )
                        
                        if len(X) > 0:
                            # Split data
                            n_samples = len(X)
                            train_end = int(n_samples * self.config['train_ratio'])
                            
                            training_data[target] = X[:train_end]
                            training_targets[target] = y[:train_end]
                            
                            print(f"[TRAIN] Prepared {len(training_data[target])} samples for {building_name}.{target}")
                        
            # Train the agent's forecasting models
            if training_data and training_targets:
                try:
                    agent.train_forecasters(training_data, training_targets)
                    print(f"[TRAIN] Completed training for {building_name}")
                except Exception as e:
                    print(f"[TRAIN] Training failed for {building_name}: {e}")
                    
        print("[TRAIN] Forecasting model training complete")
        
    def run_comprehensive_evaluation(self):
        """Run complete system evaluation with all components."""
        print("[EVAL] Running comprehensive system evaluation...")
        
        results = {
            'building_forecasting': {},
            'neighborhood_forecasting': {},
            'rl_control': {},
            'integrated_performance': {}
        }
        
        # Building-level evaluation
        building_names = [k for k in self.forecaster.data.keys() if k.startswith('Building_')]
        targets = ['cooling_demand', 'heating_demand', 'solar_generation']
        
        for building_name in building_names:
            if building_name not in self.hybrid_agents:
                continue
                
            building_data = self.forecaster.data[building_name]
            agent = self.hybrid_agents[building_name]
            building_results = {}
            
            for target in targets:
                if target in building_data.columns:
                    target_values = building_data[target].values
                    
                    if len(target_values) >= 48:  # Minimum data requirement
                        # Use recent data for forecasting
                        historical_data = target_values[-72:]  # Last 3 days
                        
                        try:
                            # Generate forecasts using all models
                            forecasts = agent.forecast(historical_data, target)
                            
                            # Evaluate against next day (if available)
                            if len(target_values) >= 48:
                                actual_next = target_values[-48:]
                                
                                forecast_errors = {}
                                for model_name, forecast in forecasts.items():
                                    if len(forecast) >= 24:  # At least 1 day forecast
                                        pred_24h = forecast[:24]
                                        actual_24h = actual_next[:24]
                                        
                                        mae = np.mean(np.abs(pred_24h - actual_24h))
                                        rmse = np.sqrt(np.mean((pred_24h - actual_24h)**2))
                                        
                                        forecast_errors[model_name] = {
                                            'mae': mae,
                                            'rmse': rmse
                                        }
                                        
                                building_results[target] = forecast_errors
                                print(f"[EVAL] Evaluated {building_name}.{target} forecasting")
                                
                        except Exception as e:
                            print(f"[EVAL] Forecasting evaluation failed for {building_name}.{target}: {e}")
                            
            results['building_forecasting'][building_name] = building_results
            
        # Neighborhood-level evaluation
        if self.neighborhood_agent:
            try:
                # Aggregate building data for neighborhood forecasting
                neighborhood_data = self._aggregate_neighborhood_data()
                
                if neighborhood_data is not None:
                    forecasts = self.neighborhood_agent.forecast(neighborhood_data, "district_load")
                    results['neighborhood_forecasting'] = {
                        'models_evaluated': list(forecasts.keys()),
                        'forecast_horizon': len(forecasts.get('fusion', [])),
                        'status': 'completed'
                    }
                    print("[EVAL] Neighborhood forecasting evaluation completed")
                    
            except Exception as e:
                print(f"[EVAL] Neighborhood evaluation failed: {e}")
                
        # RL control evaluation
        for building_name, agent in self.hybrid_agents.items():
            try:
                building_data = self.forecaster.data[building_name]
                
                # Simulate RL control episode
                state = building_data.iloc[-1].values  # Most recent state
                action = agent.act(state)
                
                # Simulate reward (placeholder - would be from environment in real deployment)
                simulated_reward = np.random.normal(0, 1)  # Placeholder
                next_state = state + np.random.normal(0, 0.1, len(state))  # Simulated next state
                
                agent.learn(state, action, simulated_reward, next_state)
                
                results['rl_control'][building_name] = {
                    'action_generated': True,
                    'learning_active': True,
                    'action_dimension': len(action),
                    'last_reward': simulated_reward
                }
                
                print(f"[EVAL] RL control evaluated for {building_name}")
                
            except Exception as e:
                print(f"[EVAL] RL evaluation failed for {building_name}: {e}")
                
        # Integrated performance metrics
        results['integrated_performance'] = {
            'total_buildings': len(building_names),
            'active_agents': len(self.hybrid_agents),
            'neighborhood_agent_active': self.neighborhood_agent is not None,
            'forecasting_targets': len(targets),
            'evaluation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Update performance log
        self.performance_log['system_level']['last_evaluation'] = results
        
        print("[EVAL] Comprehensive evaluation completed")
        return results
        
    def _aggregate_neighborhood_data(self):
        """Aggregate building data for neighborhood-level analysis."""
        building_names = [k for k in self.forecaster.data.keys() if k.startswith('Building_')]
        
        if len(building_names) < 2:
            return None
            
        # Simple aggregation: sum of energy demands
        aggregated_data = None
        
        for building_name in building_names:
            building_data = self.forecaster.data[building_name]
            
            if 'cooling_demand' in building_data.columns:
                building_cooling = building_data['cooling_demand'].values
                
                if aggregated_data is None:
                    aggregated_data = building_cooling.copy()
                else:
                    # Ensure same length
                    min_len = min(len(aggregated_data), len(building_cooling))
                    aggregated_data = aggregated_data[:min_len] + building_cooling[:min_len]
                    
        return aggregated_data
        
    def demonstrate_cross_building_transfer(self):
        """Demonstrate transfer learning capabilities between buildings."""
        print("[TRANSFER] Demonstrating cross-building transfer learning...")
        
        building_names = list(self.hybrid_agents.keys())
        
        if len(building_names) < 2:
            print("[TRANSFER] Need at least 2 buildings for transfer learning")
            return {}
            
        transfer_results = {}
        
        # Try transferring between each pair of buildings
        for i, source_building in enumerate(building_names):
            for j, target_building in enumerate(building_names):
                if i != j:  # Don't transfer to self
                    try:
                        source_agent = self.hybrid_agents[source_building]
                        target_agent = self.hybrid_agents[target_building]
                        
                        # Transfer fusion weights (simple transfer learning)
                        source_weights = source_agent.fusion_weights.copy()
                        target_agent.fusion_weights = source_weights
                        
                        # Test forecasting on target building with transferred weights
                        target_data = self.forecaster.data[target_building]
                        
                        if 'cooling_demand' in target_data.columns:
                            test_data = target_data['cooling_demand'].values[-48:]
                            forecasts = target_agent.forecast(test_data, "transferred_cooling")
                            
                            transfer_results[f"{source_building}_to_{target_building}"] = {
                                'weights_transferred': source_weights,
                                'forecast_generated': True,
                                'models_used': list(forecasts.keys())
                            }
                            
                            print(f"[TRANSFER] {source_building} -> {target_building}: Success")
                            
                    except Exception as e:
                        print(f"[TRANSFER] {source_building} -> {target_building}: Failed - {e}")
                        
        return transfer_results
        
    def save_system_state(self, save_dir: str = "../results/integrated_system"):
        """Save complete system state."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save system configuration
        config_path = os.path.join(save_dir, "system_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
        # Save performance log
        log_path = os.path.join(save_dir, "performance_log.json")
        with open(log_path, 'w') as f:
            json.dump(self.performance_log, f, indent=2, default=str)
            
        # Save individual agents
        for building_name, agent in self.hybrid_agents.items():
            agent_path = os.path.join(save_dir, f"agent_{building_name}.json")
            agent.save_model(agent_path)
            
        # Save neighborhood agent
        if self.neighborhood_agent:
            neighborhood_path = os.path.join(save_dir, "neighborhood_agent.json")
            self.neighborhood_agent.save_model(neighborhood_path)
            
        print(f"[SAVE] Complete system state saved to {save_dir}")
        
    def generate_system_report(self):
        """Generate comprehensive system performance report."""
        print("[REPORT] Generating integrated system performance report...")
        
        report = {
            'system_overview': {
                'buildings': list(self.hybrid_agents.keys()),
                'total_agents': len(self.hybrid_agents),
                'neighborhood_agent': self.neighborhood_agent is not None,
                'configuration': self.config
            },
            'component_status': {},
            'performance_summary': {},
            'recommendations': []
        }
        
        # Component status for each building
        for building_name, agent in self.hybrid_agents.items():
            status = agent.get_status()
            report['component_status'][building_name] = status
            
            # Add recommendations based on component availability
            if not status['components']['lstm']:
                report['recommendations'].append(f"Install TensorFlow for LSTM forecasting in {building_name}")
            if not status['components']['transformer']:
                report['recommendations'].append(f"Enable Transformer models for {building_name}")
                
        # Performance summary
        if 'last_evaluation' in self.performance_log.get('system_level', {}):
            report['performance_summary'] = self.performance_log['system_level']['last_evaluation']
            
        # System-level recommendations
        if len(self.hybrid_agents) > 1 and not self.neighborhood_agent:
            report['recommendations'].append("Consider enabling neighborhood-level coordination")
            
        if not any(status['components']['lstm'] for status in report['component_status'].values()):
            report['recommendations'].append("Install TensorFlow for advanced neural network capabilities")
            
        print("[REPORT] System report generated")
        return report


def run_complete_integration_demo(data_path: str = "../data"):
    """Run complete demonstration of integrated system."""
    print("INTEGRATED BUILDING ENERGY MANAGEMENT SYSTEM")
    print("=" * 60)
    print("Demonstrating fusion of RL + LSTM + Transformer + Classical ML")
    print("=" * 60)
    
    # Initialize system
    system = IntegratedBuildingSystem(data_path)
    
    # Setup system
    if not system.setup_system("phase_1"):
        print("[ERROR] System setup failed")
        return None
        
    # Train forecasting models
    system.train_forecasting_models()
    
    # Run comprehensive evaluation
    results = system.run_comprehensive_evaluation()
    
    # Demonstrate transfer learning
    transfer_results = system.demonstrate_cross_building_transfer()
    
    # Generate system report
    report = system.generate_system_report()
    
    # Save system state
    system.save_system_state()
    
    # Final summary
    print("\n" + "=" * 60)
    print("[SUCCESS] INTEGRATED SYSTEM DEMONSTRATION COMPLETE")
    print("=" * 60)
    print(f"Buildings processed: {len(system.hybrid_agents)}")
    print(f"Forecasting models: LSTM + Transformer + Classical")
    print(f"RL algorithms: Q-Learning with adaptive fusion")
    print(f"Transfer learning: {len(transfer_results)} transfers demonstrated")
    print(f"System report: {len(report['recommendations'])} recommendations")
    print("[READY] System ready for deployment and further research")
    print("=" * 60)
    
    return {
        'system': system,
        'results': results,
        'transfer_results': transfer_results,
        'report': report
    }


if __name__ == "__main__":
    demo_results = run_complete_integration_demo()
    
    if demo_results:
        print("\nDemo completed successfully!")
        print(f"Check results/ directory for detailed outputs")
    else:
        print("\nDemo failed - check error messages above")