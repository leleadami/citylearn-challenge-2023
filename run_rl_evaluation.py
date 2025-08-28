"""
Reinforcement Learning Evaluation for CityLearn Challenge 2023
Q-Learning and SAC agents with centralized/decentralized approaches

This script evaluates RL agents on building energy management:
1. Q-Learning (centralized and decentralized)
2. SAC (centralized and decentralized)  
3. Performance comparison with forecasting models
4. Training curves and convergence analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from src.rl.q_learning_agent import CentralizedQLearning, DecentralizedQLearning
from src.rl.sac_agent import CentralizedSAC, DecentralizedSAC
from src.rl.reward_functions import create_reward_function, BaseRewardFunction


class MockCityLearnEnv:
    """
    Mock CityLearn environment for RL testing.
    
    Simulates building energy management environment with:
    - Multi-building observations
    - Continuous/discrete action spaces
    - Energy cost rewards
    """
    
    def __init__(self, building_count: int = 3, obs_dim: int = 28, reward_type: str = 'balanced'):
        """
        Initialize mock environment.
        
        Args:
            building_count: Number of buildings
            obs_dim: Observation dimension per building
            reward_type: Type of reward function to use
        """
        self.building_count = building_count
        self.obs_dim = obs_dim
        self.current_step = 0
        self.max_steps = 1000
        
        # Load actual building data for realistic simulation
        self.building_data = self._load_building_data()
        
        # Initialize reward function
        self.reward_function = create_reward_function(reward_type)
        print(f"Using reward function: {self.reward_function.name}")
        
    def _load_building_data(self) -> Dict[str, pd.DataFrame]:
        """Load building data for realistic rewards."""
        buildings = {}
        phases = [
            'citylearn_challenge_2023_phase_2_online_evaluation_1',
            'citylearn_challenge_2023_phase_2_online_evaluation_2',
            'citylearn_challenge_2023_phase_2_online_evaluation_3'
        ]
        
        for building_id in [1, 2, 3]:
            all_data = []
            for phase in phases:
                file_path = f'data/{phase}/Building_{building_id}.csv'
                if os.path.exists(file_path):
                    data = pd.read_csv(file_path)
                    all_data.append(data)
            
            if all_data:
                combined = pd.concat(all_data, ignore_index=True)
                combined = combined.drop_duplicates().reset_index(drop=True)
                buildings[f'Building_{building_id}'] = combined
        
        return buildings
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        
        # Generate initial observations
        observations = []
        for i in range(self.building_count):
            building_name = f'Building_{i+1}'
            if building_name in self.building_data:
                # Use actual building features
                data = self.building_data[building_name]
                row = data.iloc[self.current_step % len(data)]
                obs = np.array([
                    row['hour'] / 24.0,
                    row['indoor_dry_bulb_temperature'] / 50.0,
                    row['indoor_relative_humidity'] / 100.0,
                    row['non_shiftable_load'] / 10.0,
                    row['cooling_demand'] / 10.0,
                    row['occupant_count'] / 5.0
                ])
                
                # Pad to obs_dim
                if len(obs) < self.obs_dim:
                    obs = np.pad(obs, (0, self.obs_dim - len(obs)))
                else:
                    obs = obs[:self.obs_dim]
            else:
                # Random observations as fallback
                obs = np.random.uniform(-1, 1, self.obs_dim)
            
            observations.append(obs)
        
        return observations
    
    def step(self, actions: List[float]):
        """
        Take environment step.
        
        Args:
            actions: Actions for each building
            
        Returns:
            next_observations, rewards, done, info
        """
        self.current_step += 1
        
        # Generate next observations
        next_observations = []
        rewards = []
        
        for i, action in enumerate(actions):
            building_name = f'Building_{i+1}'
            
            if building_name in self.building_data:
                # Use actual building data for rewards
                data = self.building_data[building_name]
                row = data.iloc[self.current_step % len(data)]
                
                # Create state dictionaries for reward function
                state = {
                    'hour': row['hour'],
                    'indoor_dry_bulb_temperature': row['indoor_dry_bulb_temperature'],
                    'indoor_relative_humidity': row['indoor_relative_humidity'],
                    'occupant_count': row['occupant_count']
                }
                
                next_state = {
                    'cooling_demand': row['cooling_demand'],
                    'heating_demand': row.get('heating_demand', 0),
                    'non_shiftable_load': row['non_shiftable_load'],
                    'indoor_dry_bulb_temperature': row['indoor_dry_bulb_temperature'],
                    'solar_generation': row.get('solar_generation', 0),
                    'hour': row['hour']
                }
                
                # Calculate reward using modular reward function
                reward = self.reward_function.calculate_reward(state, action, next_state)
                rewards.append(reward)
                
                # Next observation
                obs = np.array([
                    row['hour'] / 24.0,
                    row['indoor_dry_bulb_temperature'] / 50.0,
                    row['indoor_relative_humidity'] / 100.0,
                    row['non_shiftable_load'] / 10.0,
                    row['cooling_demand'] / 10.0,
                    row['occupant_count'] / 5.0
                ])
                
                if len(obs) < self.obs_dim:
                    obs = np.pad(obs, (0, self.obs_dim - len(obs)))
                else:
                    obs = obs[:self.obs_dim]
                    
                next_observations.append(obs)
            else:
                # Fallback reward: random but positive-leaning
                rewards.append(np.random.uniform(0.3, 0.7))  # Slightly positive baseline
                next_observations.append(np.random.uniform(-1, 1, self.obs_dim))
        
        done = self.current_step >= self.max_steps
        info = {'step': self.current_step}
        
        return next_observations, rewards, done, info


class RLEvaluator:
    """Reinforcement Learning evaluation system."""
    
    def __init__(self, reward_type: str = 'balanced'):
        """Initialize RL evaluator.
        
        Args:
            reward_type: Type of reward function to use for all agents
        """
        self.results = {}
        self.reward_type = reward_type
        print(f"\nFunzione Ricompensa: {reward_type.upper()}")
        
    def evaluate_q_learning(self, episodes: int = 100):
        """Evaluate Q-Learning agents."""
        print("\n" + "="*60)
        print("Q-LEARNING EVALUATION")
        print("="*60)
        
        env = MockCityLearnEnv(reward_type=self.reward_type)
        results = {}
        
        # Centralized Q-Learning
        print("\n[CENTRALIZED] Q-Learning Training...")
        centralized_ql = CentralizedQLearning(
            building_count=3,
            learning_rate=0.1,
            epsilon=0.3,
            epsilon_decay=0.995
        )
        
        for episode in range(episodes):
            episode_reward, episode_length = centralized_ql.train_episode(env)
            
            if episode % 20 == 0:
                print(f"  Episode {episode}: Reward={episode_reward:.2f}, "
                      f"Length={episode_length}, Epsilon={centralized_ql.agent.epsilon:.3f}")
        
        results['centralized_qlearning'] = {
            'training_rewards': centralized_ql.training_history['episode_rewards'],
            'final_epsilon': centralized_ql.agent.epsilon,
            'q_table_size': len(centralized_ql.agent.q_table)
        }
        
        # Decentralized Q-Learning
        print("\n[DECENTRALIZED] Q-Learning Training...")
        decentralized_ql = DecentralizedQLearning(
            building_count=3,
            learning_rate=0.1,
            epsilon=0.3,
            epsilon_decay=0.995
        )
        
        for episode in range(episodes):
            episode_reward, episode_length = decentralized_ql.train_episode(env)
            
            if episode % 20 == 0:
                print(f"  Episode {episode}: Reward={episode_reward:.2f}, "
                      f"Length={episode_length}")
        
        results['decentralized_qlearning'] = {
            'training_rewards': decentralized_ql.training_history['episode_rewards'],
            'agent_q_table_sizes': [len(agent.q_table) for agent in decentralized_ql.agents]
        }
        
        self.results['qlearning'] = results
        return results
    
    def evaluate_sac(self, episodes: int = 50):
        """Evaluate SAC agents."""
        print("\n" + "="*60)
        print("SAC EVALUATION")
        print("="*60)
        
        env = MockCityLearnEnv(reward_type=self.reward_type)
        results = {}
        
        # Centralized SAC
        print("\n[CENTRALIZED] SAC Training...")
        centralized_sac = CentralizedSAC(
            building_count=3,
            obs_dim=28,
            learning_rate=3e-4,
            buffer_size=10000,
            batch_size=64
        )
        
        for episode in range(episodes):
            episode_reward = centralized_sac.train_episode(env, max_steps=500)
            
            if episode % 10 == 0:
                print(f"  Episode {episode}: Reward={episode_reward:.2f}")
        
        results['centralized_sac'] = {
            'training_rewards': centralized_sac.training_history['episode_rewards'],
            'actor_losses': centralized_sac.training_history['actor_losses'][-100:],  # Last 100
            'critic_losses': centralized_sac.training_history['critic_losses'][-100:]
        }
        
        # Decentralized SAC
        print("\n[DECENTRALIZED] SAC Training...")
        decentralized_sac = DecentralizedSAC(
            building_count=3,
            obs_dim=28,
            learning_rate=3e-4,
            buffer_size=10000,
            batch_size=64
        )
        
        for episode in range(episodes):
            episode_reward = decentralized_sac.train_episode(env, max_steps=500)
            
            if episode % 10 == 0:
                print(f"  Episode {episode}: Reward={episode_reward:.2f}")
        
        results['decentralized_sac'] = {
            'training_rewards': decentralized_sac.training_history['episode_rewards']
        }
        
        self.results['sac'] = results
        return results
    
    def create_rl_visualizations(self):
        """Create comprehensive RL visualizations."""
        print("\nGenerazione visualizzazioni RL...")
        
        if not self.results:
            print("Warning: No results to visualize")
            return
        
        # Use visualization utilities
        from src.utils.visualization import create_complete_rl_evaluation_charts
        create_complete_rl_evaluation_charts(self.results)
        
        print("Visualizzazioni RL complete usando utilities!")
    
    def save_results(self):
        """Save RL evaluation results to JSON file."""
        print("\nSalvataggio risultati RL...")
        os.makedirs('results/rl_experiments', exist_ok=True)
        
        # Save main results
        results_path = 'results/rl_experiments/rl_results.json'
        with open(results_path, 'w') as f:
            import json
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"  Risultati salvati in {results_path}")


def main(reward_type: str = 'balanced'):
    """Main RL evaluation function.
    
    Args:
        reward_type: Type of reward function to use ('balanced', 'efficiency', 
                    'comfort', 'cost', 'sustainability', 'multi_objective')
    """
    print("\n" + "="*80)
    print("CITYLEARN 2023 - REINFORCEMENT LEARNING EVALUATION")
    print("Q-Learning and SAC with Centralized/Decentralized Approaches")
    print("="*80)
    
    evaluator = RLEvaluator(reward_type=reward_type)  # change reward function here
    
    # Evaluate Q-Learning
    evaluator.evaluate_q_learning(episodes=80)
    
    # Evaluate SAC  
    evaluator.evaluate_sac(episodes=40)
    
    # Create visualizations
    evaluator.create_rl_visualizations()
    
    # Save results
    evaluator.save_results()
    
    print("\n" + "="*80)
    print("REINFORCEMENT LEARNING EVALUATION COMPLETE")
    print(f"Reward Function Used: {reward_type}")
    print("Results: results/rl_experiments/rl_results.json")
    print("Visualizations: results/visualizations/rl_training_analysis.png")
    print("="*80)


def compare_reward_functions():
    """Compare different reward functions with shorter episodes."""
    print("\n" + "="*80)
    print("REWARD FUNCTION COMPARISON")
    print("="*80)
    
    reward_types = ['efficiency', 'comfort', 'balanced', 'cost', 'sustainability']
    comparison_results = {}
    
    for reward_type in reward_types:
        print(f"\n{'='*20} {reward_type.upper()} REWARD {'='*20}")
        
        evaluator = RLEvaluator(reward_type=reward_type)
        
        # Run shorter evaluations for comparison
        evaluator.evaluate_q_learning(episodes=30)
        evaluator.evaluate_sac(episodes=20)
        
        # Store results for comparison
        comparison_results[reward_type] = evaluator.results
    
    print(f"\n{'='*80}")
    print("REWARD FUNCTION COMPARISON COMPLETE")
    print(f"{'='*80}")
    
    return comparison_results


if __name__ == "__main__":
    main()