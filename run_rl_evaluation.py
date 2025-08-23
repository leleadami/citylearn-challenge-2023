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


class MockCityLearnEnv:
    """
    Mock CityLearn environment for RL testing.
    
    Simulates building energy management environment with:
    - Multi-building observations
    - Continuous/discrete action spaces
    - Energy cost rewards
    """
    
    def __init__(self, building_count: int = 3, obs_dim: int = 28):
        """
        Initialize mock environment.
        
        Args:
            building_count: Number of buildings
            obs_dim: Observation dimension per building
        """
        self.building_count = building_count
        self.obs_dim = obs_dim
        self.current_step = 0
        self.max_steps = 1000
        
        # Load actual building data for realistic simulation
        self.building_data = self._load_building_data()
        
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
                
                # Reward based on energy efficiency
                energy_consumption = row['cooling_demand'] + row['non_shiftable_load']
                
                # Penalize high consumption, reward energy savings
                efficiency_reward = -energy_consumption / 10.0
                
                # Action penalty (avoid extreme actions)
                action_penalty = -0.1 * abs(action)
                
                # Comfort reward (maintain temperature)
                temp = row['indoor_dry_bulb_temperature']
                comfort_reward = -0.1 * abs(temp - 22.0)  # Target 22°C
                
                reward = efficiency_reward + action_penalty + comfort_reward
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
                # Fallback random reward and observation
                rewards.append(np.random.uniform(-1, 0))
                next_observations.append(np.random.uniform(-1, 1, self.obs_dim))
        
        done = self.current_step >= self.max_steps
        info = {'step': self.current_step}
        
        return next_observations, rewards, done, info


class RLEvaluator:
    """Reinforcement Learning evaluation system."""
    
    def __init__(self):
        """Initialize RL evaluator."""
        self.results = {}
        
    def evaluate_q_learning(self, episodes: int = 100):
        """Evaluate Q-Learning agents."""
        print("\\n" + "="*60)
        print("Q-LEARNING EVALUATION")
        print("="*60)
        
        env = MockCityLearnEnv()
        results = {}
        
        # Centralized Q-Learning
        print("\\n[CENTRALIZED] Q-Learning Training...")
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
        print("\\n[DECENTRALIZED] Q-Learning Training...")
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
        print("\\n" + "="*60)
        print("SAC EVALUATION")
        print("="*60)
        
        env = MockCityLearnEnv()
        results = {}
        
        # Centralized SAC
        print("\\n[CENTRALIZED] SAC Training...")
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
        print("\\n[DECENTRALIZED] SAC Training...")
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
        """Create RL training visualizations."""
        print("\\n[VISUALIZATION] Creating RL training plots...")
        
        os.makedirs('results/visualizations', exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Reinforcement Learning Training Results\\nCityLearn Challenge 2023', 
                     fontsize=16, fontweight='bold')
        
        # Q-Learning Training Curves
        if 'qlearning' in self.results:
            qlearn_results = self.results['qlearning']
            
            # Centralized Q-Learning
            centralized_rewards = qlearn_results['centralized_qlearning']['training_rewards']
            axes[0,0].plot(centralized_rewards, label='Centralized Q-Learning', 
                          color='#2E86AB', linewidth=2)
            
            # Decentralized Q-Learning
            decentralized_rewards = qlearn_results['decentralized_qlearning']['training_rewards']
            axes[0,0].plot(decentralized_rewards, label='Decentralized Q-Learning', 
                          color='#A23B72', linewidth=2)
            
            axes[0,0].set_title('Q-Learning Training Progress')
            axes[0,0].set_xlabel('Episodes')
            axes[0,0].set_ylabel('Episode Reward')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        
        # SAC Training Curves
        if 'sac' in self.results:
            sac_results = self.results['sac']
            
            # Centralized SAC
            centralized_rewards = sac_results['centralized_sac']['training_rewards']
            axes[0,1].plot(centralized_rewards, label='Centralized SAC', 
                          color='#F18F01', linewidth=2)
            
            # Decentralized SAC
            decentralized_rewards = sac_results['decentralized_sac']['training_rewards']
            axes[0,1].plot(decentralized_rewards, label='Decentralized SAC', 
                          color='#C73E1D', linewidth=2)
            
            axes[0,1].set_title('SAC Training Progress')
            axes[0,1].set_xlabel('Episodes')
            axes[0,1].set_ylabel('Episode Reward')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # Algorithm Comparison
        all_methods = []
        final_rewards = []
        
        if 'qlearning' in self.results:
            qlearn = self.results['qlearning']
            all_methods.extend(['Centralized Q-Learning', 'Decentralized Q-Learning'])
            final_rewards.extend([
                np.mean(qlearn['centralized_qlearning']['training_rewards'][-10:]),
                np.mean(qlearn['decentralized_qlearning']['training_rewards'][-10:])
            ])
        
        if 'sac' in self.results:
            sac = self.results['sac']
            all_methods.extend(['Centralized SAC', 'Decentralized SAC'])
            final_rewards.extend([
                np.mean(sac['centralized_sac']['training_rewards'][-10:]),
                np.mean(sac['decentralized_sac']['training_rewards'][-10:])
            ])
        
        if all_methods:
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(all_methods)]
            bars = axes[1,0].bar(all_methods, final_rewards, color=colors, alpha=0.8)
            axes[1,0].set_title('Final Performance Comparison')
            axes[1,0].set_ylabel('Average Final Reward')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            for bar, reward in zip(bars, final_rewards):
                axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                              f'{reward:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Learning Convergence
        if 'qlearning' in self.results and 'sac' in self.results:
            qlearn_smooth = self._smooth_curve(
                self.results['qlearning']['centralized_qlearning']['training_rewards']
            )
            sac_smooth = self._smooth_curve(
                self.results['sac']['centralized_sac']['training_rewards']
            )
            
            axes[1,1].plot(qlearn_smooth, label='Q-Learning (Centralized)', 
                          color='#2E86AB', linewidth=3)
            axes[1,1].plot(sac_smooth, label='SAC (Centralized)', 
                          color='#F18F01', linewidth=3)
            
            axes[1,1].set_title('Learning Convergence (Smoothed)')
            axes[1,1].set_xlabel('Training Progress')
            axes[1,1].set_ylabel('Smoothed Reward')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/visualizations/rl_training_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: rl_training_analysis.png")
    
    def _smooth_curve(self, values: List[float], window: int = 10) -> List[float]:
        """Apply moving average smoothing."""
        if len(values) < window:
            return values
        
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            smoothed.append(np.mean(values[start:end]))
        
        return smoothed
    
    def save_results(self):
        """Save RL evaluation results."""
        os.makedirs('results/rl_experiments', exist_ok=True)
        
        with open('results/rl_experiments/rl_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\\n[SAVE] RL results saved to: results/rl_experiments/rl_results.json")


def main():
    """Main RL evaluation function."""
    evaluator = RLEvaluator()
    
    print("\\n" + "="*80)
    print("CITYLEARN 2023 - REINFORCEMENT LEARNING EVALUATION")
    print("Q-Learning and SAC with Centralized/Decentralized Approaches")
    print("="*80)
    
    # Evaluate Q-Learning
    evaluator.evaluate_q_learning(episodes=80)
    
    # Evaluate SAC  
    evaluator.evaluate_sac(episodes=40)
    
    # Create visualizations
    evaluator.create_rl_visualizations()
    
    # Save results
    evaluator.save_results()
    
    print("\\n" + "="*80)
    print("REINFORCEMENT LEARNING EVALUATION COMPLETE")
    print("Results: results/rl_experiments/rl_results.json")
    print("Visualizations: results/visualizations/rl_training_analysis.png")
    print("="*80)


if __name__ == "__main__":
    main()