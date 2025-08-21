"""
Q-Learning implementation for CityLearn building energy management.

This module implements tabular Q-Learning, a model-free reinforcement learning
algorithm particularly well-suited for building energy control because:

1. **Sample Efficiency**: Q-Learning can learn effective policies with relatively
   few environment interactions, important for energy systems where data collection
   is expensive and time-consuming.

2. **Interpretability**: The Q-table explicitly shows value estimates for each
   state-action pair, making it easy to understand and debug control decisions.

3. **No Model Required**: Q-Learning learns directly from experience without
   needing a model of building dynamics, which are complex and vary by building.

4. **Proven Convergence**: Under certain conditions, Q-Learning is guaranteed
   to converge to the optimal policy, providing theoretical backing.

The implementation supports both centralized (single agent controlling all
buildings) and decentralized (independent agent per building) approaches for
comparing coordination strategies in multi-building scenarios.

Key Applications in CityLearn:
- HVAC system control (heating/cooling setpoints)
- Battery energy storage management (charge/discharge decisions)
- Demand response participation (load shifting/shedding)
- Renewable energy integration (solar generation utilization)
"""

import numpy as np
import pandas as pd
from collections import defaultdict, deque
import pickle
from typing import Dict, Tuple, List, Optional, Any
import random
from citylearn.citylearn import CityLearnEnv


class QLearningAgent:
    """
    Tabular Q-Learning agent for building energy management and control.
    
    This agent implements the classic Q-Learning algorithm for discrete state-action
    spaces. Since building energy systems often have continuous state spaces (temperatures,
    energy levels, etc.), this implementation includes discretization mechanisms to make
    the problem tractable for tabular methods.
    
    **Q-Learning Algorithm:**
    Q-Learning learns the optimal action-value function Q*(s,a) which represents
    the expected cumulative reward when taking action 'a' in state 's' and following
    the optimal policy thereafter. The update rule is:
    
    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    
    Where:
    - α (learning_rate): Controls how much new information overrides old
    - γ (discount_factor): Importance of future rewards vs immediate rewards
    - r: Immediate reward from environment
    - s': Next state after taking action a in state s
    
    **Building Energy Control Context:**
    - States: Building conditions (temperature, occupancy, energy storage, weather)
    - Actions: Control decisions (HVAC setpoints, battery charge/discharge rates)
    - Rewards: Energy cost savings, comfort maintenance, sustainability metrics
    
    **Discretization Strategy:**
    Continuous building states are discretized into bins to create a finite
    state space suitable for tabular Q-Learning. The discretization quality
    significantly impacts performance - too few bins lose important details,
    too many bins suffer from curse of dimensionality.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 discretization_bins: int = 10):
        """
        Initialize Q-Learning agent with hyperparameters optimized for building energy control.
        
        **Parameter Selection Rationale for Building Energy Systems:**
        
        Args:
            state_dim (int): Dimensionality of building state space. Typical CityLearn states include:
                           - Indoor temperature, outdoor temperature, solar irradiance
                           - Occupancy levels, time of day, day of week
                           - Battery state of charge, grid electricity price
                           - Previous energy consumption, HVAC status
                           
            action_dim (int): Number of discrete control actions available. For buildings:
                            - HVAC: Temperature setpoint adjustments
                            - Battery: Charge/discharge/idle decisions
                            - Combined systems: Joint control strategies
                            
            learning_rate (float): Step size for Q-value updates. Default 0.1 provides:
                                 - Fast learning for building dynamics (hour-to-hour changes)
                                 - Stability against noisy rewards (weather variations)
                                 - Good convergence for typical episode lengths (days/weeks)
                                 
            discount_factor (float): Future reward importance. Default 0.99 reflects:
                                   - Long-term energy cost optimization (monthly bills)
                                   - Delayed consequences of thermal decisions (thermal mass)
                                   - Equipment wear considerations (lifecycle costs)
                                   
            epsilon (float): Initial exploration probability. Start at 1.0 to:
                           - Explore all building operating modes early in training
                           - Avoid getting stuck in local optima (comfort vs efficiency)
                           - Discover counter-intuitive but optimal strategies
                           
            epsilon_decay (float): Exploration reduction rate. 0.995 provides:
                                 - Gradual shift from exploration to exploitation
                                 - Sufficient exploration during seasonal changes
                                 - Stable exploitation once building patterns are learned
                                 
            epsilon_min (float): Minimum exploration to maintain. 0.01 ensures:
                                - Continued adaptation to changing conditions
                                - Robustness to equipment failures or unusual weather
                                - Discovery of new optimal strategies over time
                                
            discretization_bins (int): State space quantization resolution. 10 bins balance:
                                     - Sufficient granularity for control precision
                                     - Manageable Q-table size for reasonable memory usage
                                     - Fast learning convergence with adequate state visits
        """
        # Store agent configuration parameters
        self.state_dim = state_dim                    # Dimensionality of building state vector
        self.action_dim = action_dim                  # Number of available control actions
        self.learning_rate = learning_rate            # Q-value update step size (α in literature)
        self.discount_factor = discount_factor        # Future reward weighting (γ in literature)
        self.epsilon = epsilon                        # Current exploration probability
        self.epsilon_decay = epsilon_decay            # Exploration decay rate per episode
        self.epsilon_min = epsilon_min                # Minimum exploration to maintain
        self.discretization_bins = discretization_bins # State space quantization resolution
        
        # Q-table (state-action value function)
        self.q_table = defaultdict(lambda: np.zeros(action_dim))
        
        # State discretization bounds
        self.state_bounds = None
        
        # Training history
        self.training_history = {
            'rewards': [],
            'episode_lengths': [],
            'epsilon_values': []
        }
    
    def discretize_state(self, state: np.ndarray) -> tuple:
        """
        Discretize continuous state into discrete bins.
        
        Args:
            state: Continuous state vector
            
        Returns:
            Discretized state as tuple
        """
        if self.state_bounds is None:
            # Initialize bounds if not set
            self.state_bounds = [(-1, 1) for _ in range(self.state_dim)]
        
        discretized = []
        for i, value in enumerate(state):
            min_bound, max_bound = self.state_bounds[i]
            # Clip value to bounds
            clipped_value = np.clip(value, min_bound, max_bound)
            # Discretize
            bin_size = (max_bound - min_bound) / self.discretization_bins
            bin_index = int((clipped_value - min_bound) / bin_size)
            bin_index = min(bin_index, self.discretization_bins - 1)  # Ensure within bounds
            discretized.append(bin_index)
        
        return tuple(discretized)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        discrete_state = self.discretize_state(state)
        
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploitation: best action according to Q-table
            return np.argmax(self.q_table[discrete_state])
    
    def update_q_value(self, 
                      state: np.ndarray,
                      action: int,
                      reward: float,
                      next_state: np.ndarray,
                      done: bool) -> None:
        """
        Update Q-value using Bellman equation.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Current Q-value
        current_q = self.q_table[discrete_state][action]
        
        # Target Q-value
        if done:
            target_q = reward
        else:
            next_max_q = np.max(self.q_table[discrete_next_state])
            target_q = reward + self.discount_factor * next_max_q
        
        # Update Q-value
        self.q_table[discrete_state][action] = (
            current_q + self.learning_rate * (target_q - current_q)
        )
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    
    def save_model(self, filepath: str) -> None:
        """Save the Q-table and agent parameters."""
        model_data = {
            'q_table': dict(self.q_table),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'discretization_bins': self.discretization_bins,
            'state_bounds': self.state_bounds,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """Load the Q-table and agent parameters."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(self.action_dim))
        self.q_table.update(model_data['q_table'])
        self.state_dim = model_data['state_dim']
        self.action_dim = model_data['action_dim']
        self.learning_rate = model_data['learning_rate']
        self.discount_factor = model_data['discount_factor']
        self.epsilon = model_data['epsilon']
        self.epsilon_decay = model_data['epsilon_decay']
        self.epsilon_min = model_data['epsilon_min']
        self.discretization_bins = model_data['discretization_bins']
        self.state_bounds = model_data['state_bounds']
        self.training_history = model_data['training_history']


class CentralizedQLearning:
    """
    Centralized Q-Learning for multiple building control.
    Single agent controls all buildings.
    """
    
    def __init__(self, env: CityLearnEnv, **kwargs):
        """
        Initialize centralized Q-Learning.
        
        Args:
            env: CityLearn environment
            **kwargs: Additional arguments for QLearningAgent
        """
        self.env = env
        self.num_buildings = len(env.buildings)
        
        # Calculate total state and action dimensions
        state_dim = sum(len(obs) for obs in env.reset())
        action_dim = self.num_buildings  # Simplified: one discrete action per building
        
        self.agent = QLearningAgent(state_dim, action_dim, **kwargs)
    
    def train(self, num_episodes: int, max_steps_per_episode: int = 1000) -> Dict[str, Any]:
        """
        Train the centralized Q-Learning agent.
        
        Args:
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            Training results
        """
        for episode in range(num_episodes):
            # Reset environment
            observations = self.env.reset()
            state = self._flatten_observations(observations)
            
            total_reward = 0
            steps = 0
            
            for step in range(max_steps_per_episode):
                # Select action
                action = self.agent.select_action(state, training=True)
                
                # Convert single action to building actions
                building_actions = self._convert_action(action)
                
                # Execute action
                next_observations, reward, done, info = self.env.step(building_actions)
                next_state = self._flatten_observations(next_observations)
                
                # Update Q-value
                self.agent.update_q_value(state, action, reward, next_state, done)
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            
            # Decay epsilon
            self.agent.decay_epsilon()
            
            # Record training history
            self.agent.training_history['rewards'].append(total_reward)
            self.agent.training_history['episode_lengths'].append(steps)
            self.agent.training_history['epsilon_values'].append(self.agent.epsilon)
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.agent.training_history['rewards'][-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {self.agent.epsilon:.3f}")
        
        return self.agent.training_history
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the trained agent.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation results
        """
        total_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            observations = self.env.reset()
            state = self._flatten_observations(observations)
            
            total_reward = 0
            steps = 0
            
            while True:
                # Select action (no exploration)
                action = self.agent.select_action(state, training=False)
                building_actions = self._convert_action(action)
                
                # Execute action
                next_observations, reward, done, info = self.env.step(building_actions)
                next_state = self._flatten_observations(next_observations)
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            
            total_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths)
        }
    
    def _flatten_observations(self, observations: List[np.ndarray]) -> np.ndarray:
        """Flatten list of observations into single state vector."""
        return np.concatenate(observations)
    
    def _convert_action(self, action: int) -> List[float]:
        """Convert single discrete action to building actions."""
        # Simple mapping: action determines cooling/heating setpoint adjustments
        # This is a simplified approach - in practice, you'd have more sophisticated action mapping
        actions = []
        for i in range(self.num_buildings):
            if action == i:
                actions.append(1.0)  # Increase cooling/heating
            else:
                actions.append(0.0)  # No change
        return actions


class DecentralizedQLearning:
    """
    Decentralized Q-Learning for multiple building control.
    Each building has its own agent.
    """
    
    def __init__(self, env: CityLearnEnv, **kwargs):
        """
        Initialize decentralized Q-Learning.
        
        Args:
            env: CityLearn environment
            **kwargs: Additional arguments for QLearningAgent
        """
        self.env = env
        self.num_buildings = len(env.buildings)
        
        # Create one agent per building
        self.agents = []
        for i in range(self.num_buildings):
            building = env.buildings[i]
            state_dim = len(building.observation_space.low)
            action_dim = 3  # Simplified: decrease, no change, increase
            agent = QLearningAgent(state_dim, action_dim, **kwargs)
            self.agents.append(agent)
    
    def train(self, num_episodes: int, max_steps_per_episode: int = 1000) -> Dict[str, Any]:
        """
        Train all decentralized agents.
        
        Args:
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            Training results for all agents
        """
        training_results = {i: {'rewards': [], 'episode_lengths': []} 
                          for i in range(self.num_buildings)}
        
        for episode in range(num_episodes):
            # Reset environment
            observations = self.env.reset()
            
            total_rewards = [0] * self.num_buildings
            steps = 0
            
            for step in range(max_steps_per_episode):
                # Each agent selects its action
                actions = []
                for i, agent in enumerate(self.agents):
                    action = agent.select_action(observations[i], training=True)
                    actions.append(self._convert_discrete_action(action))
                
                # Execute actions
                next_observations, rewards, done, info = self.env.step(actions)
                
                # Update each agent
                for i, agent in enumerate(self.agents):
                    agent.update_q_value(
                        observations[i], 
                        actions[i], 
                        rewards[i] if isinstance(rewards, list) else rewards,
                        next_observations[i], 
                        done
                    )
                    total_rewards[i] += rewards[i] if isinstance(rewards, list) else rewards
                
                observations = next_observations
                steps += 1
                
                if done:
                    break
            
            # Decay epsilon for all agents
            for agent in self.agents:
                agent.decay_epsilon()
            
            # Record training history
            for i in range(self.num_buildings):
                training_results[i]['rewards'].append(total_rewards[i])
                training_results[i]['episode_lengths'].append(steps)
            
            if episode % 100 == 0:
                avg_rewards = [np.mean(training_results[i]['rewards'][-100:]) 
                             for i in range(self.num_buildings)]
                print(f"Episode {episode}, Average Rewards: {avg_rewards}")
        
        return training_results
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate all trained agents.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation results
        """
        evaluation_results = {i: {'rewards': [], 'episode_lengths': []} 
                            for i in range(self.num_buildings)}
        
        for episode in range(num_episodes):
            observations = self.env.reset()
            
            total_rewards = [0] * self.num_buildings
            steps = 0
            
            while True:
                # Each agent selects its action (no exploration)
                actions = []
                for i, agent in enumerate(self.agents):
                    action = agent.select_action(observations[i], training=False)
                    actions.append(self._convert_discrete_action(action))
                
                # Execute actions
                next_observations, rewards, done, info = self.env.step(actions)
                
                for i in range(self.num_buildings):
                    total_rewards[i] += rewards[i] if isinstance(rewards, list) else rewards
                
                observations = next_observations
                steps += 1
                
                if done:
                    break
            
            # Record results
            for i in range(self.num_buildings):
                evaluation_results[i]['rewards'].append(total_rewards[i])
                evaluation_results[i]['episode_lengths'].append(steps)
        
        # Calculate statistics
        stats = {}
        for i in range(self.num_buildings):
            stats[f'building_{i}'] = {
                'mean_reward': np.mean(evaluation_results[i]['rewards']),
                'std_reward': np.std(evaluation_results[i]['rewards']),
                'mean_episode_length': np.mean(evaluation_results[i]['episode_lengths']),
                'std_episode_length': np.std(evaluation_results[i]['episode_lengths'])
            }
        
        return stats
    
    def _convert_discrete_action(self, discrete_action: int) -> float:
        """Convert discrete action to continuous action."""
        # Map discrete actions to continuous values
        action_mapping = {0: -1.0, 1: 0.0, 2: 1.0}  # decrease, no change, increase
        return action_mapping.get(discrete_action, 0.0)
    
    def save_agents(self, filepath_prefix: str) -> None:
        """Save all agents."""
        for i, agent in enumerate(self.agents):
            filepath = f"{filepath_prefix}_agent_{i}.pkl"
            agent.save_model(filepath)
    
    def load_agents(self, filepath_prefix: str) -> None:
        """Load all agents."""
        for i, agent in enumerate(self.agents):
            filepath = f"{filepath_prefix}_agent_{i}.pkl"
            agent.load_model(filepath)