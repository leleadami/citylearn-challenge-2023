"""
Soft Actor-Critic (SAC) implementation for CityLearn environment.
Supports both centralized and decentralized approaches.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque
from typing import Dict, List, Tuple, Any, Optional
import pickle


class ReplayBuffer:
    """Experience replay buffer for SAC."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def size(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class Actor(keras.Model):
    """Actor network for SAC."""
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(256, activation='relu')
        
        # Mean and log standard deviation outputs
        self.mean = layers.Dense(action_dim)
        self.log_std = layers.Dense(action_dim)
        
        # Constraints on log_std
        self.log_std_min = -20
        self.log_std_max = 2
    
    def call(self, state):
        """Forward pass through actor network."""
        x = self.dense1(state)
        x = self.dense2(x)
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        """Sample action from policy distribution."""
        mean, log_std = self(state)
        std = tf.exp(log_std)
        
        # Sample from normal distribution
        normal = tf.random.normal(tf.shape(mean))
        action = mean + std * normal
        
        # Apply tanh squashing
        squashed_action = tf.tanh(action) * self.max_action
        
        # Calculate log probability
        log_prob = tf.reduce_sum(
            -0.5 * (normal ** 2 + 2 * log_std + np.log(2 * np.pi)), axis=1, keepdims=True
        )
        # Adjust for tanh squashing
        log_prob -= tf.reduce_sum(tf.math.log(1 - (squashed_action / self.max_action) ** 2 + 1e-6), axis=1, keepdims=True)
        
        return squashed_action, log_prob


class Critic(keras.Model):
    """Critic network for SAC (Q-function)."""
    
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()
        
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(256, activation='relu')
        self.output_layer = layers.Dense(1)
    
    def call(self, state, action):
        """Forward pass through critic network."""
        x = tf.concat([state, action], axis=1)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)


class SAC:
    """Soft Actor-Critic algorithm implementation."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float = 1.0,
                 learning_rate: float = 3e-4,
                 discount_factor: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 buffer_capacity: int = 1000000,
                 batch_size: int = 256):
        """
        Initialize SAC agent.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            max_action: Maximum action value
            learning_rate: Learning rate for all networks
            discount_factor: Discount factor for future rewards
            tau: Soft update coefficient
            alpha: Entropy regularization coefficient
            buffer_capacity: Replay buffer capacity
            batch_size: Batch size for training
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount_factor = discount_factor
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Networks
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)
        
        # Optimizers
        self.actor_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic1_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic2_optimizer = keras.optimizers.Adam(learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Initialize target networks
        self._initialize_networks()
        
        # Training history
        self.training_history = {
            'rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'episode_lengths': []
        }
    
    def _initialize_networks(self):
        """Initialize networks with dummy data."""
        dummy_state = tf.random.normal((1, self.state_dim))
        dummy_action = tf.random.normal((1, self.action_dim))
        
        # Initialize actor
        self.actor(dummy_state)
        
        # Initialize critics
        self.critic1(dummy_state, dummy_action)
        self.critic2(dummy_state, dummy_action)
        self.critic1_target(dummy_state, dummy_action)
        self.critic2_target(dummy_state, dummy_action)
        
        # Copy weights to target networks
        self.critic1_target.set_weights(self.critic1.get_weights())
        self.critic2_target.set_weights(self.critic2.get_weights())
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using the actor network."""
        state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        
        if training:
            action, _ = self.actor.sample(state)
        else:
            mean, _ = self.actor(state)
            action = tf.tanh(mean) * self.max_action
        
        return action.numpy()[0]
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        """Single training step."""
        # Train critics
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            # Sample next actions and log probabilities
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # Target Q-values
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = tf.minimum(target_q1, target_q2) - self.alpha * next_log_probs
            
            # Target values
            targets = rewards + self.discount_factor * (1 - dones) * target_q
            
            # Current Q-values
            current_q1 = self.critic1(states, actions)
            current_q2 = self.critic2(states, actions)
            
            # Critic losses
            critic1_loss = tf.reduce_mean(tf.square(current_q1 - targets))
            critic2_loss = tf.reduce_mean(tf.square(current_q2 - targets))
        
        # Update critics
        critic1_grads = tape1.gradient(critic1_loss, self.critic1.trainable_variables)
        critic2_grads = tape2.gradient(critic2_loss, self.critic2.trainable_variables)
        
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))
        
        # Train actor
        with tf.GradientTape() as tape:
            # Sample actions and log probabilities
            new_actions, log_probs = self.actor.sample(states)
            
            # Q-values for new actions
            q1_new = self.critic1(states, new_actions)
            q2_new = self.critic2(states, new_actions)
            q_new = tf.minimum(q1_new, q2_new)
            
            # Actor loss (negative because we want to maximize)
            actor_loss = tf.reduce_mean(self.alpha * log_probs - q_new)
        
        # Update actor
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        return actor_loss, critic1_loss, critic2_loss
    
    def train(self):
        """Train the SAC agent."""
        if self.replay_buffer.size() < self.batch_size:
            return None, None, None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Reshape rewards and dones
        rewards = tf.expand_dims(rewards, 1)
        dones = tf.expand_dims(dones, 1)
        
        # Train
        actor_loss, critic1_loss, critic2_loss = self._train_step(
            states, actions, rewards, next_states, dones
        )
        
        # Soft update target networks
        self._soft_update_targets()
        
        return actor_loss.numpy(), critic1_loss.numpy(), critic2_loss.numpy()
    
    def _soft_update_targets(self):
        """Soft update of target networks."""
        # Update critic1 target
        for target_param, param in zip(self.critic1_target.trainable_variables, 
                                      self.critic1.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)
        
        # Update critic2 target
        for target_param, param in zip(self.critic2_target.trainable_variables, 
                                      self.critic2.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)
    
    def save_model(self, filepath: str):
        """Save the SAC model."""
        # Save actor weights
        self.actor.save_weights(f"{filepath}_actor")
        self.critic1.save_weights(f"{filepath}_critic1")
        self.critic2.save_weights(f"{filepath}_critic2")
        
        # Save hyperparameters
        params = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'max_action': self.max_action,
            'discount_factor': self.discount_factor,
            'tau': self.tau,
            'alpha': self.alpha,
            'training_history': self.training_history
        }
        
        with open(f"{filepath}_params.pkl", 'wb') as f:
            pickle.dump(params, f)
    
    def load_model(self, filepath: str):
        """Load the SAC model."""
        # Load hyperparameters
        with open(f"{filepath}_params.pkl", 'rb') as f:
            params = pickle.load(f)
        
        # Update parameters
        self.state_dim = params['state_dim']
        self.action_dim = params['action_dim']
        self.max_action = params['max_action']
        self.discount_factor = params['discount_factor']
        self.tau = params['tau']
        self.alpha = params['alpha']
        self.training_history = params['training_history']
        
        # Load weights
        self.actor.load_weights(f"{filepath}_actor")
        self.critic1.load_weights(f"{filepath}_critic1")
        self.critic2.load_weights(f"{filepath}_critic2")


class CentralizedSAC:
    """Centralized SAC for multiple building control."""
    
    def __init__(self, env, **kwargs):
        """Initialize centralized SAC."""
        self.env = env
        self.num_buildings = len(env.buildings)
        
        # Calculate total state and action dimensions
        state_dim = sum(len(obs) for obs in env.reset())
        action_dim = self.num_buildings  # One continuous action per building
        
        self.agent = SAC(state_dim, action_dim, **kwargs)
    
    def train(self, num_episodes: int, max_steps_per_episode: int = 1000):
        """Train the centralized SAC agent."""
        for episode in range(num_episodes):
            observations = self.env.reset()
            state = self._flatten_observations(observations)
            
            total_reward = 0
            steps = 0
            
            for step in range(max_steps_per_episode):
                # Select action
                action = self.agent.select_action(state, training=True)
                building_actions = action.tolist()  # Use action directly
                
                # Execute action
                next_observations, reward, done, info = self.env.step(building_actions)
                next_state = self._flatten_observations(next_observations)
                
                # Store experience
                self.agent.store_experience(state, action, reward, next_state, done)
                
                # Train agent
                if step % 10 == 0:  # Train every 10 steps
                    losses = self.agent.train()
                    if losses[0] is not None:
                        self.agent.training_history['actor_losses'].append(losses[0])
                        self.agent.training_history['critic_losses'].append((losses[1] + losses[2]) / 2)
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            
            # Record episode results
            self.agent.training_history['rewards'].append(total_reward)
            self.agent.training_history['episode_lengths'].append(steps)
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.agent.training_history['rewards'][-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
        
        return self.agent.training_history
    
    def _flatten_observations(self, observations: List[np.ndarray]) -> np.ndarray:
        """Flatten list of observations into single state vector."""
        return np.concatenate(observations)


class DecentralizedSAC:
    """Decentralized SAC for multiple building control."""
    
    def __init__(self, env, **kwargs):
        """Initialize decentralized SAC."""
        self.env = env
        self.num_buildings = len(env.buildings)
        
        # Create one agent per building
        self.agents = []
        for i in range(self.num_buildings):
            building = env.buildings[i]
            state_dim = len(building.observation_space.low)
            action_dim = 1  # One continuous action per building
            agent = SAC(state_dim, action_dim, **kwargs)
            self.agents.append(agent)
    
    def train(self, num_episodes: int, max_steps_per_episode: int = 1000):
        """Train all decentralized agents."""
        training_results = {i: {'rewards': [], 'episode_lengths': []} 
                          for i in range(self.num_buildings)}
        
        for episode in range(num_episodes):
            observations = self.env.reset()
            
            total_rewards = [0] * self.num_buildings
            steps = 0
            
            for step in range(max_steps_per_episode):
                # Each agent selects its action
                actions = []
                for i, agent in enumerate(self.agents):
                    action = agent.select_action(observations[i], training=True)
                    actions.append(action[0])  # Extract scalar action
                
                # Execute actions
                next_observations, rewards, done, info = self.env.step(actions)
                
                # Store experiences and train each agent
                for i, agent in enumerate(self.agents):
                    reward_i = rewards[i] if isinstance(rewards, list) else rewards
                    agent.store_experience(
                        observations[i], 
                        [actions[i]], 
                        reward_i,
                        next_observations[i], 
                        done
                    )
                    
                    # Train agent
                    if step % 10 == 0:
                        losses = agent.train()
                        if losses[0] is not None:
                            agent.training_history['actor_losses'].append(losses[0])
                            agent.training_history['critic_losses'].append((losses[1] + losses[2]) / 2)
                    
                    total_rewards[i] += reward_i
                
                observations = next_observations
                steps += 1
                
                if done:
                    break
            
            # Record training history
            for i in range(self.num_buildings):
                training_results[i]['rewards'].append(total_rewards[i])
                training_results[i]['episode_lengths'].append(steps)
                self.agents[i].training_history['rewards'].append(total_rewards[i])
                self.agents[i].training_history['episode_lengths'].append(steps)
            
            if episode % 100 == 0:
                avg_rewards = [np.mean(training_results[i]['rewards'][-100:]) 
                             for i in range(self.num_buildings)]
                print(f"Episode {episode}, Average Rewards: {avg_rewards}")
        
        return training_results
    
    def save_agents(self, filepath_prefix: str):
        """Save all agents."""
        for i, agent in enumerate(self.agents):
            agent.save_model(f"{filepath_prefix}_agent_{i}")
    
    def load_agents(self, filepath_prefix: str):
        """Load all agents."""
        for i, agent in enumerate(self.agents):
            agent.load_model(f"{filepath_prefix}_agent_{i}")