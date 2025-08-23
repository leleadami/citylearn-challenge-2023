"""
Soft Actor-Critic (SAC) Agent for CityLearn Challenge 2023

Implements both centralized and decentralized SAC approaches
for building energy management with continuous action spaces.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
import json
import os
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class SACAgent:
    """
    Soft Actor-Critic agent for continuous control.
    
    SAC is particularly well-suited for building energy control because:
    1. Handles continuous action spaces (HVAC setpoints)
    2. Sample efficient learning
    3. Automatic entropy regularization
    4. Robust to hyperparameters
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 agent_id: str = "sac_agent",
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 buffer_size: int = 100000,
                 batch_size: int = 256):
        """
        Initialize SAC agent.
        
        Args:
            state_dim: State space dimensionality
            action_dim: Action space dimensionality  
            agent_id: Unique agent identifier
            learning_rate: Learning rate for all networks
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Entropy regularization coefficient
            buffer_size: Replay buffer size
            batch_size: Training batch size
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Networks
        self.actor = self._build_actor()
        self.critic1 = self._build_critic()
        self.critic2 = self._build_critic()
        self.target_critic1 = self._build_critic()
        self.target_critic2 = self._build_critic()
        
        # Initialize target networks
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())
        
        # Optimizers
        self.actor_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic1_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic2_optimizer = keras.optimizers.Adam(learning_rate)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Training metrics
        self.training_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.episode_count = 0
    
    def _build_actor(self) -> Model:
        """
        Build actor network for continuous actions.
        
        Returns:
            Actor network outputting mean and log_std
        """
        inputs = Input(shape=(self.state_dim,))
        
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dense(256, activation='relu')(x)
        
        # Mean of action distribution
        mean = layers.Dense(self.action_dim, activation='tanh')(x)
        
        # Log standard deviation (clamped for stability)
        log_std = layers.Dense(self.action_dim, activation='tanh')(x)
        log_std = layers.Lambda(lambda x: tf.clip_by_value(x, -20, 2))(log_std)
        
        actor = Model(inputs, [mean, log_std], name=f'{self.agent_id}_actor')
        return actor
    
    def _build_critic(self) -> Model:
        """
        Build critic network for Q-value estimation.
        
        Returns:
            Critic network outputting Q-values
        """
        state_input = Input(shape=(self.state_dim,))
        action_input = Input(shape=(self.action_dim,))
        
        # Concatenate state and action
        concat = layers.Concatenate()([state_input, action_input])
        
        x = layers.Dense(256, activation='relu')(concat)
        x = layers.Dense(256, activation='relu')(x)
        q_value = layers.Dense(1)(x)
        
        critic = Model([state_input, action_input], q_value, 
                      name=f'{self.agent_id}_critic')
        return critic
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            training: Whether to add exploration noise
            
        Returns:
            Selected action
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        mean, log_std = self.actor(state)
        
        if training:
            # Sample from policy distribution
            std = tf.exp(log_std)
            normal = tf.random.normal(tf.shape(mean))
            action = mean + std * normal
        else:
            # Use deterministic policy
            action = mean
        
        # Clip action to [-1, 1] range
        action = tf.clip_by_value(action, -1, 1)
        return action.numpy()[0]
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        """Store transition in replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self) -> Tuple[float, float]:
        """
        Perform one SAC training step.
        
        Returns:
            Actor loss and critic loss
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Train critics
        critic_loss = self._train_critics(states, actions, rewards, next_states, dones)
        
        # Train actor
        actor_loss = self._train_actor(states)
        
        # Soft update target networks
        self._soft_update_targets()
        
        return actor_loss, critic_loss
    
    @tf.function
    def _train_critics(self, states, actions, rewards, next_states, dones):
        """Train critic networks."""
        # Get next actions from current policy
        next_mean, next_log_std = self.actor(next_states)
        next_std = tf.exp(next_log_std)
        next_normal = tf.random.normal(tf.shape(next_mean))
        next_actions = next_mean + next_std * next_normal
        next_actions = tf.clip_by_value(next_actions, -1, 1)
        
        # Calculate entropy bonus
        next_log_prob = self._log_prob(next_actions, next_mean, next_log_std)
        
        # Target Q-values
        target_q1 = self.target_critic1([next_states, next_actions])
        target_q2 = self.target_critic2([next_states, next_actions])
        target_q = tf.minimum(target_q1, target_q2)
        
        target_q = rewards + self.gamma * (1 - dones) * (target_q - self.alpha * next_log_prob)
        
        # Train critics
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            q1 = self.critic1([states, actions])
            q2 = self.critic2([states, actions])
            
            critic1_loss = tf.reduce_mean(tf.square(q1 - target_q))
            critic2_loss = tf.reduce_mean(tf.square(q2 - target_q))
        
        # Update critics
        grad1 = tape1.gradient(critic1_loss, self.critic1.trainable_variables)
        grad2 = tape2.gradient(critic2_loss, self.critic2.trainable_variables)
        
        self.critic1_optimizer.apply_gradients(zip(grad1, self.critic1.trainable_variables))
        self.critic2_optimizer.apply_gradients(zip(grad2, self.critic2.trainable_variables))
        
        return (critic1_loss + critic2_loss) / 2
    
    @tf.function
    def _train_actor(self, states):
        """Train actor network."""
        with tf.GradientTape() as tape:
            mean, log_std = self.actor(states)
            std = tf.exp(log_std)
            normal = tf.random.normal(tf.shape(mean))
            actions = mean + std * normal
            actions = tf.clip_by_value(actions, -1, 1)
            
            log_prob = self._log_prob(actions, mean, log_std)
            
            q1 = self.critic1([states, actions])
            q2 = self.critic2([states, actions])
            q = tf.minimum(q1, q2)
            
            # Actor loss: maximize Q - entropy
            actor_loss = tf.reduce_mean(self.alpha * log_prob - q)
        
        # Update actor
        grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grad, self.actor.trainable_variables))
        
        return actor_loss
    
    def _log_prob(self, actions, mean, log_std):
        """Calculate log probability of actions."""
        std = tf.exp(log_std)
        normal_dist = tf.random.normal(tf.shape(mean), mean, std)
        log_prob = -0.5 * tf.reduce_sum(
            tf.square((actions - mean) / std) + 2 * log_std + tf.math.log(2 * np.pi),
            axis=-1, keepdims=True
        )
        return log_prob
    
    def _soft_update_targets(self):
        """Soft update target networks."""
        for target, source in [(self.target_critic1, self.critic1),
                              (self.target_critic2, self.critic2)]:
            for target_param, source_param in zip(target.trainable_variables,
                                                 source.trainable_variables):
                target_param.assign(
                    self.tau * source_param + (1 - self.tau) * target_param
                )
    
    def save_agent(self, directory: str):
        """Save agent networks."""
        os.makedirs(directory, exist_ok=True)
        self.actor.save(f"{directory}/{self.agent_id}_actor.h5")
        self.critic1.save(f"{directory}/{self.agent_id}_critic1.h5")
        self.critic2.save(f"{directory}/{self.agent_id}_critic2.h5")
    
    def load_agent(self, directory: str):
        """Load agent networks."""
        self.actor = keras.models.load_model(f"{directory}/{self.agent_id}_actor.h5")
        self.critic1 = keras.models.load_model(f"{directory}/{self.agent_id}_critic1.h5")
        self.critic2 = keras.models.load_model(f"{directory}/{self.agent_id}_critic2.h5")


class CentralizedSAC:
    """
    Centralized SAC for multi-building control.
    """
    
    def __init__(self, building_count: int = 3, obs_dim: int = 28, **kwargs):
        """
        Initialize centralized SAC controller.
        
        Args:
            building_count: Number of buildings
            obs_dim: Observation dimension per building
            **kwargs: Arguments for SACAgent
        """
        self.building_count = building_count
        total_state_dim = building_count * obs_dim
        total_action_dim = building_count
        
        self.agent = SACAgent(
            state_dim=total_state_dim,
            action_dim=total_action_dim,
            agent_id="centralized_sac",
            **kwargs
        )
        
        self.training_history = {
            'episode_rewards': [],
            'actor_losses': [],
            'critic_losses': []
        }
    
    def train_episode(self, env, max_steps: int = 8760):
        """Train centralized SAC agent."""
        observations = env.reset()
        episode_reward = 0
        
        # Flatten multi-building observations
        if isinstance(observations, list):
            state = np.concatenate(observations)
        else:
            state = observations.flatten()
        
        for step in range(max_steps):
            # Select actions
            actions = self.agent.select_action(state, training=True)
            
            # Environment step
            next_observations, rewards, done, info = env.step(actions)
            
            # Process next state
            if isinstance(next_observations, list):
                next_state = np.concatenate(next_observations)
            else:
                next_state = next_observations.flatten()
            
            # Calculate total reward
            total_reward = np.sum(rewards) if isinstance(rewards, list) else rewards
            
            # Store transition
            self.agent.store_transition(state, actions, total_reward, next_state, done)
            
            # Train agent
            if len(self.agent.replay_buffer) >= self.agent.batch_size:
                actor_loss, critic_loss = self.agent.train_step()
                self.training_history['actor_losses'].append(float(actor_loss))
                self.training_history['critic_losses'].append(float(critic_loss))
            
            episode_reward += total_reward
            state = next_state
            
            if done:
                break
        
        self.training_history['episode_rewards'].append(episode_reward)
        self.agent.episode_count += 1
        
        return episode_reward


class DecentralizedSAC:
    """
    Decentralized SAC with independent agents per building.
    """
    
    def __init__(self, building_count: int = 3, obs_dim: int = 28, **kwargs):
        """
        Initialize decentralized SAC controllers.
        
        Args:
            building_count: Number of buildings
            obs_dim: Observation dimension per building
            **kwargs: Arguments for SACAgent
        """
        self.building_count = building_count
        self.agents = [
            SACAgent(
                state_dim=obs_dim,
                action_dim=1,
                agent_id=f"building_{i}_sac",
                **kwargs
            ) for i in range(building_count)
        ]
        
        self.training_history = {
            'episode_rewards': [],
            'actor_losses': [[] for _ in range(building_count)],
            'critic_losses': [[] for _ in range(building_count)]
        }
    
    def train_episode(self, env, max_steps: int = 8760):
        """Train all decentralized SAC agents."""
        observations = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Each agent selects action for its building
            actions = []
            for i, agent in enumerate(self.agents):
                obs = observations[i] if isinstance(observations, list) else observations
                action = agent.select_action(obs, training=True)
                actions.append(action[0])  # Single action per building
            
            # Environment step
            next_observations, rewards, done, info = env.step(actions)
            
            # Store transitions and train each agent
            total_reward = 0
            for i, agent in enumerate(self.agents):
                obs = observations[i] if isinstance(observations, list) else observations
                next_obs = next_observations[i] if isinstance(next_observations, list) else next_observations
                reward = rewards[i] if isinstance(rewards, list) else rewards / len(self.agents)
                
                agent.store_transition(obs, [actions[i]], reward, next_obs, done)
                
                if len(agent.replay_buffer) >= agent.batch_size:
                    actor_loss, critic_loss = agent.train_step()
                    self.training_history['actor_losses'][i].append(float(actor_loss))
                    self.training_history['critic_losses'][i].append(float(critic_loss))
                
                total_reward += reward
            
            episode_reward += total_reward
            observations = next_observations
            
            if done:
                break
        
        self.training_history['episode_rewards'].append(episode_reward)
        for agent in self.agents:
            agent.episode_count += 1
        
        return episode_reward
    
    def save_agents(self, directory: str):
        """Save all decentralized agents."""
        for i, agent in enumerate(self.agents):
            agent.save_agent(f"{directory}/building_{i}")
    
    def load_agents(self, directory: str):
        """Load all decentralized agents."""
        for i, agent in enumerate(self.agents):
            agent.load_agent(f"{directory}/building_{i}")