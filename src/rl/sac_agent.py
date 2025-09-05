"""
Agente Soft Actor-Critic (SAC) per CityLearn Challenge 2023

Implementa approcci SAC sia centralizzati che decentralizzati
per la gestione energetica degli edifici con spazi di azione continui.
"""

from __future__ import annotations
import numpy as np
import tensorflow as tf
import keras
from keras import layers, Model, Input
from keras.optimizers import Adam
from tensorflow.python.framework import tensor_spec
import json
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import deque
import random


class SACAgent:
    """
    Agente Soft Actor-Critic per controllo continuo.
    
    SAC è particolarmente adatto per il controllo energetico degli edifici perché:
    1. Gestisce spazi di azione continui (setpoint HVAC)
    2. Apprendimento efficiente in termini di campioni
    3. Regolarizzazione dell'entropia automatica
    4. Robusto agli iperparametri
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
        Inizializza agente SAC.
        
        Args:
            state_dim: Dimensionalità dello spazio degli stati
            action_dim: Dimensionalità dello spazio delle azioni
            agent_id: Identificatore unico dell'agente
            learning_rate: Learning rate per tutte le reti
            gamma: Fattore di sconto
            tau: Coefficiente di aggiornamento soft
            alpha: Coefficiente di regolarizzazione dell'entropia
            buffer_size: Dimensione del buffer di replay
            batch_size: Dimensione del batch di addestramento
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Reti con tipizzazione esplicita
        self.actor: Model = self._build_actor()
        self.critic1: Model = self._build_critic()
        self.critic2: Model = self._build_critic()
        self.target_critic1: Model = self._build_critic()
        self.target_critic2: Model = self._build_critic()
        
        # Inizializza reti target
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())
        
        # Ottimizzatori con tipizzazione appropriata
        self.actor_optimizer: Adam = Adam(learning_rate=learning_rate)
        self.critic1_optimizer: Adam = Adam(learning_rate=learning_rate)
        self.critic2_optimizer: Adam = Adam(learning_rate=learning_rate)
        
        # Buffer di replay
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Metriche di addestramento
        self.training_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.episode_count = 0
    
    def _build_actor(self) -> Model:
        """
        Costruisce rete actor per azioni continue.
        
        Returns:
            Rete actor che produce media e log_std
        """
        inputs = Input(shape=(self.state_dim,))
        
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dense(256, activation='relu')(x)
        
        # Media della distribuzione delle azioni
        mean = layers.Dense(self.action_dim, activation='tanh')(x)
        
        # Log deviazione standard (limitata per stabilità)
        log_std = layers.Dense(self.action_dim, activation='tanh')(x)
        log_std = layers.Lambda(lambda x: tf.clip_by_value(x, -20, 2))(log_std)
        
        actor = Model(inputs, [mean, log_std], name=f'{self.agent_id}_actor')
        return actor
    
    def _build_critic(self) -> Model:
        """
        Costruisce rete critic per stima dei Q-values.
        
        Returns:
            Rete critic che produce Q-values
        """
        state_input = Input(shape=(self.state_dim,))
        action_input = Input(shape=(self.action_dim,))
        
        # Concatena stato e azione
        concat = layers.Concatenate()([state_input, action_input])
        
        x = layers.Dense(256, activation='relu')(concat)
        x = layers.Dense(256, activation='relu')(x)
        q_value = layers.Dense(1)(x)
        
        critic = Model([state_input, action_input], q_value, 
                      name=f'{self.agent_id}_critic')
        return critic
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Seleziona azione utilizzando la policy corrente.
        
        Args:
            state: Stato corrente
            training: Se aggiungere rumore di esplorazione
            
        Returns:
            Azione selezionata
        """
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        
        # Ottieni output della policy - restituisce una tupla di tensori
        policy_output = self.actor(state_tensor)
        mean = policy_output[0]
        log_std = policy_output[1]
        
        if training:
            # Campiona dalla distribuzione della policy
            std = tf.exp(log_std)
            normal = tf.random.normal(tf.shape(mean))
            action = tf.add(mean, tf.multiply(std, normal))
        else:
            # Usa policy deterministica
            action = mean
        
        # Limita azione al range [-1, 1] con casting appropriato
        action_clipped = tf.cast(tf.clip_by_value(action, -1, 1), tf.float32)
        return tf.squeeze(action_clipped).numpy()
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        """Memorizza transizione nel buffer di replay."""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self) -> Tuple[float, float]:
        """
        Esegue un passo di addestramento SAC.
        
        Returns:
            Loss dell'actor e loss del critic
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        # Campiona batch dal buffer di replay
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        
        # Assicura forme appropriate
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Assicura che ricompense e dones abbiano forma corretta
        if tf.rank(rewards) == 1:  # type: ignore
            rewards = tf.expand_dims(rewards, -1)
        if tf.rank(dones) == 1:  # type: ignore
            dones = tf.expand_dims(dones, -1)
        
        # Train critics
        try:
            critic_loss = self._train_critics(states, actions, rewards, next_states, dones)
        except Exception as e:
            print(f"Critic training error: {e}")
            critic_loss = 0.0
        
        # Train actor
        try:
            actor_loss = self._train_actor(states)
        except Exception as e:
            print(f"Actor training error: {e}")
            actor_loss = 0.0
        
        # Soft update target networks
        self._soft_update_targets()
        
        return float(actor_loss if actor_loss is not None else 0.0), float(critic_loss if critic_loss is not None else 0.0)
    
    @tf.function
    def _train_critics(self, states, actions, rewards, next_states, dones):
        """Addestra reti critic."""
        # Ottieni prossime azioni dalla policy corrente
        next_mean, next_log_std = self.actor(next_states)
        next_std = tf.exp(next_log_std)
        next_normal = tf.random.normal(tf.shape(next_mean))
        next_actions = next_mean + next_std * next_normal
        next_actions = tf.clip_by_value(next_actions, -1, 1)
        
        # Calcola bonus entropia
        next_log_prob = self._log_prob(next_actions, next_mean, next_log_std)
        
        # Q-values target
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
        
        # Aggiorna critics
        grad1 = tape1.gradient(critic1_loss, self.critic1.trainable_variables)
        grad2 = tape2.gradient(critic2_loss, self.critic2.trainable_variables)
        
        self.critic1_optimizer.apply_gradients(zip(grad1, self.critic1.trainable_variables))  # type: ignore
        self.critic2_optimizer.apply_gradients(zip(grad2, self.critic2.trainable_variables))  # type: ignore
        
        return (critic1_loss + critic2_loss) / 2
    
    @tf.function
    def _train_actor(self, states):
        """Addestra rete actor."""
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
            
            # Loss actor: massimizza Q - entropia
            actor_loss = tf.reduce_mean(self.alpha * log_prob - q)
        
        # Aggiorna actor
        grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grad, self.actor.trainable_variables))  # type: ignore
        
        return actor_loss
    
    def _log_prob(self, actions, mean, log_std):
        """Calcola probabilità logaritmica delle azioni."""
        std = tf.exp(log_std)
        # Calcola probabilità logaritmica della distribuzione Gaussiana
        log_prob = -0.5 * tf.reduce_sum(
            tf.square((actions - mean) / (std + 1e-8)) + 2 * log_std + tf.math.log(2 * np.pi),
            axis=-1, keepdims=True
        )
        return log_prob
    
    def _soft_update_targets(self):
        """Aggiorna soft le reti target."""
        for target, source in [(self.target_critic1, self.critic1),
                              (self.target_critic2, self.critic2)]:
            for target_param, source_param in zip(target.trainable_variables,
                                                 source.trainable_variables):
                target_param.assign(
                    self.tau * source_param + (1 - self.tau) * target_param
                )
    
    def save_agent(self, directory: str):
        """Salva reti dell'agente."""
        os.makedirs(directory, exist_ok=True)
        if self.actor is not None:
            self.actor.save(f"{directory}/{self.agent_id}_actor.h5")
        if self.critic1 is not None:
            self.critic1.save(f"{directory}/{self.agent_id}_critic1.h5")
        if self.critic2 is not None:
            self.critic2.save(f"{directory}/{self.agent_id}_critic2.h5")
    
    def load_agent(self, directory: str):
        """Carica reti dell'agente."""
        try:
            # Costruisce reti se non già costruite
            if self.actor is None:
                self.actor = self._build_actor()
            if self.critic1 is None:
                self.critic1 = self._build_critic()
            if self.critic2 is None:
                self.critic2 = self._build_critic()
            
            # Carica pesi invece di modelli completi
            self.actor.load_weights(f"{directory}/{self.agent_id}_actor.h5")
            self.critic1.load_weights(f"{directory}/{self.agent_id}_critic1.h5")
            self.critic2.load_weights(f"{directory}/{self.agent_id}_critic2.h5")
        except Exception as e:
            print(f"Error loading agent {self.agent_id}: {e}")


class CentralizedSAC:
    """
    SAC centralizzato per controllo multi-edificio.
    """
    
    def __init__(self, building_count: int = 3, obs_dim: int = 28, **kwargs):
        """
        Inizializza controllore SAC centralizzato.
        
        Args:
            building_count: Numero di edifici
            obs_dim: Dimensione osservazioni per edificio
            **kwargs: Argomenti per SACAgent
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
        """Addestra agente SAC centralizzato."""
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
            if isinstance(rewards, list):
                total_reward = sum(rewards)
            elif isinstance(rewards, np.ndarray):
                total_reward = float(np.sum(rewards))
            else:
                total_reward = float(rewards)
            
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
    SAC decentralizzato con agenti indipendenti per ogni edificio.
    
    Ogni edificio ha il proprio agente SAC che apprende in modo autonomo,
    permettendo specializzazione per le caratteristiche specifiche di ogni edificio.
    """
    
    def __init__(self, building_count: int = 3, obs_dim: int = 28, **kwargs):
        """
        Inizializza controllori SAC decentralizzati.
        
        Args:
            building_count: Numero di edifici
            obs_dim: Dimensione osservazioni per edificio
            **kwargs: Argomenti per SACAgent
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
        """Addestra tutti gli agenti SAC decentralizzati."""
        observations = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Ogni agente seleziona azione per il proprio edificio
            actions = []
            for i, agent in enumerate(self.agents):
                obs = observations[i] if isinstance(observations, list) else observations
                action = agent.select_action(obs, training=True)
                # Gestisce sia azioni scalari che array
                if np.isscalar(action):
                    actions.append(action)
                else:
                    actions.append(action[0])  # Single action per building
            
            # Environment step
            next_observations, rewards, done, info = env.step(actions)
            
            # Memorizza transizioni e addestra ogni agente
            total_reward = 0
            for i, agent in enumerate(self.agents):
                obs = observations[i] if isinstance(observations, list) else observations
                next_obs = next_observations[i] if isinstance(next_observations, list) else next_observations
                reward = rewards[i] if isinstance(rewards, list) else rewards / len(self.agents)
                
                agent.store_transition(obs, np.array([actions[i]]), reward, next_obs, done)
                
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
        """Salva tutti gli agenti decentralizzati."""
        for i, agent in enumerate(self.agents):
            agent.save_agent(f"{directory}/building_{i}")
    
    def load_agents(self, directory: str):
        """Carica tutti gli agenti decentralizzati."""
        for i, agent in enumerate(self.agents):
            agent.load_agent(f"{directory}/building_{i}")