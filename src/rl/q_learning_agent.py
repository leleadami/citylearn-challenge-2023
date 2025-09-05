"""
Agente Q-Learning per CityLearn Challenge 2023

Implementa approcci Q-learning sia centralizzati che decentralizzati
per gestione energetica degli edifici e demand response.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import json
import os


class QLearningAgent:
    """
    Agente Q-Learning per controllo energetico degli edifici.
    
    Supporta architetture sia centralizzate (singolo agente per tutti gli edifici) che
    decentralizzate (agenti indipendenti per edificio).
    """
    
    def __init__(self, 
                 agent_id: str,
                 state_bins: int = 10,
                 action_bins: int = 5,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995):
        """
        Inizializza agente Q-Learning.
        
        Args:
            agent_id: Identificatore unico per l'agente
            state_bins: Numero di bin per discretizzazione dello stato
            action_bins: Numero di azioni discrete
            learning_rate: Learning rate alpha
            discount_factor: Fattore di sconto ricompense future gamma
            epsilon: Probabilità di esplorazione
            epsilon_decay: Tasso di decadimento epsilon
        """
        self.agent_id = agent_id
        self.state_bins = state_bins
        self.action_bins = action_bins
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        # Q-table: state -> action -> Q-value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Metriche di addestramento
        self.training_rewards = []
        self.training_losses = []
        self.episode_count = 0
    
    def discretize_state(self, state: np.ndarray) -> Tuple:
        """
        Discretizza spazio di stato continuo in bin.
        
        Args:
            state: Vettore di stato continuo
            
        Returns:
            Tupla di stato discretizzato
        """
        # Normalizza stato al range [0, 1]
        normalized_state = np.clip(state, -3, 3) / 6 + 0.5
        
        # Discretizza in bin
        discrete_state = tuple(
            int(s * (self.state_bins - 1)) for s in normalized_state
        )
        
        return discrete_state
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Seleziona azione utilizzando policy epsilon-greedy.
        
        Args:
            state: Stato corrente
            training: Se in modalità di addestramento
            
        Returns:
            Indice dell'azione selezionata
        """
        discrete_state = self.discretize_state(state)
        
        # Esplorazione epsilon-greedy
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_bins)
        
        # Selezione azione greedy
        q_values = [
            self.q_table[discrete_state][action] 
            for action in range(self.action_bins)
        ]
        
        return int(np.argmax(q_values))
    
    def update_q_table(self, state: np.ndarray, action: int, 
                       reward: float, next_state: np.ndarray) -> float:
        """
        Aggiorna Q-table utilizzando regola di aggiornamento Q-learning.
        
        Args:
            state: Stato corrente
            action: Azione presa
            reward: Ricompensa ricevuta
            next_state: Prossimo stato
            
        Returns:
            Errore TD (per monitoraggio)
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Q-value corrente
        current_q = self.q_table[discrete_state][action]
        
        # Miglior Q-value della prossima azione
        next_q_values = [
            self.q_table[discrete_next_state][a] 
            for a in range(self.action_bins)
        ]
        max_next_q = max(next_q_values) if next_q_values else 0.0
        
        # Aggiornamento Q-learning: Q(s,a) = Q(s,a) + α[r + γ*max_Q(s',a') - Q(s,a)]
        target_q = reward + self.discount_factor * max_next_q
        td_error = target_q - current_q
        
        self.q_table[discrete_state][action] += self.learning_rate * td_error
        
        return abs(td_error)
    
    def decay_epsilon(self):
        """Diminuisce probabilità di esplorazione."""
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
    
    def save_agent(self, filepath: str):
        """Salva Q-table e parametri dell'agente."""
        agent_data = {
            'agent_id': self.agent_id,
            'q_table': {
                str(state): dict(actions) 
                for state, actions in self.q_table.items()
            },
            'parameters': {
                'state_bins': self.state_bins,
                'action_bins': self.action_bins,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon
            },
            'training_metrics': {
                'rewards': self.training_rewards,
                'losses': self.training_losses,
                'episodes': self.episode_count
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(agent_data, f, indent=2)
    
    def load_agent(self, filepath: str):
        """Carica Q-table e parametri dell'agente."""
        with open(filepath, 'r') as f:
            agent_data = json.load(f)
        
        # Restore Q-table
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state_str, actions in agent_data['q_table'].items():
            state = eval(state_str)  # Convert string back to tuple
            for action, q_value in actions.items():
                self.q_table[state][int(action)] = q_value
        
        # Restore parameters
        params = agent_data['parameters']
        self.epsilon = params['epsilon']
        
        # Restore training metrics
        metrics = agent_data['training_metrics']
        self.training_rewards = metrics['rewards']
        self.training_losses = metrics['losses']
        self.episode_count = metrics['episodes']


class CentralizedQLearning:
    """
    Centralized Q-Learning approach for multi-building control.
    
    Single agent controls all buildings simultaneously.
    """
    
    def __init__(self, building_count: int = 3, **kwargs):
        """
        Initialize centralized Q-learning controller.
        
        Args:
            building_count: Number of buildings to control
            **kwargs: Arguments passed to QLearningAgent
        """
        self.building_count = building_count
        self.agent = QLearningAgent("centralized_controller", **kwargs)
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_values': []
        }
    
    def train_episode(self, env, max_steps: int = 1000):
        """
        Train centralized agent for one episode.
        
        Args:
            env: CityLearn environment
            max_steps: Maximum steps per episode
            
        Returns:
            Episode reward and length
        """
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Flatten multi-building state for centralized agent
            if isinstance(state, list):
                flat_state = np.concatenate(state)
            else:
                flat_state = state.flatten()
            
            # Select actions for all buildings
            action_idx = self.agent.select_action(flat_state, training=True)
            
            # Convert action index to multi-building actions
            actions = self.decode_action(action_idx)
            
            # Esegue step nell'ambiente
            next_state, reward, done, info = env.step(actions)
            
            # Flatten next state
            if isinstance(next_state, list):
                flat_next_state = np.concatenate(next_state)
            else:
                flat_next_state = next_state.flatten()
            
            # Update Q-table
            total_reward = np.sum(reward) if isinstance(reward, list) else reward
            td_error = self.agent.update_q_table(
                flat_state, action_idx, total_reward, flat_next_state
            )
            
            episode_reward += total_reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # Decay exploration
        self.agent.decay_epsilon()
        self.agent.episode_count += 1
        
        # Record training metrics
        self.training_history['episode_rewards'].append(episode_reward)
        self.training_history['episode_lengths'].append(episode_length)
        self.training_history['epsilon_values'].append(self.agent.epsilon)
        
        return episode_reward, episode_length
    
    def decode_action(self, action_idx: int) -> List[float]:
        """
        Convert single action index to multi-building actions.
        
        Args:
            action_idx: Centralized action index
            
        Returns:
            List of actions for each building
        """
        # Simple decoding: same action for all buildings
        # In practice, this could be more sophisticated
        action_value = (action_idx / (self.agent.action_bins - 1)) * 2 - 1  # [-1, 1]
        return [action_value] * self.building_count


class DecentralizedQLearning:
    """
    Decentralized Q-Learning approach for multi-building control.
    
    Independent agents for each building.
    """
    
    def __init__(self, building_count: int = 3, **kwargs):
        """
        Initialize decentralized Q-learning controllers.
        
        Args:
            building_count: Number of buildings
            **kwargs: Arguments passed to each QLearningAgent
        """
        self.building_count = building_count
        self.agents = [
            QLearningAgent(f"building_{i}_agent", **kwargs) 
            for i in range(building_count)
        ]
        
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_values': [[] for _ in range(building_count)]
        }
    
    def train_episode(self, env, max_steps: int = 1000):
        """
        Train all decentralized agents for one episode.
        
        Args:
            env: CityLearn environment
            max_steps: Maximum steps per episode
            
        Returns:
            Episode reward and length
        """
        states = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Each agent selects action for its building
            actions = []
            for i, agent in enumerate(self.agents):
                building_state = states[i] if isinstance(states, list) else states
                action_idx = agent.select_action(building_state, training=True)
                action_value = (action_idx / (agent.action_bins - 1)) * 2 - 1  # [-1, 1]
                actions.append(action_value)
            
            # Esegue step nell'ambiente
            next_states, rewards, done, info = env.step(actions)
            
            # Update each agent's Q-table
            total_reward = 0
            for i, agent in enumerate(self.agents):
                building_state = states[i] if isinstance(states, list) else states
                next_building_state = next_states[i] if isinstance(next_states, list) else next_states
                building_reward = rewards[i] if isinstance(rewards, list) else rewards / len(self.agents)
                
                action_idx = agent.select_action(building_state, training=False)  # Get last action
                agent.update_q_table(
                    building_state, action_idx, building_reward, next_building_state
                )
                
                total_reward += building_reward
            
            episode_reward += total_reward
            episode_length += 1
            states = next_states
            
            if done:
                break
        
        # Decay exploration for all agents
        for i, agent in enumerate(self.agents):
            agent.decay_epsilon()
            agent.episode_count += 1
            self.training_history['epsilon_values'][i].append(agent.epsilon)
        
        # Record training metrics
        self.training_history['episode_rewards'].append(episode_reward)
        self.training_history['episode_lengths'].append(episode_length)
        
        return episode_reward, episode_length
    
    def save_agents(self, directory: str):
        """Save all decentralized agents."""
        os.makedirs(directory, exist_ok=True)
        for i, agent in enumerate(self.agents):
            agent.save_agent(f"{directory}/building_{i}_agent.json")
    
    def load_agents(self, directory: str):
        """Load all decentralized agents."""
        for i, agent in enumerate(self.agents):
            agent.load_agent(f"{directory}/building_{i}_agent.json")