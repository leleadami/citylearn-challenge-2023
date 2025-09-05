"""
Funzioni Ricompensa per Gestione Energetica Intelligente degli Edifici

Questo modulo implementa varie funzioni ricompensa per agenti di reinforcement learning
nell'ottimizzazione energetica degli edifici. Diverse funzioni ricompensa incoraggiano
comportamenti diversi e obiettivi di ottimizzazione differenti.
"""

import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


class BaseRewardFunction(ABC):
    """Base class for all reward functions."""
    
    def __init__(self, name: str):
        """Initialize reward function with a name."""
        self.name = name
    
    @abstractmethod
    def calculate_reward(self, state: Dict, action: float, next_state: Dict, **kwargs) -> float:
        """
        Calculate reward based on state transition.
        
        Args:
            state: Current building state
            action: Action taken
            next_state: Next building state after action
            **kwargs: Additional parameters
            
        Returns:
            Reward value
        """
        pass
    
    def __str__(self):
        return f"RewardFunction({self.name})"


class EfficiencyRewardFunction(BaseRewardFunction):
    """
    Reward function focused purely on energy efficiency.
    Minimizes total energy consumption without considering comfort.
    """
    
    def __init__(self, max_energy: float = 50.0):
        """
        Initialize efficiency-focused reward.
        
        Args:
            max_energy: Maximum expected energy consumption for normalization
        """
        super().__init__("Energy_Efficiency")
        self.max_energy = max_energy
    
    def calculate_reward(self, state: Dict, action: float, next_state: Dict, **kwargs) -> float:
        """Calculate reward based on energy efficiency only."""
        # Total energy consumption
        cooling_demand = next_state.get('cooling_demand', 0)
        heating_demand = next_state.get('heating_demand', 0)
        non_shiftable_load = next_state.get('non_shiftable_load', 0)
        
        total_energy = cooling_demand + heating_demand + non_shiftable_load
        
        # Normalize and invert (lower consumption = higher reward)
        normalized_energy = min(total_energy / self.max_energy, 1.0)
        efficiency_reward = 1.0 - normalized_energy
        
        # Small penalty for extreme actions
        action_penalty = -0.01 * (action ** 2)
        
        return efficiency_reward + action_penalty


class ComfortRewardFunction(BaseRewardFunction):
    """
    Reward function focused on occupant comfort.
    Prioritizes maintaining optimal temperature regardless of energy cost.
    """
    
    def __init__(self, target_temp: float = 22.0, temp_tolerance: float = 2.0):
        """
        Initialize comfort-focused reward.
        
        Args:
            target_temp: Target temperature (°C)
            temp_tolerance: Acceptable temperature deviation (°C)
        """
        super().__init__("Occupant_Comfort")
        self.target_temp = target_temp
        self.temp_tolerance = temp_tolerance
    
    def calculate_reward(self, state: Dict, action: float, next_state: Dict, **kwargs) -> float:
        """Calculate reward based on occupant comfort only."""
        temperature = next_state.get('indoor_dry_bulb_temperature', self.target_temp)
        
        # Temperature deviation from target
        temp_deviation = abs(temperature - self.target_temp)
        
        # Exponential penalty for temperature outside comfort zone
        if temp_deviation <= self.temp_tolerance:
            comfort_reward = 1.0 - (temp_deviation / self.temp_tolerance) ** 2
        else:
            # Heavy penalty for temperatures outside acceptable range
            comfort_reward = -0.5 * (temp_deviation - self.temp_tolerance)
        
        return comfort_reward


class BalancedRewardFunction(BaseRewardFunction):
    """
    Balanced reward function combining efficiency, comfort, and stability.
    This is the default reward function for most applications.
    """
    
    def __init__(self, 
                 efficiency_weight: float = 0.6,
                 comfort_weight: float = 0.3,
                 stability_weight: float = 0.1,
                 target_temp: float = 22.0,
                 max_energy: float = 50.0):
        """
        Initialize balanced reward function.
        
        Args:
            efficiency_weight: Weight for energy efficiency component
            comfort_weight: Weight for occupant comfort component  
            stability_weight: Weight for action stability component
            target_temp: Target temperature (°C)
            max_energy: Maximum expected energy consumption
        """
        super().__init__("Balanced_Energy_Comfort")
        self.efficiency_weight = efficiency_weight
        self.comfort_weight = comfort_weight  
        self.stability_weight = stability_weight
        self.target_temp = target_temp
        self.max_energy = max_energy
        
        # Normalize weights
        total_weight = efficiency_weight + comfort_weight + stability_weight
        self.efficiency_weight /= total_weight
        self.comfort_weight /= total_weight
        self.stability_weight /= total_weight
    
    def calculate_reward(self, state: Dict, action: float, next_state: Dict, **kwargs) -> float:
        """Calculate balanced reward combining multiple objectives."""
        # 1. Energy Efficiency Component
        cooling_demand = next_state.get('cooling_demand', 0)
        heating_demand = next_state.get('heating_demand', 0) 
        non_shiftable_load = next_state.get('non_shiftable_load', 0)
        total_energy = cooling_demand + heating_demand + non_shiftable_load
        
        normalized_energy = min(total_energy / self.max_energy, 1.0)
        efficiency_reward = 1.0 - normalized_energy
        
        # 2. Comfort Component
        temperature = next_state.get('indoor_dry_bulb_temperature', self.target_temp)
        temp_deviation = abs(temperature - self.target_temp) / 10.0  # Normalize
        comfort_reward = max(0, 1.0 - temp_deviation)
        
        # 3. Action Stability Component (discourage extreme actions)
        stability_reward = -0.1 * (action ** 2)  # Quadratic penalty
        
        # Combine components
        total_reward = (self.efficiency_weight * efficiency_reward + 
                       self.comfort_weight * comfort_reward + 
                       self.stability_weight * stability_reward)
        
        return total_reward


class CostOptimizedRewardFunction(BaseRewardFunction):
    """
    Reward function optimized for monetary cost minimization.
    Takes into account dynamic electricity pricing and demand charges.
    """
    
    def __init__(self, base_price: float = 0.15, demand_charge: float = 10.0):
        """
        Initialize cost-optimized reward.
        
        Args:
            base_price: Base electricity price ($/kWh)
            demand_charge: Demand charge for peak usage ($/kW)
        """
        super().__init__("Cost_Optimization")
        self.base_price = base_price
        self.demand_charge = demand_charge
    
    def calculate_reward(self, state: Dict, action: float, next_state: Dict, **kwargs) -> float:
        """Calculate reward based on monetary cost minimization."""
        # Energy consumption
        cooling_demand = next_state.get('cooling_demand', 0)
        heating_demand = next_state.get('heating_demand', 0)
        non_shiftable_load = next_state.get('non_shiftable_load', 0)
        total_energy = cooling_demand + heating_demand + non_shiftable_load
        
        # Time-of-use pricing (peak hours more expensive)
        hour = next_state.get('hour', 12)
        if 16 <= hour <= 20:  # Peak hours
            energy_price = self.base_price * 2.0
        elif 22 <= hour <= 6:  # Off-peak hours
            energy_price = self.base_price * 0.7  
        else:  # Standard hours
            energy_price = self.base_price
        
        # Calculate costs
        energy_cost = total_energy * energy_price
        demand_cost = max(0, total_energy - 5.0) * self.demand_charge  # Demand charge over 5kW
        total_cost = energy_cost + demand_cost
        
        # Reward is negative cost (minimize cost = maximize reward)
        cost_reward = -total_cost / 20.0  # Scale to reasonable range
        
        # Small comfort component to avoid extreme discomfort
        temperature = next_state.get('indoor_dry_bulb_temperature', 22.0)
        comfort_penalty = -0.1 * max(0, abs(temperature - 22.0) - 3.0)  # Penalty only if >3°C off
        
        return cost_reward + comfort_penalty


class SustainabilityRewardFunction(BaseRewardFunction):
    """
    Reward function focused on environmental sustainability.
    Minimizes carbon emissions and promotes renewable energy usage.
    """
    
    def __init__(self, carbon_intensity: float = 0.5):
        """
        Initialize sustainability-focused reward.
        
        Args:
            carbon_intensity: kg CO2 per kWh from grid
        """
        super().__init__("Environmental_Sustainability")
        self.carbon_intensity = carbon_intensity
    
    def calculate_reward(self, state: Dict, action: float, next_state: Dict, **kwargs) -> float:
        """Calculate reward based on environmental impact."""
        # Energy consumption
        cooling_demand = next_state.get('cooling_demand', 0)
        heating_demand = next_state.get('heating_demand', 0)
        non_shiftable_load = next_state.get('non_shiftable_load', 0)
        total_energy = cooling_demand + heating_demand + non_shiftable_load
        
        # Solar generation (reduces grid dependency)
        solar_generation = next_state.get('solar_generation', 0)
        net_energy = max(0, total_energy - solar_generation)
        
        # Carbon emissions
        carbon_emissions = net_energy * self.carbon_intensity
        
        # Reward components
        carbon_reward = -carbon_emissions / 10.0  # Scale emissions penalty
        renewable_bonus = min(solar_generation / 5.0, 1.0)  # Bonus for using solar
        
        # Basic comfort constraint
        temperature = next_state.get('indoor_dry_bulb_temperature', 22.0)
        comfort_penalty = -0.2 * max(0, abs(temperature - 22.0) - 4.0)  # Allow wider range
        
        return carbon_reward + renewable_bonus + comfort_penalty


class MultiObjectiveRewardFunction(BaseRewardFunction):
    """
    Advanced multi-objective reward function with configurable weights.
    Allows dynamic balancing of different objectives during training.
    """
    
    def __init__(self, 
                 efficiency_weight: float = 0.4,
                 comfort_weight: float = 0.3, 
                 cost_weight: float = 0.2,
                 sustainability_weight: float = 0.1):
        """
        Initialize multi-objective reward.
        
        Args:
            efficiency_weight: Weight for energy efficiency
            comfort_weight: Weight for occupant comfort
            cost_weight: Weight for monetary cost
            sustainability_weight: Weight for environmental impact
        """
        super().__init__("Multi_Objective")
        
        # Normalize weights
        total = efficiency_weight + comfort_weight + cost_weight + sustainability_weight
        self.weights = {
            'efficiency': efficiency_weight / total,
            'comfort': comfort_weight / total,
            'cost': cost_weight / total,
            'sustainability': sustainability_weight / total
        }
        
        # Initialize component reward functions
        self.efficiency_rf = EfficiencyRewardFunction()
        self.comfort_rf = ComfortRewardFunction()
        self.cost_rf = CostOptimizedRewardFunction()
        self.sustainability_rf = SustainabilityRewardFunction()
    
    def calculate_reward(self, state: Dict, action: float, next_state: Dict, **kwargs) -> float:
        """Calculate multi-objective reward."""
        rewards = {
            'efficiency': self.efficiency_rf.calculate_reward(state, action, next_state),
            'comfort': self.comfort_rf.calculate_reward(state, action, next_state),
            'cost': self.cost_rf.calculate_reward(state, action, next_state),
            'sustainability': self.sustainability_rf.calculate_reward(state, action, next_state)
        }
        
        # Weighted combination
        total_reward = sum(self.weights[obj] * reward for obj, reward in rewards.items())
        
        return total_reward


# Factory function for easy reward function creation
def create_reward_function(reward_type: str, **kwargs) -> BaseRewardFunction:
    """
    Factory function to create reward functions by name.
    
    Args:
        reward_type: Type of reward function
        **kwargs: Parameters for the reward function
        
    Returns:
        Reward function instance
        
    Available types:
        - 'efficiency': Pure energy efficiency optimization
        - 'comfort': Pure comfort optimization
        - 'balanced': Balanced efficiency-comfort (default)
        - 'cost': Monetary cost optimization
        - 'sustainability': Environmental optimization
        - 'multi_objective': Configurable multi-objective
    """
    reward_functions = {
        'efficiency': EfficiencyRewardFunction,
        'comfort': ComfortRewardFunction,
        'balanced': BalancedRewardFunction,
        'cost': CostOptimizedRewardFunction,
        'sustainability': SustainabilityRewardFunction,
        'multi_objective': MultiObjectiveRewardFunction
    }
    
    if reward_type not in reward_functions:
        raise ValueError(f"Unknown reward type: {reward_type}. Available: {list(reward_functions.keys())}")
    
    return reward_functions[reward_type](**kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test different reward functions
    test_state = {'hour': 14, 'indoor_dry_bulb_temperature': 23.0}
    test_action = 0.5
    test_next_state = {
        'cooling_demand': 5.0,
        'heating_demand': 0.0,
        'non_shiftable_load': 3.0,
        'indoor_dry_bulb_temperature': 22.5,
        'solar_generation': 2.0,
        'hour': 15
    }
    
    # Test each reward function
    reward_types = ['efficiency', 'comfort', 'balanced', 'cost', 'sustainability', 'multi_objective']
    
    print("REWARD FUNCTION COMPARISON")
    print("=" * 50)
    
    for reward_type in reward_types:
        rf = create_reward_function(reward_type)
        reward = rf.calculate_reward(test_state, test_action, test_next_state)
        print(f"{rf.name:25}: {reward:+6.3f}")
    
    print("\nAll reward functions tested successfully!")