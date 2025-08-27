# Smart Building Energy Management with Neural Networks and Reinforcement Learning

A comprehensive implementation of advanced machine learning approaches for building energy forecasting and optimization, featuring LSTM variants, Transformers, and RL agents with modular reward functions.

## 🏢 Project Overview

This project implements state-of-the-art neural network architectures and reinforcement learning algorithms for smart building energy management, focusing on:

- **Cross-building generalization** for energy forecasting
- **Multi-objective optimization** with configurable reward functions
- **Comprehensive model comparison** across different architectures
- **Real-world dataset** from CityLearn Challenge 2023

## 🎯 Key Features

### Neural Network Models
- **LSTM Variants**: Standard, Bidirectional, Convolutional LSTM
- **Transformer Models**: Self-attention mechanisms for time series
- **Baseline Models**: Linear Regression, Random Forest, Polynomial Regression
- **Cross-building evaluation**: Train on one building, test on others

### Reinforcement Learning
- **Q-Learning**: Centralized and decentralized approaches
- **SAC (Soft Actor-Critic)**: Advanced policy gradient methods
- **Modular Reward Functions**: 6 different optimization strategies
- **Multi-objective optimization**: Balance efficiency, comfort, cost, sustainability

### Reward Function Types
- `efficiency`: Pure energy minimization
- `comfort`: Occupant comfort optimization
- `balanced`: Multi-objective balance (default)
- `cost`: Monetary cost optimization
- `sustainability`: Environmental impact focus
- `multi_objective`: Configurable weights

## 📊 Dataset

**CityLearn Challenge 2023** - Complete dataset including:
- **5 phases** of data (Phase 1 + 4 Phase 2 evaluations)
- **3 buildings** with 122 days each (2928 hourly samples)
- **Multiple features**: Temperature, humidity, occupancy, solar generation
- **Target**: Cooling demand prediction and optimization

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/smart-building-energy-ml
cd smart-building-energy-ml

# Install dependencies
pip install -r requirements.txt
```

### Running Neural Network Evaluation

```bash
python run_neural_evaluation.py
```

**Output:**
- Results: `results/neural_networks/results.json`
- Visualizations: `results/visualizations/01-04*.png`

### Running RL Evaluation

```bash
# Default balanced reward
python run_rl_evaluation.py

# Different reward functions
python -c "from run_rl_evaluation import main; main('efficiency')"
python -c "from run_rl_evaluation import main; main('comfort')"
python -c "from run_rl_evaluation import main; main('cost')"
```

**Output:**
- Results: `results/rl_experiments/rl_results.json` 
- Visualizations: `results/visualizations/05*.png`

## 📈 Results

### Neural Network Performance
- **Best cross-building**: Building_3→Building_2 (RMSE=0.520, R²=0.809)
- **Competitive baselines**: Linear Regression rivals complex deep models
- **Cross-building transfer**: Successful generalization across different buildings

### RL Performance
- **SAC vs Q-Learning**: Policy gradient methods show superior convergence
- **Centralized vs Decentralized**: Trade-offs between coordination and scalability
- **Reward function impact**: Different strategies lead to distinct optimization behaviors

## 🛠️ Project Structure

```
├── src/
│   ├── forecasting/          # Neural network models
│   │   ├── lstm_models.py     # LSTM variants
│   │   ├── transformer_models.py # Transformer architectures
│   │   └── base_models.py     # Baseline models
│   ├── rl/                   # Reinforcement learning
│   │   ├── q_learning_agent.py # Q-Learning implementations
│   │   ├── sac_agent.py      # SAC agent
│   │   └── reward_functions.py # Modular reward system
│   └── utils/                # Utilities
│       ├── data_utils.py     # Data processing
│       └── visualization.py  # Plotting functions
├── data/                     # CityLearn 2023 dataset
├── results/                  # Results and visualizations
├── run_neural_evaluation.py  # Neural network evaluation
├── run_rl_evaluation.py     # RL evaluation
└── requirements.txt          # Dependencies
```

## 🎨 Visualizations

The project automatically generates comprehensive visualizations:

1. **Algorithm Comparison Table** - Performance metrics across all models
2. **Cross-Building Generalization** - Heatmap of transfer learning performance  
3. **Training Convergence** - Learning curves for neural networks
4. **Model Architectures** - Visual representation of network structures
5. **RL Training Analysis** - Agent learning progress and comparison

## ⚙️ Configuration

### Neural Networks
- **Epochs**: 50 (configurable)
- **Sequence length**: 24 hours
- **Features**: 7 (temperature, humidity, occupancy, etc.)
- **Cross-validation**: 80/20 split with early stopping

### Reinforcement Learning  
- **Episodes**: Q-Learning (80), SAC (40)
- **Environment**: MockCityLearnEnv with realistic building dynamics
- **Reward functions**: Modular system supporting custom objectives

## 📚 Dependencies

- `tensorflow>=2.16.0`
- `keras>=3.0.0`
- `pandas>=1.5.0`
- `numpy>=1.21.0`
- `matplotlib>=3.5.0`
- `seaborn>=0.11.0`
- `scikit-learn>=1.1.0`

## 🔬 Key Insights

- **Linear models remain competitive** for cross-building energy forecasting
- **Transformer models** require careful tuning for time series applications
- **Multi-objective RL** enables flexible optimization strategies
- **Cross-building generalization** is challenging but achievable with proper architectures
