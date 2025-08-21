# CityLearn Challenge 2023: Reinforcement Learning and Time Series Forecasting

This project implements a comprehensive solution for the **CityLearn Challenge 2023**, covering both the Control Track and Forecasting Track with state-of-the-art machine learning approaches.

##  Project Overview

This implementation addresses the multi-faceted nature of advanced building control by providing:

1. **Reinforcement Learning (Control Track)**: Q-learning and SAC algorithms for optimal building energy management
2. **Time Series Forecasting (Forecasting Track)**: Multiple ML models for 48-hour ahead energy and carbon intensity prediction
3. **CityLearn Challenge 2023 Compliance**: Official competition targets, metrics, and evaluation procedures

## Project Structure

```
tesi/
├── data/                   # CityLearn Challenge 2023 datasets
├── src/                    # Source code
│   ├── rl/                 # Reinforcement learning implementations
│   │   ├── q_learning.py   # Q-learning (centralized/decentralized)
│   │   └── sac.py          # Soft Actor-Critic (centralized/decentralized)
│   ├── forecasting/        # Time series forecasting models
│   │   ├── base_models.py  # LSTM, ANN, Gaussian, Linear, RF, Polynomial
│   │   ├── transformer_models.py # Transformer architectures
│   │   └── citylearn_challenge.py # CityLearn Challenge specific
│   └── utils/              # Utility functions
│       ├── data_utils.py   # Data loading, preprocessing
│       └── visualization.py # Results visualization
├── notebooks/              # Jupyter notebooks
│   ├── tutorial.ipynb      # CityLearn tutorial walkthrough
│   └── results_analysis.ipynb # Results demonstration
├── results/                # Experimental outputs
│   ├── models/             # Trained models
│   ├── tables/             # Results tables
│   └── plots/              # Visualizations
├── run_experiments.py      # Unified experiment runner (adaptive)
└── requirements.txt        # Python dependencies
```

## Setup

1. Create virtual environment:
```bash
python -m venv citylearn_env
citylearn_env\Scripts\activate  # Windows
# or: source citylearn_env/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

##  CityLearn Challenge 2023 Compliance

### Forecasting Track Targets
**Building-Level (per building, 48h ahead):**
- `cooling_demand` (kWh)
- `dhw_demand` (kWh) 
- `non_shiftable_load` (kWh) - Equipment Electric Power

**Neighborhood-Level (aggregated, 48h ahead):**
- `carbon_intensity` (kgCO2e/kWh)
- `solar_generation` (W/kW)

### Evaluation Metrics
- **Primary**: Normalized Root Mean Square Error (RMSE)
- **Secondary**: MAE, R², MAPE for detailed analysis

##  Quick Start

### 1. Run CityLearn Challenge Experiments
```bash
# Run unified experiments (automatically adapts to available libraries)
python run_experiments.py
```

### 2. Explore with Jupyter
```bash
jupyter lab notebooks/tutorial.ipynb
```

##  Implementation Details

### Reinforcement Learning (Control Track)
- **Q-learning**: Discrete action space with centralized/decentralized approaches
- **SAC**: Continuous control with entropy regularization  
- **Multi-building coordination**: Both centralized and decentralized strategies

### Time Series Forecasting (Forecasting Track)
- **Deep Learning**: LSTM, ANN with 48-hour prediction horizon
- **Traditional ML**: Gaussian Process, Random Forest, Linear/Polynomial regression
- **Challenge-specific**: Normalized RMSE evaluation as per competition requirements

### Advanced Analysis
- **Cross-building generalization**: Train on one building, test on others
- **Neighborhood aggregation**: Portfolio-level forecasting with error compensation
- **Ablation studies**: Model comparison across different building types and targets

##  Results and Evaluation

The project generates comprehensive results including:
- Competition-compliant normalized RMSE scores
- Cross-building generalization performance
- Neighborhood-level aggregation benefits
- Model comparison tables and visualizations
- Training curves and prediction plots


##  Dataset Structure

Using the official **CityLearn Challenge 2023** dataset:
- **3 buildings** (Building_1, Building_2, Building_3)
- **720 timesteps** per phase (1-month simulation)
- **Multiple phases** (Phase 1, Phase 2 with variations)
- **Realistic building dynamics** with LSTM-based thermal models