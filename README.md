# CityLearn Challenge 2023 - Neural Network Forecasting

Scientific neural network implementation for building energy forecasting using LSTM and Transformer architectures with cross-building generalization.

## Project Overview

This project implements deep learning approaches for building energy consumption prediction using the CityLearn Challenge 2023 dataset. The focus is on neural networks with cross-building generalization and real training with 200+ epochs.

## Key Features

- **Neural Networks**: LSTM and Transformer architectures for time series forecasting
- **Complete Dataset**: 3-month dataset (June-August, 92 days) with 2208 samples per building
- **Cross-Building Evaluation**: Train on one building, test on others for generalization
- **Scientific Training**: Real 200+ epoch training with early stopping and learning rate scheduling
- **Professional Visualizations**: Training curves and performance analysis

## Project Structure

```
├── src/
│   ├── forecasting/          # Neural network models (LSTM, Transformer)
│   ├── utils/                # Data processing utilities
│   └── ...
├── data/                     # CityLearn 2023 datasets (3 phases, 3 months)
├── results/
│   ├── neural_networks/      # Neural network results
│   └── visualizations/       # Training curves and analysis
├── notebooks/                # Jupyter analysis
└── run_neural_evaluation.py  # Main neural network script
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Neural Network Evaluation**:
   ```bash
   python run_neural_evaluation.py
   ```

3. **View Results**:
   - Neural Results: `results/neural_networks/results.json`
   - Visualizations: `results/visualizations/`

## Neural Network Architectures

### LSTM (Long Short-Term Memory)
- **Architecture**: 3-layer LSTM with 64 hidden units per layer
- **Features**: Dropout regularization, Adam optimizer
- **Training**: 200+ epochs with early stopping
- **Sequence Length**: 24 hours (daily patterns)

### Transformer
- **Architecture**: 4-layer encoder with multi-head attention
- **Features**: 8 attention heads, positional encoding
- **Training**: Advanced learning rate scheduling
- **Benefits**: Long-range dependencies, parallel processing

## Cross-Building Generalization

The project evaluates neural networks on cross-building scenarios:
- Train on Building 1 → Test on Buildings 2, 3
- Train on Building 2 → Test on Buildings 1, 3  
- Train on Building 3 → Test on Buildings 1, 2

This tests the ability of models to generalize energy patterns across different building characteristics.

## Visualizations

The project generates 5 professional visualizations:

1. **Algorithm Performance Analysis** - 6-panel comprehensive comparison
2. **Cross-Building Matrix** - Heatmaps for each algorithm
3. **Scientific Publication Analysis** - 9-panel publication-ready plots
4. **Final Results Dashboard** - Executive summary
5. **Cross-Building Performance** - Overview analysis

## Academic Requirements

This project fulfills academic requirements for:
- Cross-building generalization evaluation
- Multiple algorithm comparison with statistical significance
- Professional visualization standards
- Comprehensive experimental design (3×4×3 = 36 experiments)

## Technical Details

- **Framework**: CityLearn 1.8.1, TensorFlow 2.13.0, PyTorch
- **Evaluation**: RMSE and R² metrics with cross-validation
- **Visualization**: Matplotlib/Seaborn with 300 DPI publication quality
- **Data**: CityLearn Challenge 2023 official datasets (5 phases)

## License

Academic research project for CityLearn Challenge 2023 participation.