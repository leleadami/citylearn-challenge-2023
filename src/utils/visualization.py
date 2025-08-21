"""
Visualization utilities for CityLearn project results.

This module provides comprehensive visualization tools for analyzing building
energy forecasting models and results in the CityLearn Challenge context.
It supports visualization of training progress, forecasting performance,
cross-building generalization, and neighborhood-level analysis.

Key visualization categories:
1. Training dynamics: Learning curves, loss evolution, convergence analysis
2. Forecasting quality: Time series comparisons, residual analysis, error distributions
3. Model comparison: Performance metrics across algorithms and buildings
4. Cross-building analysis: Generalization patterns and transfer learning insights
5. Neighborhood analysis: District-level energy patterns and aggregation benefits

Design principles:
- Clear, publication-ready plots with consistent styling
- Comprehensive analysis with multiple plot types per function
- Energy domain-specific interpretations and insights
- Statistical rigor in model comparison and significance testing
- Interactive insights for energy system optimization

Visualization best practices for energy data:
- Time series plots show temporal patterns and seasonality
- Residual analysis reveals model bias and heteroscedasticity
- Scatter plots indicate prediction accuracy and outliers
- Heatmaps highlight cross-building performance patterns
- Statistical tests validate performance differences
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os


def setup_plotting_style():
    """
    Setup consistent plotting style for professional energy forecasting visualizations.
    
    This function configures matplotlib and seaborn with settings optimized for
    energy data visualization and publication-quality figures. The style choices
    are designed to enhance readability and provide clear insights into building
    energy patterns.
    
    Style design rationale:
    - Seaborn style: Clean, modern appearance with subtle grid lines
    - Color palette: 'husl' provides distinct colors for multiple buildings/models
    - Figure size: 12x8 balances detail with screen/paper real estate
    - Font sizes: Hierarchical sizing for titles, labels, and text
    - Grid lines: Light alpha for reference without visual clutter
    
    Energy data visualization considerations:
    - Time series require clear temporal progression
    - Multiple buildings need distinguishable colors
    - Seasonal patterns benefit from longer time windows
    - Forecasting accuracy needs residual analysis
    - Cross-building comparisons require consistent scaling
    """
    # Apply seaborn style for clean, professional appearance
    plt.style.use('seaborn-v0_8')
    
    # Use 'husl' color palette for distinct, visually appealing colors
    # Important for distinguishing multiple buildings or models
    sns.set_palette("husl")
    
    # Configure figure and font parameters for optimal readability
    plt.rcParams['figure.figsize'] = (12, 8)    # Standard size for detailed analysis
    plt.rcParams['font.size'] = 10              # Base font size for good readability
    plt.rcParams['axes.titlesize'] = 12         # Emphasize plot titles
    plt.rcParams['axes.labelsize'] = 11         # Clear axis labels
    plt.rcParams['legend.fontsize'] = 10        # Readable legends


def plot_training_curves(training_history: Dict[str, List], 
                        title: str = "Training Progress",
                        save_path: Optional[str] = None):
    """
    Plot training curves for RL or forecasting models.
    
    Args:
        training_history: Dictionary with training metrics
        title: Plot title
        save_path: Path to save plot
    """
    setup_plotting_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot rewards/losses
    if 'rewards' in training_history:
        axes[0, 0].plot(training_history['rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Moving average
        if len(training_history['rewards']) > 10:
            window = min(100, len(training_history['rewards']) // 10)
            moving_avg = pd.Series(training_history['rewards']).rolling(window).mean()
            axes[0, 0].plot(moving_avg, color='red', alpha=0.7, label=f'Moving Avg ({window})')
            axes[0, 0].legend()
    
    # Plot episode lengths
    if 'episode_lengths' in training_history:
        axes[0, 1].plot(training_history['episode_lengths'])
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot losses if available
    if 'actor_losses' in training_history:
        axes[1, 0].plot(training_history['actor_losses'])
        axes[1, 0].set_title('Actor Loss')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    if 'critic_losses' in training_history:
        axes[1, 1].plot(training_history['critic_losses'])
        axes[1, 1].set_title('Critic Loss')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot epsilon decay if available
    if 'epsilon_values' in training_history:
        axes[1, 0].plot(training_history['epsilon_values'])
        axes[1, 0].set_title('Epsilon Decay')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_forecasting_results(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           model_name: str = "Model",
                           building_name: str = "Building",
                           target_name: str = "Energy",
                           save_path: Optional[str] = None):
    """
    Plot forecasting results comparison.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        building_name: Name of the building
        target_name: Name of the target variable
        save_path: Path to save plot
    """
    setup_plotting_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name} - {building_name} - {target_name}', fontsize=16)
    
    # Flatten arrays if needed
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Plot 1: Time Series Comparison
    # Shows temporal patterns and model's ability to track energy dynamics
    n_points = min(500, len(y_true_flat))  # Limit for visual clarity
    time_steps = np.arange(n_points)
    
    axes[0, 0].plot(time_steps, y_true_flat[:n_points], 
                   label='Actual', alpha=0.8, linewidth=1.5, color='blue')
    axes[0, 0].plot(time_steps, y_pred_flat[:n_points], 
                   label='Predicted', alpha=0.8, linewidth=1.5, color='red')
    
    axes[0, 0].set_title(f'Time Series Comparison (First {n_points} points)')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel(f'{target_name} (Energy Units)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add summary statistics to the plot
    mae = np.mean(np.abs(y_true_flat[:n_points] - y_pred_flat[:n_points]))
    axes[0, 0].text(0.02, 0.98, f'MAE: {mae:.3f}', transform=axes[0, 0].transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Scatter Plot (Predicted vs. Actual)
    # Assesses prediction accuracy and identifies systematic biases
    axes[0, 1].scatter(y_true_flat, y_pred_flat, alpha=0.5, s=20, edgecolors='none')
    
    # Perfect prediction line (diagonal)
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 
                   'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
    
    # Add correlation coefficient
    correlation = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
    axes[0, 1].set_xlabel(f'True {target_name} Values')
    axes[0, 1].set_ylabel(f'Predicted {target_name} Values')
    axes[0, 1].set_title(f'Prediction Accuracy (r = {correlation:.3f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Ensure equal aspect ratio for better interpretation
    axes[0, 1].set_aspect('equal', adjustable='box')
    
    # Plot 3: Residual Plot
    # Diagnoses model bias, heteroscedasticity, and prediction patterns
    residuals = y_true_flat - y_pred_flat
    axes[1, 0].scatter(y_pred_flat, residuals, alpha=0.5, s=20, edgecolors='none')
    
    # Zero line for reference (perfect predictions)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2,
                      label='Zero Error Line')
    
    # Add trend line to identify systematic bias
    if len(residuals) > 10:
        z = np.polyfit(y_pred_flat, residuals, 1)
        p = np.poly1d(z)
        axes[1, 0].plot(y_pred_flat, p(y_pred_flat), "g-", alpha=0.7,
                       label=f'Trend (slope={z[0]:.4f})')
    
    axes[1, 0].set_xlabel(f'Predicted {target_name} Values')
    axes[1, 0].set_ylabel('Residuals (True - Predicted)')
    axes[1, 0].set_title('Residual Analysis')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add residual statistics
    residual_std = np.std(residuals)
    axes[1, 0].text(0.02, 0.98, f'Std: {residual_std:.3f}', 
                   transform=axes[1, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Plot 4: Residual Distribution
    # Analyzes error distribution normality and identifies outliers
    n_bins = min(50, max(10, len(residuals) // 20))  # Adaptive bin count
    
    axes[1, 1].hist(residuals, bins=n_bins, alpha=0.7, density=True, 
                   color='skyblue', edgecolor='black', linewidth=0.5)
    
    # Overlay normal distribution for comparison
    if len(residuals) > 10:
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        normal_curve = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        axes[1, 1].plot(x, normal_curve, 'r-', linewidth=2, 
                       label=f'Normal (μ={mu:.3f}, σ={sigma:.3f})')
    
    # Add vertical line at zero
    axes[1, 1].axvline(x=0, color='green', linestyle='--', alpha=0.8, linewidth=2,
                      label='Zero Error')
    
    axes[1, 1].set_xlabel('Residuals (True - Predicted)')
    axes[1, 1].set_ylabel('Probability Density')
    axes[1, 1].set_title('Error Distribution Analysis')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save high-quality figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Forecasting results saved to: {save_path}")
    
    plt.show()


def create_results_table(results: Dict[str, Dict[str, Dict[str, float]]],
                        save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a comprehensive results table for model comparison and analysis.
    
    This function transforms nested experimental results into a structured
    table format suitable for statistical analysis, publication, and
    model selection. It handles complex experimental designs with multiple
    algorithms, buildings, and evaluation metrics.
    
    Table structure and organization:
    
    1. Hierarchical indexing:
       - Primary index: Building/Parameter combinations
       - Secondary index: Evaluation metrics (MAE, RMSE, R², MAPE)
       - Columns: Different algorithms or model variants
    
    2. Statistical comparison benefits:
       - Easy identification of best-performing algorithms
       - Clear visualization of performance across buildings
       - Enables statistical significance testing
       - Supports model selection and hyperparameter analysis
    
    3. Publication-ready format:
       - Rounded values for readability
       - Consistent decimal places across metrics
       - Professional table structure
       - CSV export for further analysis
    
    Typical use cases:
    - Algorithm comparison (LSTM vs. Transformer vs. Linear models)
    - Cross-building performance analysis
    - Hyperparameter sensitivity studies
    - Ablation studies and feature importance
    - Statistical significance testing preparation
    
    Args:
        results: Nested dictionary with structure:
                {algorithm_name: {
                    building_or_param: {
                        metric_name: metric_value
                    }
                }}
                Example:
                {
                    'LSTM': {
                        'Building_1': {'mae': 0.15, 'rmse': 0.23, 'r2': 0.85},
                        'Building_2': {'mae': 0.18, 'rmse': 0.27, 'r2': 0.82}
                    },
                    'Transformer': {
                        'Building_1': {'mae': 0.12, 'rmse': 0.20, 'r2': 0.88},
                        'Building_2': {'mae': 0.16, 'rmse': 0.24, 'r2': 0.84}
                    }
                }
        save_path: Optional path to save the table as CSV
        
    Returns:
        DataFrame with hierarchical index (Building/Parameter, Metric)
        and algorithm performance values as columns
        
    Example:
        >>> results = collect_experiment_results()
        >>> table = create_results_table(results, 'model_comparison.csv')
        >>> print(table.loc[('Building_1', 'rmse')])  # RMSE for Building_1 across algorithms
    """
    # Flatten nested results into long-format DataFrame
    # This intermediate step enables flexible pivoting and analysis
    rows = []
    
    for algorithm, algorithm_results in results.items():
        for building_param, metrics in algorithm_results.items():
            # Handle missing metrics gracefully
            if not isinstance(metrics, dict):
                print(f"Warning: Invalid metrics format for {algorithm}-{building_param}")
                continue
                
            for metric, value in metrics.items():
                # Validate numeric values
                if not isinstance(value, (int, float)) or np.isnan(value):
                    print(f"Warning: Invalid value {value} for {algorithm}-{building_param}-{metric}")
                    continue
                    
                rows.append({
                    'Algorithm': algorithm,
                    'Building/Parameter': building_param,
                    'Metric': metric,
                    'Value': value
                })
    
    if not rows:
        print("Warning: No valid results found")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Create pivot table for structured comparison
    # Hierarchical index enables easy slicing and analysis
    try:
        pivot_df = df.pivot_table(
            index=['Building/Parameter', 'Metric'],
            columns='Algorithm',
            values='Value',
            aggfunc='mean'  # Handle potential duplicates
        ).round(4)
    except Exception as e:
        print(f"Error creating pivot table: {e}")
        return df
    
    # Add summary statistics for each metric
    # This helps identify best-performing algorithms
    if len(pivot_df.columns) > 1:
        pivot_df['Mean'] = pivot_df.mean(axis=1).round(4)
        pivot_df['Std'] = pivot_df.std(axis=1).round(4)
        pivot_df['Best_Algorithm'] = pivot_df.drop(['Mean', 'Std'], axis=1).idxmin(axis=1)
    
    # Save table with timestamp and metadata
    if save_path:
        # Save main table
        pivot_df.to_csv(save_path)
        
        # Save metadata summary
        metadata_path = save_path.replace('.csv', '_metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"Results Table Metadata\n")
            f.write(f"========================\n")
            f.write(f"Algorithms: {list(pivot_df.columns)}\n")
            f.write(f"Buildings/Parameters: {pivot_df.index.get_level_values(0).unique().tolist()}\n")
            f.write(f"Metrics: {pivot_df.index.get_level_values(1).unique().tolist()}\n")
            f.write(f"Total experiments: {len(df)}\n")
        
        print(f"Results table saved to: {save_path}")
        print(f"Metadata saved to: {metadata_path}")
        
    return pivot_df


def plot_comparative_results(results_df: pd.DataFrame,
                           metric: str = 'rmse',
                           save_path: Optional[str] = None):
    """
    Plot comparative results across algorithms and buildings.
    
    This function creates dual visualization for comprehensive algorithm
    comparison across different buildings and experimental conditions.
    It enables clear identification of best-performing models and
    understanding of performance consistency.
    
    Visualization components:
    
    1. Bar Plot (Left panel):
       - X-axis: Algorithms/Models
       - Y-axis: Performance metric values
       - Colors: Different buildings/parameters
       - Purpose: Direct comparison of algorithm performance
       - Insights: Which algorithm performs best overall?
    
    2. Heatmap (Right panel):
       - Rows: Buildings/Parameters
       - Columns: Algorithms/Models
       - Color intensity: Performance metric values
       - Purpose: Pattern identification across conditions
       - Insights: Which building-algorithm combinations work best?
    
    Interpretation guidelines:
    
    For error metrics (MAE, RMSE, MAPE):
    - Lower values (darker colors) indicate better performance
    - Look for consistent performers across buildings
    - Identify algorithm-building mismatches
    
    For accuracy metrics (R², correlation):
    - Higher values (lighter colors) indicate better performance
    - Consistent high values show robust algorithms
    
    Statistical significance considerations:
    - Visual differences should be validated with statistical tests
    - Look for patterns rather than individual values
    - Consider practical significance vs. statistical significance
    
    Energy forecasting specific insights:
    - Some algorithms may specialize in certain building types
    - Weather-dependent buildings may favor different models
    - Building size and complexity affect model performance
    - Seasonal patterns may influence algorithm effectiveness
    
    Args:
        results_df: DataFrame from create_results_table() with hierarchical index
                   (Building/Parameter, Metric) and algorithm columns
        metric: Metric to visualize (e.g., 'rmse', 'mae', 'r2', 'mape')
        save_path: Optional path to save the figure
        
    Example:
        >>> results_table = create_results_table(experiment_results)
        >>> plot_comparative_results(results_table, 'rmse', 'algorithm_comparison.png')
    """
    setup_plotting_style()
    
    # Extract data for specific metric from hierarchical DataFrame
    try:
        metric_data = results_df.xs(metric, level='Metric')
    except KeyError:
        available_metrics = results_df.index.get_level_values('Metric').unique()
        raise ValueError(f"Metric '{metric}' not found. Available metrics: {available_metrics.tolist()}")
    
    if metric_data.empty:
        raise ValueError(f"No data available for metric '{metric}'")
    
    # Create dual visualization layout
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bar Plot - Algorithm comparison across buildings
    # Transpose to get algorithms on x-axis, buildings as separate bars
    metric_data.T.plot(kind='bar', ax=axes[0], width=0.8)
    axes[0].set_title(f'{metric.upper()} Comparison by Algorithm\\n(Lower is better for error metrics)')
    axes[0].set_xlabel('Algorithm')
    axes[0].set_ylabel(f'{metric.upper()} Value')
    axes[0].legend(title='Building/Parameter', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3, axis='y')  # Only horizontal grid lines
    
    # Rotate x-axis labels for better readability
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars for precise reading
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.3f', rotation=90, fontsize=8)
    
    # Plot 2: Heatmap - Pattern identification across building-algorithm combinations
    # Choose colormap based on metric type (lower vs. higher is better)
    error_metrics = ['mae', 'mse', 'rmse', 'mape']
    cmap = 'viridis_r' if metric.lower() in error_metrics else 'viridis'
    
    # Create heatmap with annotations
    heatmap = sns.heatmap(metric_data, 
                         annot=True, 
                         fmt='.4f', 
                         cmap=cmap, 
                         ax=axes[1],
                         cbar_kws={'label': f'{metric.upper()} Value'},
                         square=False)
    
    axes[1].set_title(f'{metric.upper()} Performance Matrix\\n(Buildings vs. Algorithms)')
    axes[1].set_xlabel('Algorithm')
    axes[1].set_ylabel('Building/Parameter')
    
    # Rotate labels for better readability
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)
    
    # Highlight best performer in each row (building)
    if metric.lower() in error_metrics:
        # For error metrics, find minimum (best) values
        best_algos = metric_data.idxmin(axis=1)
    else:
        # For accuracy metrics, find maximum (best) values
        best_algos = metric_data.idxmax(axis=1)
    
    # Add text annotations for best performers
    for i, (building, best_algo) in enumerate(best_algos.items()):
        j = metric_data.columns.get_loc(best_algo)
        axes[1].add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=3))
    
    plt.tight_layout()
    
    # Add overall statistics as text
    error_metrics = ['mae', 'mse', 'rmse', 'mape']
    stats_text = f"Overall Statistics for {metric.upper()}:\\n"
    stats_text += f"Best Algorithm: {metric_data.mean().idxmin() if metric.lower() in error_metrics else metric_data.mean().idxmax()}\\n"
    stats_text += f"Mean ± Std: {metric_data.values.mean():.4f} ± {metric_data.values.std():.4f}\\n"
    stats_text += f"Range: [{metric_data.values.min():.4f}, {metric_data.values.max():.4f}]"
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Save high-quality figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Comparative results saved to: {save_path}")
    
    plt.show()


def plot_cross_building_generalization(results: Dict[str, Dict[str, Dict]],
                                     metric: str = 'rmse',
                                     save_path: Optional[str] = None):
    """
    Plot cross-building generalization results.
    
    This function visualizes the critical capability of energy forecasting models
    to generalize across different buildings. It creates heatmap matrices showing
    performance when models trained on one building are tested on others.
    
    Cross-building generalization importance:
    
    1. Real-world deployment scenarios:
       - New buildings without historical data
       - Transfer learning from data-rich to data-poor buildings
       - Scaling energy management to large building portfolios
       - Rapid deployment in new locations
    
    2. Model robustness assessment:
       - Tests if models learn fundamental energy patterns
       - Identifies building-specific overfitting
       - Validates feature engineering choices
       - Assesses model architecture generalizability
    
    3. Building similarity analysis:
       - Diagonal elements: within-building performance (baseline)
       - Off-diagonal elements: cross-building transfer performance
       - Performance patterns reveal building similarities
       - Identifies optimal source buildings for transfer learning
    
    Interpretation guidelines:
    
    1. Matrix structure:
       - Rows: Training buildings (source)
       - Columns: Test buildings (target)
       - Values: Performance metric (lower is better for error metrics)
       - Color intensity: Performance level
    
    2. Performance patterns:
       - Dark diagonal: Good within-building performance
       - Light off-diagonal: Poor generalization
       - Consistent rows: Robust source buildings
       - Consistent columns: Easy target buildings
    
    3. Transfer learning insights:
       - Best source building: Row with lowest off-diagonal values
       - Hardest target building: Column with highest values
       - Building clusters: Similar performance patterns
       - Architecture validation: Consistent performance across buildings
    
    Energy domain considerations:
    - Similar building types (office, residential) transfer better
    - Climate and weather patterns affect transferability
    - Building size and occupancy patterns influence generalization
    - HVAC systems and control strategies impact model transfer
    
    Args:
        results: Nested dictionary from cross_building_split() with structure:
                {algorithm: {train_building: {test_building: {metric: value}}}}
        metric: Performance metric to visualize (e.g., 'rmse', 'mae', 'r2')
        save_path: Optional path to save the figure
        
    Example:
        >>> cross_results = run_cross_building_experiments()
        >>> plot_cross_building_generalization(cross_results, 'rmse', 'generalization.png')
    """
    setup_plotting_style()
    
    # Validate and prepare data for visualization
    if not results:
        raise ValueError("Empty results dictionary provided")
    
    algorithms = list(results.keys())
    
    # Get building names from first algorithm
    first_algo = algorithms[0]
    train_buildings = list(results[first_algo].keys())
    
    if not train_buildings:
        raise ValueError(f"No training buildings found for algorithm {first_algo}")
    
    # Get test buildings from first training building
    first_train_building = train_buildings[0]
    test_buildings = list(results[first_algo][first_train_building].keys())
    
    if not test_buildings:
        raise ValueError(f"No test buildings found for {first_algo}-{first_train_building}")
    
    # Create subplot layout: one row per algorithm
    fig, axes = plt.subplots(len(algorithms), 1, figsize=(12, 5 * len(algorithms)))
    if len(algorithms) == 1:
        axes = [axes]
    
    # Track overall statistics for summary
    all_values = []
    
    for i, algorithm in enumerate(algorithms):
        # Create performance matrix: train_buildings x test_buildings
        matrix = np.full((len(train_buildings), len(test_buildings)), np.nan)
        
        for j, train_building in enumerate(train_buildings):
            for k, test_building in enumerate(test_buildings):
                try:
                    value = results[algorithm][train_building][test_building][metric]
                    matrix[j, k] = value
                    all_values.append(value)
                except KeyError:
                    # Handle missing combinations gracefully
                    continue
        
        # Choose colormap based on metric type
        error_metrics = ['mae', 'mse', 'rmse', 'mape']
        cmap = 'Reds' if metric.lower() in error_metrics else 'Greens'
        
        # Create heatmap with improved styling
        im = axes[i].imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')
        
        # Enhanced title with interpretation guidance
        perf_note = "(Lower is Better)" if metric.lower() in error_metrics else "(Higher is Better)"
        axes[i].set_title(f'{algorithm} - Cross-Building {metric.upper()} {perf_note}', 
                         fontsize=14, fontweight='bold')
        
        # Set labels and ticks
        axes[i].set_xlabel('Test Building (Target)', fontsize=12)
        axes[i].set_ylabel('Train Building (Source)', fontsize=12)
        axes[i].set_xticks(range(len(test_buildings)))
        axes[i].set_xticklabels(test_buildings, rotation=45, ha='right')
        axes[i].set_yticks(range(len(train_buildings)))
        axes[i].set_yticklabels(train_buildings)
        
        # Add colorbar with proper labeling
        cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.set_label(f'{metric.upper()} Value', fontsize=11)
        
        # Add value annotations with adaptive text color
        for j in range(len(train_buildings)):
            for k in range(len(test_buildings)):
                if not np.isnan(matrix[j, k]):
                    # Choose text color based on value for visibility
                    text_color = 'white' if matrix[j, k] > np.nanmean(matrix) else 'black'
                    axes[i].text(k, j, f'{matrix[j, k]:.3f}',
                               ha='center', va='center', color=text_color,
                               fontweight='bold', fontsize=10)
        
        # Highlight diagonal (within-building performance) with border
        for j in range(min(len(train_buildings), len(test_buildings))):
            if train_buildings[j] in test_buildings:
                k = test_buildings.index(train_buildings[j])
                rect = plt.Rectangle((k-0.5, j-0.5), 1, 1, 
                                   fill=False, edgecolor='blue', linewidth=3)
                axes[i].add_patch(rect)
        
        # Add performance summary statistics
        if not np.isnan(matrix).all():
            within_building = np.array([matrix[j, test_buildings.index(train_buildings[j])] 
                                      for j in range(len(train_buildings)) 
                                      if train_buildings[j] in test_buildings and 
                                      not np.isnan(matrix[j, test_buildings.index(train_buildings[j])])])
            
            cross_building = matrix[~np.eye(matrix.shape[0], matrix.shape[1], dtype=bool)]
            cross_building = cross_building[~np.isnan(cross_building)]
            
            if len(within_building) > 0 and len(cross_building) > 0:
                summary_text = f"Within-building: {np.mean(within_building):.3f}±{np.std(within_building):.3f}\\n"
                summary_text += f"Cross-building: {np.mean(cross_building):.3f}±{np.std(cross_building):.3f}"
                
                axes[i].text(0.02, 0.98, summary_text, transform=axes[i].transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='lightblue', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    
    # Add overall summary if multiple algorithms
    if len(algorithms) > 1 and all_values:
        overall_stats = f"Overall Cross-Building Analysis:\\n"
        overall_stats += f"Algorithms: {len(algorithms)}, Buildings: {len(train_buildings)}\\n"
        overall_stats += f"Total comparisons: {len(all_values)}\\n"
        overall_stats += f"Mean {metric.upper()}: {np.mean(all_values):.4f} ± {np.std(all_values):.4f}"
        
        fig.text(0.02, 0.98, overall_stats, fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                verticalalignment='top')
    
    # Save high-quality figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Cross-building generalization plot saved to: {save_path}")
    
    plt.show()


def plot_rl_comparison(centralized_results: Dict,
                      decentralized_results: Dict,
                      save_path: Optional[str] = None):
    """
    Compare centralized vs decentralized RL results.
    
    Args:
        centralized_results: Results from centralized approach
        decentralized_results: Results from decentralized approach
        save_path: Path to save plot
    """
    setup_plotting_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Centralized vs Decentralized RL Comparison', fontsize=16)
    
    # Rewards comparison
    axes[0, 0].plot(centralized_results['rewards'], label='Centralized', alpha=0.7)
    
    # For decentralized, sum rewards across buildings
    if isinstance(decentralized_results, dict) and 0 in decentralized_results:
        # Sum rewards across all buildings
        total_rewards = []
        for episode in range(len(decentralized_results[0]['rewards'])):
            total_reward = sum(decentralized_results[i]['rewards'][episode] 
                             for i in range(len(decentralized_results)))
            total_rewards.append(total_reward)
        axes[0, 0].plot(total_rewards, label='Decentralized (Sum)', alpha=0.7)
    
    axes[0, 0].set_title('Total Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Moving averages
    window = 100
    if len(centralized_results['rewards']) > window:
        cent_ma = pd.Series(centralized_results['rewards']).rolling(window).mean()
        axes[0, 1].plot(cent_ma, label='Centralized MA', alpha=0.8)
    
    if 'total_rewards' in locals():
        dec_ma = pd.Series(total_rewards).rolling(window).mean()
        axes[0, 1].plot(dec_ma, label='Decentralized MA', alpha=0.8)
    
    axes[0, 1].set_title(f'Moving Average (window={window})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[1, 0].plot(centralized_results['episode_lengths'], label='Centralized', alpha=0.7)
    if isinstance(decentralized_results, dict) and 0 in decentralized_results:
        axes[1, 0].plot(decentralized_results[0]['episode_lengths'], 
                       label='Decentralized', alpha=0.7)
    
    axes[1, 0].set_title('Episode Lengths')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final performance comparison (box plot of last 100 episodes)
    final_cent = centralized_results['rewards'][-100:]
    final_dec = total_rewards[-100:] if 'total_rewards' in locals() else []
    
    if final_dec:
        axes[1, 1].boxplot([final_cent, final_dec], labels=['Centralized', 'Decentralized'])
        axes[1, 1].set_title('Final Performance (Last 100 Episodes)')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save high-quality figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()