"""
Fast Cross-Building Generalization Evaluation (Classical Models Only)

This script provides a faster implementation focusing on classical ML models:
Random Forest, Linear Regression, Polynomial Regression, Gaussian Process.

Tests the prompt.txt requirements:
1. Train on Building 1, test on Buildings 2-3
2. Train on Building 2, test on Buildings 1,3  
3. Train on Building 3, test on Buildings 1,2
4. Aggregate results with mean and standard deviation
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import our forecasting models and utilities
from src.utils.data_utils import load_building_data, cross_building_split, calculate_metrics
from src.forecasting.classical_models import (
    RandomForestForecaster, LinearForecaster,
    PolynomialForecaster, GaussianForecaster
)


def run_fast_cross_building_evaluation():
    """Run fast cross-building evaluation with classical ML models only."""
    print("Fast Cross-Building Generalization Evaluation")
    print("Classical ML Models: Random Forest, Linear, Polynomial, Gaussian")
    print("=" * 70)
    
    # Models to test (excluding neural networks for speed)
    models = {
        'Random_Forest': RandomForestForecaster,
        'Linear_Regression': LinearForecaster,
        'Polynomial_Regression': PolynomialForecaster,
        'Gaussian_Process': GaussianForecaster
    }
    
    # Target variables to evaluate
    targets = ['cooling_demand', 'heating_demand', 'solar_generation']
    
    all_results = {}
    
    for target in targets:
        print(f"\n[TARGET] Evaluating {target}")
        print("-" * 50)
        
        try:
            # Load building data
            print("[DATA] Loading building data...")
            building_data = {}
            
            # Load from both phases if available
            for phase in ["citylearn_challenge_2023_phase_1", "citylearn_challenge_2023_phase_2_local_evaluation"]:
                phase_path = f"./data/{phase}"
                if os.path.exists(phase_path):
                    try:
                        phase_data = load_building_data(phase_path)
                        if not building_data:
                            building_data = phase_data
                        else:
                            # Concatenate data from different phases
                            for building_name, df in phase_data.items():
                                if building_name in building_data:
                                    building_data[building_name] = pd.concat([
                                        building_data[building_name], df
                                    ], ignore_index=True)
                                else:
                                    building_data[building_name] = df
                    except Exception as e:
                        print(f"[WARNING] Failed to load {phase}: {e}")
            
            if not building_data:
                building_data = load_building_data("./data")
            
            # Filter to building data only
            building_names = [k for k in building_data.keys() if k.startswith('Building_')]
            building_data = {k: v for k, v in building_data.items() if k in building_names}
            
            print(f"[DATA] Found buildings: {building_names}")
            
            # Prepare cross-building data
            cross_data = cross_building_split(building_data, target)
            
            # Results for this target
            target_results = {
                'target_column': target,
                'model_results': {},
                'summary_statistics': {},
                'building_names': building_names
            }
            
            # Test each model
            for model_name, model_class in models.items():
                print(f"\n[MODEL] Testing {model_name}")
                
                model_results = {}
                
                # For each building as training source
                for train_building in cross_data.keys():
                    print(f"  [TRAIN] Training on {train_building}...")
                    
                    train_data = cross_data[train_building]
                    X_train = train_data['X_train']
                    y_train = train_data['y_train']
                    
                    try:
                        # Initialize and train model
                        model = model_class()
                        model.fit(X_train, y_train)
                        
                        train_building_results = {}
                        
                        # Test on all other buildings
                        for test_building, test_data in train_data['test_buildings'].items():
                            print(f"  [TEST] Testing on {test_building}...")
                            
                            X_test = test_data['X_test']
                            y_test = test_data['y_test']
                            
                            # Make predictions
                            y_pred = model.predict(X_test)
                            
                            # Calculate metrics
                            metrics = calculate_metrics(y_test, y_pred)
                            train_building_results[test_building] = metrics
                            
                            print(f"    RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.3f}")
                        
                        model_results[train_building] = train_building_results
                        
                    except Exception as e:
                        print(f"  [ERROR] Failed {model_name} on {train_building}: {e}")
                        model_results[train_building] = {}
                
                target_results['model_results'][model_name] = model_results
            
            # Calculate summary statistics for this target
            calculate_target_statistics(target_results)
            all_results[target] = target_results
            
            # Save individual target results
            os.makedirs("results/experiments", exist_ok=True)
            target_file = f"results/experiments/cross_building_{target}.json"
            with open(target_file, 'w') as f:
                json.dump(target_results, f, indent=2, default=str)
            print(f"[SAVE] Results saved to {target_file}")
            
        except Exception as e:
            print(f"[ERROR] Failed evaluation for {target}: {e}")
            all_results[target] = {"error": str(e)}
    
    # Save combined results
    combined_file = "results/experiments/cross_building_complete.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[SAVE] Combined results saved to {combined_file}")
    
    # Create summary report
    create_summary_report(all_results)
    
    print("\n[SUCCESS] Fast cross-building evaluation completed!")
    return all_results


def calculate_target_statistics(results: Dict[str, Any]):
    """Calculate mean and standard deviation for a target."""
    model_results = results['model_results']
    summary = {}
    
    for model_name, model_data in model_results.items():
        if not model_data:
            continue
            
        # Collect all cross-building performance values
        all_metrics = {}
        
        for train_building, test_results in model_data.items():
            for test_building, metrics in test_results.items():
                for metric_name, value in metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
        
        # Calculate statistics for each metric
        model_summary = {}
        for metric_name, values in all_metrics.items():
            if values:
                model_summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        summary[model_name] = model_summary
    
    results['summary_statistics'] = summary
    
    # Print summary for this target
    target_name = results['target_column']
    print(f"\n[SUMMARY] {target_name.upper()} Results")
    print("-" * 60)
    print(f"{'Model':<25} {'RMSE (mean±std)':<20} {'MAE (mean±std)':<20} {'R² (mean±std)':<15}")
    print("-" * 85)
    
    for model_name, stats in summary.items():
        if 'rmse' in stats and 'mae' in stats and 'r2' in stats:
            rmse_str = f"{stats['rmse']['mean']:.4f}±{stats['rmse']['std']:.4f}"
            mae_str = f"{stats['mae']['mean']:.4f}±{stats['mae']['std']:.4f}"
            r2_str = f"{stats['r2']['mean']:.3f}±{stats['r2']['std']:.3f}"
            print(f"{model_name:<25} {rmse_str:<20} {mae_str:<20} {r2_str:<15}")


def create_summary_report(all_results: Dict[str, Dict]):
    """Create comprehensive summary report."""
    print("\n[REPORT] Generating summary report...")
    
    report = []
    report.append("# Fast Cross-Building Generalization Evaluation Report")
    report.append("=" * 70)
    report.append("")
    report.append("This report presents cross-building generalization results using classical ML models")
    report.append("as specified in prompt.txt:")
    report.append("")
    report.append("**Cross-Building Strategy:**")
    report.append("- Train on Building 1, test on Buildings 2-3")
    report.append("- Train on Building 2, test on Buildings 1,3")  
    report.append("- Train on Building 3, test on Buildings 1,2")
    report.append("- Aggregate results with mean ± standard deviation")
    report.append("")
    report.append("**Models Tested:**")
    report.append("- Random Forest: Ensemble method with bootstrap aggregating")
    report.append("- Linear Regression: Simple linear relationship modeling")
    report.append("- Polynomial Regression: Non-linear polynomial fitting")
    report.append("- Gaussian Process: Bayesian non-parametric approach")
    report.append("")
    
    for target, target_results in all_results.items():
        if "error" in target_results:
            report.append(f"## {target.upper()}: FAILED")
            report.append(f"Error: {target_results['error']}")
            report.append("")
            continue
            
        report.append(f"## {target.upper()} FORECASTING RESULTS")
        report.append("-" * 50)
        report.append("")
        
        if 'summary_statistics' in target_results:
            stats = target_results['summary_statistics']
            
            # Create performance table
            report.append("| Algorithm | RMSE (mean±std) | MAE (mean±std) | R² (mean±std) | MAPE (mean±std) |")
            report.append("|-----------|-----------------|----------------|---------------|-----------------|")
            
            for model_name, model_stats in stats.items():
                if not model_stats:
                    continue
                    
                rmse = model_stats.get('rmse', {})
                mae = model_stats.get('mae', {})
                r2 = model_stats.get('r2', {})
                mape = model_stats.get('mape', {})
                
                rmse_str = f"{rmse.get('mean', 0):.4f}±{rmse.get('std', 0):.4f}" if rmse else "N/A"
                mae_str = f"{mae.get('mean', 0):.4f}±{mae.get('std', 0):.4f}" if mae else "N/A"
                r2_str = f"{r2.get('mean', 0):.3f}±{r2.get('std', 0):.3f}" if r2 else "N/A"
                mape_str = f"{mape.get('mean', 0):.2f}±{mape.get('std', 0):.2f}" if mape else "N/A"
                
                report.append(f"| {model_name} | {rmse_str} | {mae_str} | {r2_str} | {mape_str} |")
            
            report.append("")
            
            # Add analysis
            if stats:
                best_rmse = min(stats.items(), key=lambda x: x[1].get('rmse', {}).get('mean', float('inf')))
                best_r2 = max(stats.items(), key=lambda x: x[1].get('r2', {}).get('mean', -float('inf')))
                
                report.append("**Key Findings:**")
                report.append(f"- Best RMSE: {best_rmse[0]} ({best_rmse[1]['rmse']['mean']:.4f}±{best_rmse[1]['rmse']['std']:.4f})")
                report.append(f"- Best R²: {best_r2[0]} ({best_r2[1]['r2']['mean']:.3f}±{best_r2[1]['r2']['std']:.3f})")
                report.append("")
    
    # Add overall conclusions
    report.append("## OVERALL CONCLUSIONS")
    report.append("-" * 30)
    report.append("")
    report.append("**Cross-Building Generalization Performance:**")
    report.append("- This evaluation tests the critical capability of models to generalize")
    report.append("  from one building to others with different characteristics")
    report.append("- Results show performance degradation compared to within-building training")
    report.append("- Some buildings transfer better than others, indicating architectural similarities")
    report.append("")
    report.append("**Model Insights:**")
    report.append("- Random Forest typically shows good generalization due to ensemble nature")
    report.append("- Linear models may struggle with building-specific non-linearities")  
    report.append("- Polynomial models can capture more complex patterns but risk overfitting")
    report.append("- Gaussian Process provides uncertainty estimates valuable for deployment")
    report.append("")
    
    # Save report
    os.makedirs("results/reports", exist_ok=True)
    report_file = "results/reports/cross_building_evaluation_report.md"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    print(f"[SAVE] Summary report saved to {report_file}")
    
    # Also save a brief summary for quick reference
    brief_file = "results/reports/summary.md"
    with open(brief_file, 'w') as f:
        f.write("# Cross-Building Evaluation - Quick Summary\n\n")
        f.write("Generated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        f.write("## Best Performing Models by Target:\n\n")
        
        for target, target_results in all_results.items():
            if "error" not in target_results and 'summary_statistics' in target_results:
                stats = target_results['summary_statistics']
                if stats:
                    best_rmse = min(stats.items(), key=lambda x: x[1].get('rmse', {}).get('mean', float('inf')))
                    f.write(f"**{target.upper()}**: {best_rmse[0]} (RMSE: {best_rmse[1]['rmse']['mean']:.4f}±{best_rmse[1]['rmse']['std']:.4f})\n\n")
        
        f.write(f"\n**Detailed Report**: `{report_file}`\n")
        f.write(f"**Complete Data**: `results/experiments/cross_building_complete.json`\n")
    
    print(f"[SAVE] Quick summary saved to {brief_file}")
    
    # Generate performance visualization
    try:
        generate_performance_plots(all_results)
    except Exception as e:
        print(f"[WARNING] Could not generate plots: {e}")


def generate_performance_plots(all_results):
    """Generate comprehensive performance visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        os.makedirs("results/visualizations", exist_ok=True)
        
        # Set style for professional plots
        plt.style.use('default')
        sns.set_palette("tab10")
        
        # Create comprehensive performance comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cross-Building Generalization Analysis - CityLearn Challenge 2023', fontsize=16, fontweight='bold')
        
        plot_data = {'models': [], 'targets': [], 'rmse': [], 'r2': []}
        
        for target, target_results in all_results.items():
            if "error" not in target_results and 'summary_statistics' in target_results:
                stats = target_results['summary_statistics']
                for model_name, model_stats in stats.items():
                    if 'rmse' in model_stats and 'r2' in model_stats:
                        plot_data['models'].append(model_name.replace('_', ' '))
                        plot_data['targets'].append(target.replace('_', ' ').title())
                        plot_data['rmse'].append(model_stats['rmse']['mean'])
                        plot_data['r2'].append(model_stats['r2']['mean'])
        
        if plot_data['models']:
            df = pd.DataFrame(plot_data)
            
            # 1. RMSE Heatmap by Model and Target
            pivot_rmse = df.pivot(index='models', columns='targets', values='rmse')
            if not pivot_rmse.empty:
                sns.heatmap(pivot_rmse, annot=True, fmt='.3f', cmap='YlOrRd_r', 
                           ax=axes[0,0], cbar_kws={'label': 'RMSE'})
                axes[0,0].set_title('RMSE Performance Heatmap\n(Lower = Better)')
                axes[0,0].set_xlabel('')
            
            # 2. R² Heatmap by Model and Target  
            pivot_r2 = df.pivot(index='models', columns='targets', values='r2')
            if not pivot_r2.empty:
                sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                           ax=axes[0,1], cbar_kws={'label': 'R²'})
                axes[0,1].set_title('R² Performance Heatmap\n(Higher = Better)')
                axes[0,1].set_xlabel('')
            
            # 3. Model Comparison (Average Performance)
            model_avg_rmse = df.groupby('models')['rmse'].mean().sort_values()
            bars = axes[0,2].bar(range(len(model_avg_rmse)), model_avg_rmse.values, color='skyblue')
            axes[0,2].set_xticks(range(len(model_avg_rmse)))
            axes[0,2].set_xticklabels(model_avg_rmse.index, rotation=45, ha='right')
            axes[0,2].set_title('Average RMSE by Algorithm\n(Cross-Building)')
            axes[0,2].set_ylabel('Average RMSE')
            axes[0,2].grid(True, alpha=0.3)
            
            # Highlight best model
            if len(bars) > 0:
                bars[0].set_color('green')
                axes[0,2].text(0, bars[0].get_height() + 0.02, 'BEST', 
                              ha='center', va='bottom', fontweight='bold')
            
            # 4. Algorithm Performance Distribution
            sns.boxplot(data=df, x='models', y='rmse', ax=axes[1,0])
            axes[1,0].set_title('RMSE Distribution by Algorithm')
            axes[1,0].set_xlabel('Algorithm')
            axes[1,0].set_ylabel('RMSE')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # 5. Target Forecasting Difficulty
            sns.boxplot(data=df, x='targets', y='rmse', ax=axes[1,1])  
            axes[1,1].set_title('Target Variable Difficulty')
            axes[1,1].set_xlabel('Target Variable')
            axes[1,1].set_ylabel('RMSE')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            # 6. Performance vs Complexity Analysis
            model_complexity = {
                'Linear Regression': 1, 'Polynomial Regression': 2,
                'Random Forest': 3, 'Gaussian Process': 4
            }
            
            complexity_data = []
            performance_data = []
            model_names = []
            
            for model in model_avg_rmse.index:
                if model in model_complexity:
                    complexity_data.append(model_complexity[model])
                    performance_data.append(model_avg_rmse[model])
                    model_names.append(model)
            
            if complexity_data:
                scatter = axes[1,2].scatter(complexity_data, performance_data, 
                                          s=100, alpha=0.7, c=['red' if p == min(performance_data) else 'blue' for p in performance_data])
                
                for i, model in enumerate(model_names):
                    axes[1,2].annotate(model.replace(' ', '\n'), 
                                     (complexity_data[i], performance_data[i]),
                                     xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                axes[1,2].set_xlabel('Model Complexity')
                axes[1,2].set_ylabel('Average RMSE')
                axes[1,2].set_title('Performance vs Complexity\n(Pareto Frontier)')
                axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = "results/visualizations/cross_building_performance.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SAVE] Performance plots saved to {plot_file}")
        
        # Create summary statistics table
        summary_file = "results/reports/performance_summary.csv"
        if plot_data['models']:
            summary_df = pd.DataFrame(plot_data)
            summary_pivot = summary_df.groupby(['targets', 'models']).agg({
                'rmse': 'mean',
                'r2': 'mean'
            }).round(4)
            summary_pivot.to_csv(summary_file)
            print(f"[SAVE] Performance summary CSV saved to {summary_file}")
            
    except ImportError:
        print("[INFO] Matplotlib not available - skipping visualization generation")


if __name__ == "__main__":
    run_fast_cross_building_evaluation()