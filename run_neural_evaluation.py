"""
Valutazione di Reti Neurali per la Gestione Energetica Intelligente degli Edifici
Implementazioni avanzate di LSTM, Transformer e modelli baseline

Questo modulo fornisce una valutazione completa delle architetture neurali per:
1. Previsione del consumo energetico degli edifici
2. Architetture neurali avanzate multiple (varianti LSTM + Transformer)
3. Addestramento con validazione e early stopping
4. Test di generalizzazione cross-building
5. Analisi di aggregazione a livello di quartiere
6. Previsione di generazione solare
7. Analisi delle performance e visualizzazioni complete
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from src.forecasting.lstm_models import (
    LSTMForecaster, 
    BidirectionalLSTMForecaster, 
    ConvLSTMForecaster
)
from src.forecasting.simple_transformer import (
    SimpleTransformerForecaster, 
    SimpleTimesFMForecaster
)
from src.forecasting.base_models import get_baseline_forecasters
from src.utils.data_utils import calculate_metrics, create_time_features
# Visualization utilities sono ora integrate in thesis_plots


class ComprehensiveNeuralEvaluator:
    """Comprehensive neural network evaluation for building energy forecasting."""
    
    def __init__(self):
        """Initialize comprehensive neural evaluator."""
        self.results = {}
        self.training_histories = {}
        self.model_instances = {}
        
        # Modelli neurali implementati
        self.neural_models = {
            'LSTM': LSTMForecaster(
                sequence_length=24,
                hidden_units=64,
                num_layers=2,
                dropout_rate=0.2,
                learning_rate=0.001
            ),
            'Transformer': SimpleTransformerForecaster(
                sequence_length=24,
                d_model=64,
                num_heads=4,
                num_layers=2,
                dropout_rate=0.1,
                learning_rate=0.001
            ),
            'TimesFM': SimpleTimesFMForecaster(
                sequence_length=24,
                d_model=128,
                num_heads=8,
                num_layers=3,
                patch_size=4,
                learning_rate=0.0001
            )
        }
        
        # Baseline models for comprehensive comparison
        self.baseline_models = get_baseline_forecasters()
        
        # All models combined
        self.all_models = {**self.neural_models, **self.baseline_models}
    
    def load_complete_dataset(self) -> Dict[str, pd.DataFrame]:
        """Load complete CityLearn 2023 dataset (Phase 1 and 2)."""
        print("[DATA] Loading complete CityLearn 2023 dataset...")
        
        # All available phases for maximum data
        phases = [
            'citylearn_challenge_2023_phase_1',
            'citylearn_challenge_2023_phase_2_local_evaluation',
            'citylearn_challenge_2023_phase_2_online_evaluation_1',
            'citylearn_challenge_2023_phase_2_online_evaluation_2',
            'citylearn_challenge_2023_phase_2_online_evaluation_3'
        ]
        
        buildings = {}
        
        for building_id in [1, 2, 3]:
            building_name = f'Building_{building_id}'
            all_data = []
            
            for phase in phases:
                file_path = f'data/{phase}/Building_{building_id}.csv'
                if os.path.exists(file_path):
                    data = pd.read_csv(file_path)
                    # Aggiunge caratteristiche temporali per l'analisi
                    data = create_time_features(data)
                    all_data.append(data)
                    print(f"  Loaded {phase}: {len(data)} samples")
            
            if all_data:
                combined = pd.concat(all_data, ignore_index=True)
                combined = combined.drop_duplicates().reset_index(drop=True)
                buildings[building_name] = combined
                print(f"  {building_name}: {len(combined)} total samples ({len(combined)/24:.1f} days)")
        
        return buildings
    
    def load_neighborhood_data(self) -> Dict[str, pd.DataFrame]:
        """Load weather and carbon intensity for neighborhood aggregation."""
        print("[DATA] Loading neighborhood-level data...")
        
        neighborhood_data = {}
        phase = 'citylearn_challenge_2023_phase_1'  # Use consistent phase
        
        # Load weather data
        weather_path = f'data/{phase}/weather.csv'
        if os.path.exists(weather_path):
            weather = pd.read_csv(weather_path)
            weather = create_time_features(weather)
            neighborhood_data['weather'] = weather
            print(f"  Weather data: {len(weather)} samples")
        
        # Load carbon intensity
        carbon_path = f'data/{phase}/carbon_intensity.csv'
        if os.path.exists(carbon_path):
            carbon = pd.read_csv(carbon_path)
            carbon = create_time_features(carbon)
            neighborhood_data['carbon_intensity'] = carbon
            print(f"  Carbon intensity: {len(carbon)} samples")
        
        return neighborhood_data
    
    def create_enhanced_sequences(self, data: pd.DataFrame, target: str, seq_len: int = 24):
        """Create enhanced sequences with all available features."""
        # Set di caratteristiche migliorate per l'analisi
        base_features = [
            'hour', 'day_of_week', 'month', 'is_weekend',
            'outdoor_dry_bulb_temperature', 'outdoor_relative_humidity',
            'indoor_dry_bulb_temperature', 'indoor_relative_humidity',
            'non_shiftable_load', 'occupant_count'
        ]
        
        # Add target to features
        all_features = base_features + [target]
        available_features = [f for f in all_features if f in data.columns]
        
        print(f"  Using features: {available_features}")
        
        feature_data = data[available_features].ffill().values
        
        X, y = [], []
        for i in range(seq_len, len(feature_data)):
            X.append(feature_data[i-seq_len:i, :])
            target_idx = available_features.index(target)
            y.append(feature_data[i, target_idx])
        
        return np.array(X), np.array(y), available_features
    
    def create_solar_sequences(self, buildings_data: Dict[str, pd.DataFrame], 
                              neighborhood_data: Dict[str, pd.DataFrame], seq_len: int = 24):
        """Create sequences for solar generation forecasting (neighborhood level)."""
        print("[SOLAR] Creating solar generation sequences...")
        
        # Combine building data for neighborhood context
        combined_features = []
        for building_name, data in buildings_data.items():
            # Select key features from each building
            building_features = data[['non_shiftable_load', 'occupant_count']].values
            combined_features.append(building_features)
        
        # Stack building features
        neighborhood_features = np.concatenate(combined_features, axis=1)
        
        # Add weather data if available
        if 'weather' in neighborhood_data:
            weather_features = ['outdoor_dry_bulb_temperature', 'diffuse_solar_irradiance', 
                               'direct_solar_irradiance', 'outdoor_relative_humidity']
            available_weather = [f for f in weather_features if f in neighborhood_data['weather'].columns]
            
            if available_weather:
                weather_data = neighborhood_data['weather'][available_weather].ffill().values
                # Ensure same length
                min_len = min(len(neighborhood_features), len(weather_data))
                neighborhood_features = neighborhood_features[:min_len]
                weather_data = weather_data[:min_len]
                
                # Combine all features
                all_features = np.concatenate([neighborhood_features, weather_data], axis=1)
            else:
                all_features = neighborhood_features
        else:
            all_features = neighborhood_features
        
        # Create target (sum of solar generation - to be implemented with actual solar data)
        # For now, create synthetic solar based on weather patterns
        if 'weather' in neighborhood_data and 'diffuse_solar_irradiance' in neighborhood_data['weather'].columns:
            solar_target = (neighborhood_data['weather']['diffuse_solar_irradiance'].fillna(0) + 
                           neighborhood_data['weather']['direct_solar_irradiance'].fillna(0)).values
        else:
            # Synthetic solar pattern based on hour
            hours = np.arange(len(all_features)) % 24
            solar_target = np.maximum(0, np.sin((hours - 6) * np.pi / 12)) * 100  # Peak at noon
        
        # Ensure same length
        min_len = min(len(all_features), len(solar_target))
        all_features = all_features[:min_len]
        solar_target = solar_target[:min_len]
        
        # Create sequences
        X, y = [], []
        for i in range(seq_len, len(all_features)):
            X.append(all_features[i-seq_len:i])
            y.append(solar_target[i])
        
        return np.array(X), np.array(y)
    
    def create_neighborhood_sequences(self, buildings_data: Dict[str, pd.DataFrame], target: str, seq_len: int = 24):
        """Crea sequenze per aggregazione neighborhood (3 buildings)."""
        print(f"[NEIGHBORHOOD] Creazione sequenze aggregate per {target}...")
        
        # Carica dati dei 3 buildings
        building_names = ['Building_1', 'Building_2', 'Building_3']
        combined_data = []
        
        for building_name in building_names:
            if building_name in buildings_data:
                data = buildings_data[building_name]
                print(f"  {building_name}: {len(data)} campioni")
                combined_data.append(data)
        
        if not combined_data:
            print("  Errore: nessun dato building trovato per neighborhood")
            return np.array([]), np.array([])
        
        # Prende la lunghezza minima per allineamento temporale
        min_length = min(len(data) for data in combined_data)
        print(f"  Lunghezza minima allineata: {min_length}")
        
        # Crea features aggregate
        aggregated_features = []
        aggregated_targets = []
        
        for i in range(min_length):
            # Caratteristiche temporali dal primo building (sono uguali per tutti)
            time_features = [
                combined_data[0].iloc[i]['hour'],
                combined_data[0].iloc[i]['month']
            ]
            
            # Aggrega features da tutti i buildings
            building_features = []
            building_targets = []
            
            for data in combined_data:
                row = data.iloc[i]
                building_features.extend([
                    row['indoor_dry_bulb_temperature'],
                    row['indoor_relative_humidity'],
                    row['non_shiftable_load'],
                    row['occupant_count']
                ])
                building_targets.append(row[target])
            
            # Combina time features + building features
            all_features = time_features + building_features
            aggregated_features.append(all_features)
            
            # Target aggregato (somma per neighborhood)
            aggregated_targets.append(sum(building_targets))
        
        # Converte in numpy array
        feature_data = np.array(aggregated_features)
        target_data = np.array(aggregated_targets)
        
        # Crea sequenze temporali
        X, y = [], []
        for i in range(seq_len, len(feature_data)):
            # Sequenza di features per predizione
            sequence = feature_data[i-seq_len:i]
            X.append(sequence)
            y.append(target_data[i])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"  Sequenze neighborhood create: X={X.shape}, y={y.shape}")
        print(f"  Target aggregato stats: min={y.min():.2f}, max={y.max():.2f}, media={y.mean():.2f}")
        
        return X, y
    
    def train_model_professional(self, model, X_train, y_train, X_val, y_val, 
                                model_name: str, target: str, epochs: int = 50):
        """Train model with professional configuration and monitoring."""
        print(f"[TRAIN] {model_name} for {target} - {epochs} epochs")
        print(f"  Training sequences: {len(X_train)}")
        print(f"  Validation sequences: {len(X_val)}")
        print(f"  Sequence shape: {X_train.shape}")
        
        # Adjust epochs for different model types
        if 'transformer' in model_name.lower() or 'timesfm' in model_name.lower():
            epochs = min(epochs, 15)  # Reduce epochs for Transformer models to speed up training
        
        if 'baseline' in model_name.lower() or 'random_forest' in model_name.lower() or 'gaussian_process' in model_name.lower() or 'ann' in model_name.lower():
            # Baseline models don't need epochs
            if hasattr(model, 'fit'):
                # Flatten sequences for non-neural models
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_val_flat = X_val.reshape(X_val.shape[0], -1) if X_val is not None else None
                model.fit(X_train_flat, y_train, X_val_flat, y_val if y_val is not None else None)
            else:
                print(f"  {model_name} is not a trainable model")
        else:
            # Neural network training with proper monitoring (LSTM, Transformer, TimesFM)
            print(f"  Starting {model_name} training...")
            print(f"  Data shapes - X: {X_train.shape}, y: {y_train.shape}")
            
            try:
                print(f"  Calling model.fit()...")
                history = model.fit(
                    X_train, y_train, 
                    X_val, y_val, 
                    epochs=epochs, 
                    batch_size=16,  # Reduced batch size
                    verbose=1  # Show progress to debug
                )
                print(f"  Training completed successfully!")
                
                # Store training history
                if history is not None:
                    self.training_histories[f"{model_name}_{target}"] = history
                
            except Exception as e:
                print(f"  Warning: Training failed for {model_name}: {str(e)}")
                print(f"  Attempting simplified training...")
                try:
                    model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=1)
                    print(f"  Simplified training completed!")
                except Exception as e2:
                    print(f"  Error: Could not train {model_name}: {str(e2)}")
                    return None  # Return None for failed models
        
        return model
    
    def evaluate_cross_building(self):
        """Funzione principale di valutazione."""
        print("\n" + "="*80)
        print("NEURAL NETWORK EVALUATION - CITYLEARN 2023")
        print("LSTM, ANN, Gaussian e baseline - cooling_demand + solar_generation")
        print("="*80)
        
        buildings = self.load_complete_dataset()
        # Valuta cooling demand, solar generation e neighborhood aggregation
        targets = ['cooling_demand', 'solar_generation', 'neighborhood_cooling', 'neighborhood_solar']
        
        print(f"\n[TARGETS] {targets}")
        
        results = {}
        
        # Itera su entrambi i target
        for target in targets:
            print(f"\n{'='*60}")
            print(f"EVALUATING TARGET: {target.upper()}")
            print('='*60)
            
            target_results = {}
            
            for model_name, model in self.all_models.items():
                print(f"\n[MODEL] {model_name} per {target}")
                model_results = {}
                
                # Cross-building combinations
                combinations = [
                    ('Building_1', ['Building_2', 'Building_3']),
                    ('Building_2', ['Building_1', 'Building_3']),
                    ('Building_3', ['Building_1', 'Building_2'])
                ]
                
                # Gestione diversa per neighborhood vs single building targets
                if target.startswith('neighborhood_'):
                    # Target neighborhood: usa tutti i buildings aggregati
                    base_target = target.replace('neighborhood_', '')  # 'cooling' -> 'cooling_demand'
                    if base_target == 'cooling':
                        base_target = 'cooling_demand'
                    elif base_target == 'solar':
                        base_target = 'solar_generation'
                    
                    print(f"\n  Neighborhood aggregation per {base_target}")
                    X_full, y_full = self.create_neighborhood_sequences(buildings, base_target)
                    
                    if len(X_full) == 0:
                        print("  Skipping - no neighborhood data")
                        continue
                    
                    # Divisione 80/20 per neighborhood
                    split = int(len(X_full) * 0.8)
                    X_train, X_val = X_full[:split], X_full[split:]
                    y_train, y_val = y_full[:split], y_full[split:]
                    
                    # Crea nuova istanza per neighborhood
                    if model_name in ['Transformer', 'TimesFM']:
                        if model_name == 'Transformer':
                            fresh_model = SimpleTransformerForecaster(
                                sequence_length=24, d_model=64, num_heads=4, num_layers=2,
                                dropout_rate=0.1, learning_rate=0.001
                            )
                        else:  # TimesFM
                            fresh_model = SimpleTimesFMForecaster(
                                sequence_length=24, d_model=128, num_heads=8, num_layers=3,
                                patch_size=4, learning_rate=0.0001
                            )
                    else:
                        fresh_model = model
                    
                    # Addestra su neighborhood aggregato
                    trained_model = self.train_model_professional(
                        fresh_model, X_train, y_train, X_val, y_val, model_name, target
                    )
                    
                    if trained_model is None:
                        print(f"    Skipping neighborhood - training failed")
                        continue
                    
                    # Test su neighborhood (usa validation set)
                    try:
                        predictions = trained_model.predict(X_val)
                        metrics = calculate_metrics(y_val, predictions)
                        model_results['Neighborhood'] = {'Neighborhood': metrics}
                        print(f"    Neighborhood: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
                    except Exception as e:
                        print(f"    Error testing neighborhood: {e}")
                        model_results['Neighborhood'] = {'Neighborhood': {'rmse': 999, 'mae': 999, 'r2': -999}}
                
                else:
                    # Target single building: logica originale cross-building
                    for train_building, test_buildings in combinations:
                        print(f"\n  Train: {train_building} -> Test: {test_buildings}")
                        
                        # Crea sequenze di addestramento
                        train_data = buildings[train_building]
                        X_full, y_full, features = self.create_enhanced_sequences(train_data, target)
                        
                        # Divisione 80/20
                        split = int(len(X_full) * 0.8)
                        X_train, X_val = X_full[:split], X_full[split:]
                        y_train, y_val = y_full[:split], y_full[split:]
                        
                        # Crea nuova istanza del modello per ogni training (importante per Transformer)
                        if model_name in ['Transformer', 'TimesFM']:
                            # Ricreo istanza fresca per evitare conflitti
                            if model_name == 'Transformer':
                                fresh_model = SimpleTransformerForecaster(
                                    sequence_length=24, d_model=64, num_heads=4, num_layers=2,
                                    dropout_rate=0.1, learning_rate=0.001
                                )
                            else:  # TimesFM
                                fresh_model = SimpleTimesFMForecaster(
                                    sequence_length=24, d_model=128, num_heads=8, num_layers=3,
                                    patch_size=4, learning_rate=0.0001
                                )
                        else:
                            fresh_model = model
                        
                        # Addestra modello
                        trained_model = self.train_model_professional(
                            fresh_model, X_train, y_train, X_val, y_val, model_name, target
                        )
                        
                        # Salta se addestramento fallito
                        if trained_model is None:
                            print(f"    Skipping {train_building} - training failed")
                            model_results[train_building] = {}
                            continue
                        
                        # Testa su ogni edificio
                        building_results = {}
                        for test_building in test_buildings:
                            print(f"    Testing on {test_building}...")
                            test_data = buildings[test_building]
                            X_test, y_test, _ = self.create_enhanced_sequences(test_data, target)
                            
                            try:
                                predictions = trained_model.predict(X_test)
                                metrics = calculate_metrics(y_test, predictions)
                                
                                building_results[test_building] = metrics
                                print(f"    {test_building}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
                            except Exception as e:
                                print(f"    Error testing {test_building}: {e}")
                                building_results[test_building] = {'rmse': 999, 'mae': 999, 'r2': -999}
                        
                        model_results[train_building] = building_results
                
                target_results[model_name] = model_results
            
            # Salva risultati per questo target
            self.results[target] = target_results
        
        self.save_results()
        self.generate_visualizations()
        print(f"\n[SAVE] Results and visualizations saved to results/neural_networks/")
    
    def save_results(self):
        """Save neural network results."""
        os.makedirs('results/neural_networks', exist_ok=True)
        
        with open('results/neural_networks/results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Crea tabella risultati
        self._create_results_table()
    
    def _create_results_table(self):
        """Crea tabella comparativa algoritmi (algoritmi su colonne, building/parametri su righe)."""
        try:
            from src.utils.results_table import create_results_table, save_results_table
            table = create_results_table(self.results)
            save_results_table(table)
            print("  Tabella risultati creata!")
        except Exception as e:
            print(f"  Warning: Non è possibile creare tabella risultati: {e}")
    
    def generate_visualizations(self):
        """Generate comparative results table."""
        print("\n[VISUALIZING] Creazione tabella risultati comparativa...")
        
        if not self.results:
            print("  Warning: No results to visualize")
            return
        
        # Crea grafici per tesi
        try:
            from src.utils.data_utils import create_thesis_visualizations
            create_thesis_visualizations(self.results)
            print("  ✓ Grafici per tesi creati!")
        except Exception as e:
            print(f"  Warning: Errore creazione grafici: {e}")
            
        print("  ✓ Tabella algoritmi x building/parametri creata!")
    
    def _prepare_results_for_visualization(self):
        """Prepare results in format expected by visualization utils."""
        simplified = {}
        
        for target in self.results:
            for model_name in self.results[target]:
                if model_name not in simplified:
                    simplified[model_name] = {'mae': [], 'rmse': [], 'r2': []}
                
                # Collect all metrics for this model across buildings
                for train_building in self.results[target][model_name]:
                    for test_building in self.results[target][model_name][train_building]:
                        result = self.results[target][model_name][train_building][test_building]
                        simplified[model_name]['mae'].append(result.get('mae', 0))
                        simplified[model_name]['rmse'].append(result.get('rmse', 0))
                        simplified[model_name]['r2'].append(result.get('r2', 0))
        
        # Average the metrics
        for model_name in simplified:
            for metric in simplified[model_name]:
                if simplified[model_name][metric]:
                    simplified[model_name][metric] = np.mean(simplified[model_name][metric])
                else:
                    simplified[model_name][metric] = 0
        
        return simplified


def main():
    """Main execution."""
    evaluator = ComprehensiveNeuralEvaluator()
    evaluator.evaluate_cross_building()
    
    print("\n" + "="*80)
    print("NEURAL NETWORK EVALUATION COMPLETE")
    print("Results: results/neural_networks/results.json")
    print("="*80)


if __name__ == "__main__":
    main()
