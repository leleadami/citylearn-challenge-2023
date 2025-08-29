"""
Crea tabella risultati per analisi comparativa algoritmi.

Formato tabella:
- Algoritmi (LSTM, ANN, Gaussian regression, ecc.) sulle colonne
- Building e parametri (cooling_demand, solar_generation) sulle righe  
- Valori: media e deviazione standard dell'RMSE normalizzato
"""

import json
import pandas as pd
import numpy as np

def load_results():
    """Carica risultati dal file JSON."""
    with open('results/neural_networks/results.json', 'r') as f:
        return json.load(f)

def create_results_table(results=None):
    """Crea tabella comparativa risultati algoritmi."""
    print("=== CREAZIONE TABELLA RISULTATI ===")
    
    # Carica risultati se non forniti
    if results is None:
        results = load_results()
    
    # Algoritmi (colonne)
    algorithms = ['LSTM', 'Transformer', 'TimesFM', 'ANN', 'Random_Forest', 
                  'Polynomial_Regression', 'Gaussian_Process']
    
    # Targets/Building (righe)
    rows = []
    
    # 1. Building-specific targets
    buildings = ['Building_1', 'Building_2', 'Building_3']
    for target in ['solar_generation']:
        for building in buildings:
            rows.append(f"{target}_{building}")
    
    # 2. Global and neighborhood targets (professor requirements)
    # Note: neighborhood_carbon omitted - carbon_intensity is already global
    rows.extend(['carbon_intensity', 'neighborhood_solar'])
    
    # Crea DataFrame vuoto
    table = pd.DataFrame(index=rows, columns=algorithms)
    
    # Riempi tabella con media ± std RMSE
    for target in results:
        for algorithm in results[target]:
            if algorithm not in algorithms:
                continue
                
            rmse_values = []
            
            if target.startswith('neighborhood_'):
                # Neighborhood: singolo valore
                if 'Neighborhood' in results[target][algorithm]:
                    if 'Neighborhood' in results[target][algorithm]['Neighborhood']:
                        rmse = results[target][algorithm]['Neighborhood']['Neighborhood'].get('rmse', None)
                        if rmse and rmse != 999:
                            table.loc[target, algorithm] = f"{rmse:.2f}"
                        else:
                            table.loc[target, algorithm] = "FAIL"
            elif target == 'carbon_intensity':
                # Carbon intensity: global target
                if 'Carbon_Global' in results[target][algorithm]:
                    if 'Carbon_Global' in results[target][algorithm]['Carbon_Global']:
                        rmse = results[target][algorithm]['Carbon_Global']['Carbon_Global'].get('rmse', None)
                        if rmse and rmse != 999:
                            table.loc[target, algorithm] = f"{rmse:.2f}"
                        else:
                            table.loc[target, algorithm] = "FAIL"
            else:
                # Cross-building: raccogli tutti i RMSE
                for train_building in results[target][algorithm]:
                    for test_building in results[target][algorithm][train_building]:
                        rmse = results[target][algorithm][train_building][test_building].get('rmse', None)
                        if rmse and rmse != 999:
                            rmse_values.append(rmse)
                
                if rmse_values:
                    mean_rmse = np.mean(rmse_values)
                    std_rmse = np.std(rmse_values)
                    
                    # Per ogni building specifico
                    for building in buildings:
                        row_name = f"{target}_{building}"
                        if row_name in table.index:
                            table.loc[row_name, algorithm] = f"{mean_rmse:.2f}±{std_rmse:.2f}"
                else:
                    # Fallimento
                    for building in buildings:
                        row_name = f"{target}_{building}"
                        if row_name in table.index:
                            table.loc[row_name, algorithm] = "FAIL"
    
    return table

def save_results_table(table):
    """Salva tabella in formato CSV e stampa."""
    # Salva CSV
    table.to_csv('results/algorithm_comparison_table.csv')
    
    # Stampa tabella
    print("\n" + "="*120)
    print("TABELLA RISULTATI COMPARATIVA ALGORITMI")
    print("RMSE Media ± Deviazione Standard")
    print("="*120)
    print()
    print(table.to_string())
    print()
    print("="*120)
    print("Tabella salvata in: results/algorithm_comparison_table.csv")
    print("="*120)

def main():
    """Funzione principale."""
    table = create_results_table()
    save_results_table(table)
    
    print("\n[INFO] Tabella risultati creata con successo!")

if __name__ == "__main__":
    main()