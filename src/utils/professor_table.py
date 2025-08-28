"""
Crea tabella risultati specifica richiesta dal professore.

Dal prompt.txt (riga 19):
"conviene avere una tabella con i vari algoritmi (LSTM, ANN, Gaussian regression, ...) 
sulle colonne, il building e/o il parametro (solar generation, CO2) sulle righe, 
e in corrispondenza di ogni combinazione riporterei la media e deviazione standard 
dell'errore assoluto o RMSE (normalizzati)"
"""

import json
import pandas as pd
import numpy as np

def load_results():
    """Carica risultati dal file JSON."""
    with open('results/neural_networks/results.json', 'r') as f:
        return json.load(f)

def create_professor_table(results=None):
    """Crea la tabella esatta richiesta dal professore."""
    print("=== CREAZIONE TABELLA PROFESSORE ===")
    
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
    for target in ['cooling_demand', 'solar_generation']:
        for building in buildings:
            rows.append(f"{target}_{building}")
    
    # 2. Neighborhood aggregation
    rows.extend(['neighborhood_cooling', 'neighborhood_solar'])
    
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

def save_professor_table(table):
    """Salva tabella in formato CSV e stampa."""
    # Salva CSV
    table.to_csv('results/professor_results_table.csv')
    
    # Stampa tabella
    print("\n" + "="*120)
    print("TABELLA RISULTATI RICHIESTA DAL PROFESSORE")
    print("RMSE Media ± Deviazione Standard")
    print("="*120)
    print()
    print(table.to_string())
    print()
    print("="*120)
    print("Tabella salvata in: results/professor_results_table.csv")
    print("="*120)

def main():
    """Funzione principale."""
    table = create_professor_table()
    save_professor_table(table)
    
    print("\n[INFO] Tabella del professore creata con successo!")

if __name__ == "__main__":
    main()