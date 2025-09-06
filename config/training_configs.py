"""
Configurazioni di training ottimali per diversi scenari e modelli.
"""

# Configurazioni numero epoche ottimale per diversi scenari di training
EPOCHS_CONFIG = {
    # Training veloce per test e debug
    'quick': {
        'LSTM': 15,
        'LSTM_Attention': 20,
        'Transformer': 12, 
        'TimesFM': 10,
        'ANN': 15,
        'fallback_lstm': 5
    },

    # Training bilanciato per risultati affidabili
    'standard': {
        'LSTM': 50,
        'LSTM_Attention': 60,
        'Transformer': 50,
        'TimesFM': 40, 
        'ANN': 40,
        'fallback_lstm': 10
    },

    # Training esteso per performance ottimali
    'optimal': {
        'LSTM': 80,
        'LSTM_Attention': 100,
        'Transformer': 60,
        'TimesFM': 50,
        'ANN': 60,
        'fallback_lstm': 15
    },
    
    # Training intensivo per ricerca e sperimentazione
    'research': {
        'LSTM': 150,
        'LSTM_Attention': 200,
        'Transformer': 100, 
        'TimesFM': 80,
        'ANN': 120,
        'fallback_lstm': 25
    }
}

# Criteri di convergenza per early stopping
CONVERGENCE_CRITERIA = {
    'LSTM': {
        'patience': 15,
        'min_delta': 0.001,
        'monitor': 'val_loss'
    },
    'LSTM_Attention': {
        'patience': 18,
        'min_delta': 0.0005,
        'monitor': 'val_loss'
    },
    'Transformer': {
        'patience': 8,
        'min_delta': 0.002, 
        'monitor': 'val_loss'
    },
    'TimesFM': {
        'patience': 6,
        'min_delta': 0.001,
        'monitor': 'val_loss'
    },
    'ANN': {
        'patience': 10,
        'min_delta': 0.001,
        'monitor': 'val_loss'
    }
}

def get_optimal_epochs(model_name: str, training_mode: str = 'standard') -> int:
    """
    Restituisce numero ottimale di epoche per un modello specifico.
    
    Args:
        model_name: Nome del modello (LSTM, Transformer, etc.)
        training_mode: Modalità training ('quick', 'standard', 'optimal', 'research')
        
    Returns:
        Numero epoche ottimale per il modello e modalità specificati
    """
    if training_mode not in EPOCHS_CONFIG:
        training_mode = 'standard'
    
    # Modelli non-neurali (Random_Forest, etc.) non richiedono epoche
    if model_name not in EPOCHS_CONFIG[training_mode]:
        return 1
        
    return EPOCHS_CONFIG[training_mode].get(model_name, EPOCHS_CONFIG['standard'][model_name])

def get_convergence_config(model_name: str) -> dict:
    """
    Restituisce la configurazione per early stopping.
    
    Args:
        model_name: Nome del modello
        
    Returns:
        Dizionario con configurazione early stopping
    """
    return CONVERGENCE_CRITERIA.get(model_name, CONVERGENCE_CRITERIA['LSTM'])

# Stima tempi di training su hardware medio
TRAINING_TIME_ESTIMATES = {
    'quick': {
        'total_time': '5-10 minuti',
        'LSTM': 2,
        'Transformer': 3, 
        'TimesFM': 2,
        'ANN': 1
    },
    'standard': {
        'total_time': '15-30 minuti',  
        'LSTM': 8,
        'Transformer': 12,
        'TimesFM': 8,
        'ANN': 5
    },
    'optimal': {
        'total_time': '45-90 minuti',
        'LSTM': 20,
        'Transformer': 25, 
        'TimesFM': 15,
        'ANN': 10
    },
    'research': {
        'total_time': '2-4 ore',
        'LSTM': 45,
        'Transformer': 60, 
        'TimesFM': 30,
        'ANN': 35,
        'LSTM_Attention': 80
    }
}

def print_training_summary(training_mode: str = 'standard'):
    """Stampa un riassunto della configurazione di training."""
    config = EPOCHS_CONFIG[training_mode]
    times = TRAINING_TIME_ESTIMATES[training_mode]
    
    print(f"\n{'='*60}")
    print(f"CONFIGURAZIONE TRAINING: {training_mode.upper()}")
    print(f"{'='*60}")
    print(f"Tempo stimato totale: {times['total_time']}")
    print(f"\nEpoche per modello:")
    for model, epochs in config.items():
        if model != 'fallback_lstm':
            time_est = times.get(model, '?')
            print(f"  {model:12}: {epochs:3} epoche (~{time_est} min)")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Test delle configurazioni
    for mode in ['quick', 'standard', 'optimal', 'research']:
        print_training_summary(mode)
        print()