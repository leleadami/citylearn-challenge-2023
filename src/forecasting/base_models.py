"""
Classi base e utilità per modelli di forecasting CityLearn Challenge.

Questo modulo fornisce la classe BaseForecaster fondamentale da cui ereditano
tutti i modelli di previsione. Le implementazioni specifiche sono organizzate in:

- lstm_models.py: Reti neurali ricorrenti LSTM
- neural_models.py: Reti neurali feedforward (ANN, ResNet, ecc.)  
- classical_models.py: ML tradizionale (Random Forest, Linear, GP, ecc.)
- transformer_models.py: Modelli Transformer e basati su attention

La classe BaseForecaster standardizza l'interfaccia per:
1. Addestramento del modello (metodo fit)
2. Generazione previsioni (metodo predict)  
3. Persistenza del modello (metodi save_model/load_model)
4. Utilità di valutazione delle performance
"""

import numpy as np
from typing import Optional, Tuple, Any, Dict
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor


class BaseForecaster(ABC):
    """
    Classe base per tutti i modelli di forecasting nella CityLearn Challenge.
    
    Questa classe astratta definisce l'interfaccia comune che tutti i modelli
    di previsione devono implementare. Fornisce metodi standardizzati per
    addestramento, previsione e persistenza del modello.
    
    La classe supporta diversi approcci di forecasting delle serie temporali:
    - Modelli di deep learning (LSTM, ANN, Transformers)
    - Modelli di machine learning (Random Forest, Gaussian Process)
    - Modelli statistici tradizionali (Linear, Polynomial regression)
    """
    
    def __init__(self, name: str):
        """
        Inizializza il forecaster base.
        
        Args:
            name (str): Nome identificativo del modello di previsione
                       (es. "LSTM", "Random_Forest", "Linear_Regression")
        """
        self.name = name                 # Identificatore per logging/risultati
        self.model = None               # Memorizza il modello ML/DL effettivo
        self.is_fitted = False          # Flag per tracciare se il modello è addestrato
        
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        Addestra il modello di forecasting su dati storici di serie temporali.
        
        Questo metodo deve essere implementato da ogni forecaster specifico
        per gestire il processo di addestramento appropriato per quel tipo di algoritmo.
        
        Args:
            X_train (np.ndarray): Sequenze input di forma (n_samples, sequence_length, n_features)
                                 Ogni campione contiene valori storici per la predizione
            y_train (np.ndarray): Valori target di forma (n_samples, prediction_horizon)
                                 Valori futuri da predire
            **kwargs: Parametri di addestramento aggiuntivi specifici per ogni modello
        
        Raises:
            NotImplementedError: Deve essere implementato dalle sottoclassi
        """
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Genera previsioni per nuove sequenze di input.
        
        Utilizza il modello addestrato per predire valori futuri basati su sequenze input.
        Il modello deve essere addestrato prima di chiamare questo metodo.
        
        Args:
            X (np.ndarray): Sequenze input di forma (n_samples, sequence_length, n_features)
                           Dati storici per cui generare previsioni
        
        Returns:
            np.ndarray: Valori predetti di forma (n_samples, prediction_horizon)
                       Valori futuri previsti dal modello
        
        Raises:
            ValueError: Se il modello non è stato ancora addestrato
            NotImplementedError: Deve essere implementato dalle sottoclassi
        """
        pass
    
    def save_model(self, filepath: str) -> None:
        """
        Salva il modello addestrato su disco per uso futuro.
        
        Questo metodo può essere sovrascritto dalle sottoclassi per implementare
        logica di salvataggio specifica del modello.
        
        Args:
            filepath (str): Percorso dove salvare il modello
        
        Raises:
            NotImplementedError: Deve essere implementato dalle sottoclassi se necessario
        """
        raise NotImplementedError("Le sottoclassi devono implementare il metodo save_model")
    
    def load_model(self, filepath: str) -> None:
        """
        Carica un modello precedentemente addestrato da disco.
        
        Questo metodo può essere sovrascritto dalle sottoclassi per implementare
        logica di caricamento specifica del modello.
        
        Args:
            filepath (str): Percorso al file del modello salvato
        
        Raises:
            NotImplementedError: Deve essere implementato dalle sottoclassi se necessario
        """
        raise NotImplementedError("Le sottoclassi devono implementare il metodo load_model")


# Nota per gli utilizzatori di questo modulo:
# I modelli di forecasting disponibili sono organizzati in file separati:
# - lstm_models.py: LSTMForecaster, BidirectionalLSTMForecaster, ConvLSTMForecaster
# - transformer_models.py: TransformerForecaster, TimesFMForecaster
# - base_models.py: get_baseline_forecasters() per Random Forest, Linear, ecc.
# 
# Oppure usa la factory function: create_forecaster('lstm', hidden_units=64)


def create_forecaster(model_type: str, **kwargs) -> BaseForecaster:
    """
    Factory function per creare modelli di forecasting per nome.
    
    Fornisce un modo conveniente per istanziare modelli di previsione
    senza importare ogni classe individualmente.
    
    Args:
        model_type (str): Nome del modello di forecasting da creare
                         Opzioni: 'lstm', 'ann', 'random_forest', 'linear', 
                                'polynomial', 'gaussian', 'transformer'
        **kwargs: Parametri da passare al costruttore del modello
    
    Returns:
        BaseForecaster: Istanza del modello di forecasting richiesto
    
    Example:
        >>> lstm_model = create_forecaster('lstm', hidden_units=64, num_layers=2)
        >>> rf_model = create_forecaster('random_forest', n_estimators=200)
    """
    model_type = model_type.lower()
    
    try:
        if model_type in ['lstm', 'bidirectional_lstm', 'conv_lstm']:
            from .lstm_models import LSTMForecaster, BidirectionalLSTMForecaster, ConvLSTMForecaster
            if model_type == 'lstm':
                return LSTMForecaster(**kwargs)
            elif model_type == 'bidirectional_lstm':
                return BidirectionalLSTMForecaster(**kwargs)
            elif model_type == 'conv_lstm':
                return ConvLSTMForecaster(**kwargs)
                
        elif model_type in ['transformer', 'timesfm']:
            from .transformer_models import TransformerForecaster, TimesFMForecaster
            if model_type == 'transformer':
                return TransformerForecaster(**kwargs)
            elif model_type == 'timesfm':
                return TimesFMForecaster(**kwargs)
        
        # Tipo di modello non riconosciuto
        raise ValueError(f"Tipo di modello sconosciuto: {model_type}")
            
    except ImportError as e:
        raise ImportError(f"Impossibile importare modello {model_type}. Verificare dipendenze: {e}")
    except Exception as e:
        raise RuntimeError(f"Errore creando modello {model_type}: {e}")


def get_available_models() -> Dict[str, list]:
    """
    Ottiene lista di tutti i modelli di forecasting disponibili organizzati per categoria.
    
    Returns:
        Dict[str, list]: Dizionario che mappa categorie modelli ai modelli disponibili
    """
    return {
        'lstm': ['lstm', 'bidirectional_lstm', 'conv_lstm'],
        'transformer': ['transformer', 'timesfm'],
        'baseline': ['random_forest', 'linear_regression', 'polynomial_regression', 'gaussian_process', 'ann']
    }


class SklearnForecaster(BaseForecaster):
    """Wrapper per modelli sklearn per lavorare con forecasting di serie temporali."""
    
    def __init__(self, name: str, model):
        super().__init__(name)
        self.model = model
        self._needs_flattening = False  # Flag per gestione appiattimento dati
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, **kwargs) -> None:
        """Addestra modello sklearn su sequenze appiattite."""
        # Appiattisce sequenze per modelli sklearn
        X_flat = X_train.reshape(X_train.shape[0], -1)
        y_flat = y_train.flatten() if len(y_train.shape) > 1 else y_train
        
        self.model.fit(X_flat, y_flat)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Genera previsioni usando modello sklearn."""
        if not self.is_fitted:
            raise ValueError(f"Modello {self.name} deve essere addestrato prima della previsione")
            
        # Applica appiattimento se necessario (per compatibilità con fallback LSTM)
        if self._needs_flattening or len(X.shape) > 2:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
            
        predictions = self.model.predict(X_flat)
        
        # Ritorna array 2D per consistenza
        return predictions.reshape(-1, 1) if len(predictions.shape) == 1 else predictions


def get_baseline_forecasters() -> Dict[str, BaseForecaster]:
    """
    Ottiene collezione di modelli baseline per confronto.
    
    Returns:
        Dict[str, BaseForecaster]: Dizionario dei modelli baseline
    """
    return {
        'Random_Forest': SklearnForecaster(
            'Random_Forest',
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        ),
        'Polynomial_Regression': SklearnForecaster(
            'Polynomial_Regression',
            Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ])
        ),
        'Gaussian_Process': SklearnForecaster(
            'Gaussian_Process',
            GaussianProcessRegressor(random_state=42)
        ),
        'ANN': SklearnForecaster(
            'ANN',
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        ),
        'Ensemble_Voting': EnsembleForecaster(
            'Ensemble_Voting',
            ensemble_type='voting'
        ),
        'Ensemble_Stacking': EnsembleForecaster(
            'Ensemble_Stacking', 
            ensemble_type='stacking'
        )
    }


class EnsembleForecaster(BaseForecaster):
    """
    Forecaster ensemble che combina più modelli base.
    Combina Random Forest, Linear Regression e ANN per migliori performance.
    """
    
    def __init__(self, name: str, ensemble_type: str = 'voting'):
        """
        Inizializza forecaster ensemble.
        
        Args:
            name: Nome dell'ensemble
            ensemble_type: 'voting' per media semplice, 'stacking' per meta-learner
        """
        super().__init__(name)
        self.ensemble_type = ensemble_type
        self.base_models = {}
        self.meta_learner = None
        self.weights = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, **kwargs) -> None:
        """Addestra ensemble di modelli base."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.neural_network import MLPRegressor
        
        # Appiattisce sequenze per modelli ML tradizionali
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1) if X_val is not None else None
        
        # Definisce modelli base per ensemble
        self.base_models = {
            'rf': RandomForestRegressor(n_estimators=50, random_state=42),
            'lr': LinearRegression(),
            'ann': MLPRegressor(hidden_layer_sizes=(32,), max_iter=200, random_state=42)
        }
        
        # Addestra modelli base
        base_predictions_train = []
        base_predictions_val = []
        
        for name, model in self.base_models.items():
            model.fit(X_train_flat, y_train)
            
            # Ottiene predizioni per addestramento meta-learner
            train_pred = model.predict(X_train_flat)
            base_predictions_train.append(train_pred)
            
            if X_val_flat is not None:
                val_pred = model.predict(X_val_flat)
                base_predictions_val.append(val_pred)
        
        # Combina predizioni in matrice
        base_predictions_train = np.column_stack(base_predictions_train)
        
        if self.ensemble_type == 'stacking' and X_val_flat is not None:
            # Addestra meta-learner su predizioni di validazione
            base_predictions_val = np.column_stack(base_predictions_val)
            self.meta_learner = LinearRegression()
            self.meta_learner.fit(base_predictions_val, y_val)
        elif self.ensemble_type == 'voting':
            # Pesi uguali semplici (possono essere ottimizzati)
            self.weights = np.ones(len(self.base_models)) / len(self.base_models)
        
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Genera predizioni ensemble."""
        if not self.is_fitted:
            raise ValueError("Il modello deve essere addestrato prima della predizione")
            
        # Appiattisce sequenze
        X_flat = X.reshape(X.shape[0], -1)
        
        # Ottiene predizioni dai modelli base
        base_predictions = []
        for model in self.base_models.values():
            pred = model.predict(X_flat)
            base_predictions.append(pred)
        
        base_predictions = np.column_stack(base_predictions)
        
        if self.ensemble_type == 'stacking' and self.meta_learner is not None:
            # Usa meta-learner
            ensemble_pred = self.meta_learner.predict(base_predictions)
        else:
            # Voting (media pesata)
            ensemble_pred = np.average(base_predictions, axis=1, weights=self.weights)
        
        return ensemble_pred