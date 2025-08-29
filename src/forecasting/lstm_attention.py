"""
Modello ibrido LSTM + Self-Attention per forecasting energetico avanzato.

Questo modulo implementa un approccio innovativo che combina:
- LSTM per catturare dipendenze temporali sequenziali
- Self-Attention per identificare time-steps e feature più rilevanti
- Architettura ibrida ottimizzata per serie temporali di media dimensione

Ideale per forecasting solare dove pattern stagionali, meteo e dipendenze
temporali complesse richiedono sia memoria sequenziale che focus selettivo.
"""

import numpy as np
from typing import Tuple, Optional
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from .base_models import BaseForecaster


class LSTMAttentionForecaster(BaseForecaster):
    """
    Modello ibrido LSTM + Self-Attention per forecasting energetico avanzato.
    
    Combina la capacità degli LSTM di catturare dipendenze temporali sequenziali
    con i meccanismi di attenzione per identificare i momenti e le feature più
    rilevanti per la predizione. Questo approccio ibrido è particolarmente
    efficace per serie temporali complesse con pattern stagionali e correlazioni
    cross-feature non ovvie.
    
    Architettura:
    Input → LSTM Encoder → Self-Attention → Global Pooling → Dense → Output
    
    Vantaggi:
    - LSTM cattura le dipendenze temporali a lungo termine
    - Attention identifica automaticamente i time-steps più importanti
    - Interpretabilità attraverso i pesi di attention
    - Performance superiore su dataset di dimensioni medie
    - Combina il meglio di approcci sequenziali e attention-based
    """
    
    def __init__(self, 
                 sequence_length: int = 24,
                 lstm_units: int = 64,
                 attention_units: int = 32,
                 num_heads: int = 4,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Inizializza il modello LSTM+Attention.
        
        Args:
            sequence_length: Lunghezza sequenza temporale
            lstm_units: Unità nascoste LSTM
            attention_units: Dimensioni spazio attention (key_dim)
            num_heads: Numero teste multi-head attention
            dropout_rate: Tasso di dropout per regolarizzazione
            learning_rate: Learning rate per l'ottimizzatore
        """
        super().__init__("LSTM_Attention")
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """Costruisce l'architettura ibrida LSTM+Attention."""
        
        # Input layer
        inputs = Input(shape=input_shape, name="lstm_attention_input")
        
        # LSTM Encoder - cattura dipendenze temporali
        lstm_out = LSTM(self.lstm_units, 
                       return_sequences=True,  # Mantiene dimensione temporale per attention
                       name="lstm_encoder")(inputs)
        lstm_out = Dropout(self.dropout_rate, name="lstm_dropout")(lstm_out)
        
        # Self-Attention Layer - identifica time-steps importanti
        attention_out = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.attention_units,
            name="multi_head_attention"
        )(lstm_out, lstm_out)  # Self-attention: query=key=value=lstm_out
        
        # Skip connection + Layer Normalization (stile Transformer)
        attention_out = LayerNormalization(name="attention_norm")(attention_out + lstm_out)
        attention_out = Dropout(self.dropout_rate, name="attention_dropout")(attention_out)
        
        # Global pooling per aggregare informazioni temporali con attention weights
        pooled = GlobalAveragePooling1D(name="global_pooling")(attention_out)
        
        # Dense layers finali per predizione
        dense1 = Dense(self.attention_units, 
                      activation='relu', 
                      name="dense_1")(pooled)
        dense1 = Dropout(self.dropout_rate, name="dense_dropout")(dense1)
        
        # Output layer
        outputs = Dense(1, activation='linear', name="prediction_output")(dense1)
        
        # Compila il modello
        self.model = Model(inputs=inputs, outputs=outputs, name="LSTM_Attention_Model")
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Print model summary for debugging
        print(f"  LSTM+Attention architecture summary:")
        print(f"    LSTM units: {self.lstm_units}")
        print(f"    Attention heads: {self.num_heads}, key_dim: {self.attention_units}")
        print(f"    Dropout rate: {self.dropout_rate}")
        print(f"    Total parameters: {self.model.count_params()}")
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, verbose=0):
        """
        Addestra il modello LSTM+Attention.
        
        Args:
            X_train: Training features [samples, timesteps, features]
            y_train: Training targets [samples]
            X_val: Validation features (opzionale)
            y_val: Validation targets (opzionale)
            epochs: Numero massimo di epoche
            batch_size: Dimensione batch
            verbose: Verbosità output
        """
        print(f"[LSTM+ATTENTION] Inizio addestramento - {epochs} epoche")
        
        # Costruisci modello se necessario
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self._build_model(input_shape)
            print(f"  Modello ibrido LSTM+Attention costruito: input_shape={input_shape}")
        
        # Callbacks per training ottimale
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss', 
                         patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss', 
                             factor=0.5, patience=7, min_lr=1e-6)
        ]
        
        # Addestramento
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=str(verbose)
        )
        
        print("  Addestramento LSTM+Attention completato!")
        self.is_fitted = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Genera previsioni usando il modello ibrido."""
        assert self.model is not None
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Assicura forma corretta
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        predictions = self.model.predict(X, verbose='0')
        return predictions.flatten() if len(predictions.shape) > 1 else predictions


class LSTMAttentionEnsemble(BaseForecaster):
    """
    Ensemble di modelli LSTM+Attention con configurazioni diverse.
    
    Combina più modelli LSTM+Attention con iperparametri diversi per
    ridurre overfitting e migliorare la robustezza delle predizioni.
    """
    
    def __init__(self, 
                 sequence_length: int = 24,
                 n_models: int = 3,
                 learning_rate: float = 0.001):
        """Inizializza ensemble di modelli LSTM+Attention."""
        super().__init__("LSTM_Attention_Ensemble")
        self.sequence_length = sequence_length
        self.n_models = n_models
        self.learning_rate = learning_rate
        self.models = []
        self.is_fitted = False
        
        # Configurazioni diverse per diversità nell'ensemble
        configs = [
            {"lstm_units": 64, "attention_units": 32, "num_heads": 4, "dropout_rate": 0.2},
            {"lstm_units": 48, "attention_units": 24, "num_heads": 3, "dropout_rate": 0.3},
            {"lstm_units": 80, "attention_units": 40, "num_heads": 2, "dropout_rate": 0.15},
        ]
        
        # Crea modelli con configurazioni diverse
        for i in range(min(n_models, len(configs))):
            config = configs[i]
            model = LSTMAttentionForecaster(
                sequence_length=sequence_length,
                lstm_units=config["lstm_units"],
                attention_units=config["attention_units"],
                num_heads=config["num_heads"],
                dropout_rate=config["dropout_rate"],
                learning_rate=learning_rate
            )
            self.models.append(model)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, verbose=0):
        """Addestra tutti i modelli dell'ensemble."""
        print(f"[LSTM+ATTENTION ENSEMBLE] Addestramento {len(self.models)} modelli...")
        
        for i, model in enumerate(self.models):
            print(f"  Modello {i+1}/{len(self.models)}:")
            model.fit(X_train, y_train, X_val, y_val, epochs, batch_size, verbose)
        
        self.is_fitted = True
        print("  Ensemble training completato!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Genera predizioni ensemble (media delle predizioni)."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Media delle predizioni
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred