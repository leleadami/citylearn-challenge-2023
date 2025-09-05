"""
Implementazione semplificata di Transformer per CityLearn Challenge

Questo modulo implementa versioni semplificate ma funzionanti dei modelli Transformer
per evitare problemi complessi con l'architettura originale.
"""

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from .base_models import BaseForecaster


class TransformerForecaster(BaseForecaster):
    """
    Implementazione semplificata di Transformer per forecasting energetico.
    
    Usa MultiHeadAttention nativo di Keras per evitare problemi di compatibilità.
    """
    
    def __init__(self, 
                 sequence_length: int = 24,
                 d_model: int = 32,
                 num_heads: int = 2,
                 num_layers: int = 1,
                 dropout_rate: float = 0.1,
                 learning_rate: float = 0.0001):
        """
        Inizializza Simple Transformer.
        
        Args:
            sequence_length: Lunghezza sequenza input
            d_model: Dimensione modello
            num_heads: Numero teste attenzione
            num_layers: Numero layer transformer
            dropout_rate: Tasso dropout
            learning_rate: Learning rate
        """
        super().__init__("Transformer")
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        
    def _build_model(self, input_shape):
        """Costruisce modello Transformer semplificato con Functional API."""
        seq_len, features = input_shape
        
        # Input layer
        inputs = Input(shape=(seq_len, features))
        
        # Proiezione iniziale con normalizzazione
        x = Dense(self.d_model)(inputs)
        x = LayerNormalization()(x)
        
        # Layer Transformer semplificati (Pre-LN)
        for i in range(self.num_layers):
            # Pre-LN Self-attention
            x_norm = LayerNormalization()(x)
            attention_output = MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                dropout=self.dropout_rate,
                name=f'attention_{i}'
            )(x_norm, x_norm)
            x = x + Dropout(self.dropout_rate)(attention_output)
            
            # Pre-LN Feed-forward network
            x_norm = LayerNormalization()(x)
            ff_output = Dense(self.d_model * 2, activation='relu')(x_norm)
            ff_output = Dropout(self.dropout_rate)(ff_output)
            ff_output = Dense(self.d_model)(ff_output)
            x = x + Dropout(self.dropout_rate)(ff_output)
        
        # Final layer normalization e output
        x = LayerNormalization()(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), # type: ignore
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=60, batch_size=64, verbose=0):
        """Addestra Simple Transformer."""
        print(f"[TRANSFORMER] Inizio addestramento - {epochs} epoche")
        
        # Costruisce modello se necessario
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self._build_model(input_shape)
            print(f"  Modello costruito: input_shape={input_shape}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss', 
                         patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss', 
                             factor=0.3, patience=6)
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
        
        self.is_fitted = True
        print(f"  Addestramento completato!")
        return history
    
    def predict(self, X):
        """Genera predizioni."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Il modello deve essere addestrato prima della predizione")
        
        predictions = self.model.predict(X, verbose='0')
        return predictions.flatten()


class TimesFMForecaster(BaseForecaster):
    """
    Implementazione semplificata ispirata a TimesFM per forecasting.
    
    Usa architettura più semplice ma funzionale per evitare errori complessi.
    """
    
    def __init__(self, 
                 sequence_length: int = 24,
                 d_model: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 patch_size: int = 4,
                 learning_rate: float = 0.0001):
        """
        Inizializza Simple TimesFM.
        
        Args:
            sequence_length: Lunghezza sequenza
            d_model: Dimensione modello
            num_heads: Numero teste attenzione
            num_layers: Numero layer
            patch_size: Dimensione patch (semplificato)
            learning_rate: Learning rate
        """
        super().__init__("TimesFM")
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.learning_rate = learning_rate
        self.model = None
        
    def _build_model(self, input_shape):
        """Costruisce modello TimesFM semplificato con Functional API."""
        seq_len, features = input_shape
        
        # Input layer
        inputs = Input(shape=(seq_len, features))
        
        # Proiezione iniziale con normalizzazione (simula patching semplificato)
        x = Dense(self.d_model)(inputs)
        x = LayerNormalization()(x)
        
        # Stack di layer attention (architettura foundation model style Pre-LN)
        for i in range(self.num_layers):
            # Pre-LN Self-attention
            x_norm = LayerNormalization()(x)
            attention_output = MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                dropout=0.1,
                name=f'timesfm_attention_{i}'
            )(x_norm, x_norm)
            x = x + Dropout(0.1)(attention_output)
            
            # Pre-LN Feed-forward con espansione (stile foundation model)
            x_norm = LayerNormalization()(x)
            ff_output = Dense(self.d_model * 2, activation='gelu')(x_norm)  # Ridotto da 4x a 2x
            ff_output = Dropout(0.1)(ff_output)
            ff_output = Dense(self.d_model)(ff_output)
            x = x + Dropout(0.1)(ff_output)
        
        # Final layer normalization e head di predizione finale
        x = LayerNormalization()(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(self.d_model // 2, activation='gelu')(x)
        x = Dropout(0.1)(x)
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), # type: ignore
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, verbose=0):
        """Addestra Simple TimesFM."""
        print(f"[TIMESFM] Inizio addestramento - {epochs} epoche")
        
        # Costruisce modello
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self._build_model(input_shape)
            print(f"  Modello TimesFM costruito: input_shape={input_shape}")
        
        # Callbacks per foundation model training
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss', 
                         patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss', 
                             factor=0.3, patience=6)  # Più aggressivo per foundation model
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
        
        self.is_fitted = True
        print(f"  Addestramento TimesFM completato!")
        return history
    
    def predict(self, X):
        """Genera predizioni TimesFM."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Il modello deve essere addestrato prima della predizione")
        
        predictions = self.model.predict(X, verbose='0')
        return predictions.flatten()