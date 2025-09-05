"""
Modelli LSTM (Long Short-Term Memory) per CityLearn Challenge

Questo modulo implementa modelli di forecasting serie temporali basati su LSTM
progettati specificamente per la previsione energetica degli edifici. Le reti LSTM
sono particolarmente adatte per il forecasting energetico perché possono catturare:

1. Dipendenze a lungo termine nei pattern di consumo energetico
2. Cicli stagionali e settimanali complessi
3. Relazioni non-lineari tra meteo e domanda energetica
4. Memoria di pattern di consumo passati per predizione futura

Vantaggi Architettura LSTM per Forecasting Energetico:
- Memoria selettiva attraverso gate (forget, input, output)
- Preservazione flusso del gradiente per sequenze lunghe
- Capacità di apprendere pattern temporali complessi
- Robustezza a lunghezze di sequenza variabili e dati mancanti
"""

import numpy as np
from typing import Tuple, Optional
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from .base_models import BaseForecaster




class LSTMForecaster(BaseForecaster):
    """
    Rete neurale LSTM per previsione di serie temporali energetiche degli edifici.
    
    Questa implementazione fornisce un'architettura LSTM flessibile che può adattarsi
    a diversi tipi di edifici, variabili energetiche e orizzonti di previsione.
    Caratteristiche principali:
    - Architettura LSTM multi-layer
    - Regolarizzazione dropout per prevenire overfitting
    - Early stopping e riduzione del learning rate
    - Adattamento automatico della forma di input
    """

    def __init__(self, 
                 sequence_length: int = 24,
                 hidden_units: int = 50,
                 num_layers: int = 2,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Inizializza il previsore LSTM con parametri di architettura.
        
        Args:
            sequence_length: Numero di step temporali storici da usare per la previsione
            hidden_units: Numero di unità LSTM in ogni layer
            num_layers: Numero di layer LSTM impilati (profondità della rete)
            dropout_rate: Frazione di unità da escludere per regolarizzazione
            learning_rate: Learning rate iniziale per l'ottimizzatore
        """
        super().__init__("LSTM")
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Modello costruito dinamicamente in fit() per adattarsi alle dimensioni input
        self.model = None
        
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Costruisce l'architettura della rete neurale LSTM.
        
        Crea una rete LSTM profonda con il numero specificato di layer e
        regolarizzazione. L'architettura è:
        1. Layer di input che accetta sequenze di forma (sequence_length, features)
        2. Layer LSTM multipli con regolarizzazione dropout
        3. Layer denso di output per la previsione
        
        Args:
            input_shape: Forma delle sequenze di input (sequence_length, n_features)
        """
        
        # Inizializza modello sequenziale (layer impilati in ordine)
        self.model = Sequential()
        
        # Primo layer LSTM - deve specificare la forma di input per Keras
        self.model.add(LSTM(
            self.hidden_units,
            return_sequences=True if self.num_layers > 1 else False,
            input_shape=input_shape,
            name=f"lstm_layer_1"
        ))
        self.model.add(Dropout(self.dropout_rate, name="dropout_1"))
        
        # Aggiunge layer LSTM aggiuntivi se specificati
        for i in range(1, self.num_layers):
            return_sequences = i < self.num_layers - 1
            
            self.model.add(LSTM(
                self.hidden_units, 
                return_sequences=return_sequences,
                name=f"lstm_layer_{i+1}"
            ))
            self.model.add(Dropout(self.dropout_rate, name=f"dropout_{i+1}"))
        
        # Layer denso di output per previsione finale
        self.model.add(Dense(1, name="prediction_output"))
        
        # Compila con ottimizzatore Adam, loss MSE e metrica MAE
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), # type: ignore
            loss='mse',
            metrics=['mae']
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: int = 0) -> None:
        """
        Addestra il modello LSTM su dati di serie temporali energetiche degli edifici.
        
        Args:
            X_train: Sequenze di addestramento di forma (n_samples, sequence_length, n_features)
            y_train: Target di addestramento di forma (n_samples, 1)
            X_val: Sequenze di validazione opzionali
            y_val: Target di validazione opzionali
            epochs: Numero massimo di epoche di addestramento
            batch_size: Numero di campioni per batch
            verbose: Livello di verbosità dell'addestramento
        """
        # Garantisce forma input corretta per LSTM (3D: samples, timesteps, features)
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
            
        # Costruisce architettura adattata alla forma degli input
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self._build_model(input_shape)
            
        assert self.model is not None
        
        # Callback per ottimizzazione: early stopping e riduzione learning rate
        callbacks = [
            EarlyStopping(
                monitor='loss',
                patience=15,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=0
            )
        ]
        
        # Prepara dati validazione con forma corretta se forniti
        validation_data = None
        if X_val is not None and y_val is not None:
            if len(X_val.shape) == 2:
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            if len(y_val.shape) == 1:
                y_val = y_val.reshape(-1, 1)
            validation_data = (X_val, y_val)
        
        # Esegue addestramento con callback
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose, # type: ignore
            shuffle=True
        )
        
        self.is_fitted = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Genera previsioni per nuove sequenze di input.
        
        Args:
            X: Sequenze di input di forma (n_samples, sequence_length, n_features)
            
        Returns:
            Previsioni di forma (n_samples, 1)
        """
        assert self.model is not None
        if not self.is_fitted:
            raise ValueError("Il modello deve essere addestrato prima della predizione")
        
        # Garantisce forma input corretta per predizione
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        predictions = self.model.predict(X, verbose='0')
        return predictions.flatten() if len(predictions.shape) > 1 else predictions


class BidirectionalLSTMForecaster(BaseForecaster):
    """
    LSTM bidirezionale per catturare pattern temporali sia in avanti che all'indietro.
    
    Utile per previsioni energetiche quando il contesto futuro (es. previsioni meteo,
    eventi programmati) potrebbe essere disponibile e rilevante per previsioni attuali.
    """
    
    def __init__(self, 
                 sequence_length: int = 24,
                 hidden_units: int = 50,
                 num_layers: int = 2,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """Inizializza forecaster LSTM Bidirezionale."""
        super().__init__("Bidirectional_LSTM")
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """Costruisce architettura LSTM bidirezionale."""
        from keras.layers import Bidirectional
        
        self.model = Sequential()
        
        # Primo layer LSTM bidirezionale
        self.model.add(Bidirectional(
            LSTM(self.hidden_units, 
                 return_sequences=True if self.num_layers > 1 else False),
            input_shape=input_shape,
            name="bidirectional_lstm_1"
        ))
        self.model.add(Dropout(self.dropout_rate))
        
        # Layer bidirezionali aggiuntivi se specificati
        for i in range(1, self.num_layers):
            return_sequences = i < self.num_layers - 1
            self.model.add(Bidirectional(
                LSTM(self.hidden_units, return_sequences=return_sequences),
                name=f"bidirectional_lstm_{i+1}"
            ))
            self.model.add(Dropout(self.dropout_rate))
        
        # Layer denso finale per previsione
        self.model.add(Dense(1, name="prediction_output"))
        
        # Compila con Adam e MSE loss
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), # type: ignore
            loss='mse',
            metrics=['mae']
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: int = 0) -> None:
        """Addestra il modello LSTM bidirezionale."""
        # Implementazione identica all'LSTM unidirezionale
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
            
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self._build_model(input_shape)
            
        assert self.model is not None
        
        callbacks = [
            EarlyStopping(monitor='loss', patience=15, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, min_lr=1e-7, verbose=0)
        ]
        
        validation_data = None
        if X_val is not None and y_val is not None:
            if len(X_val.shape) == 2:
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            if len(y_val.shape) == 1:
                y_val = y_val.reshape(-1, 1)
            validation_data = (X_val, y_val)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=str(verbose),
            shuffle=True
        )
        
        self.is_fitted = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Genera previsioni utilizzando LSTM bidirezionale."""
        assert self.model is not None
        if not self.is_fitted:
            raise ValueError("Il modello deve essere addestrato prima della predizione")
        
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        predictions = self.model.predict(X, verbose='0')
        return predictions.flatten() if len(predictions.shape) > 1 else predictions


class ConvLSTMForecaster(BaseForecaster):
    """
    LSTM Convoluzionale per catturare pattern spazio-temporali nei dati degli edifici.
    
    Utile quando si lavora con edifici multipli o caratteristiche spaziali come
    distribuzioni di temperatura, pattern di irraggiamento solare, ecc.
    """
    
    def __init__(self, 
                 sequence_length: int = 24,
                 filters: int = 32,
                 kernel_size: int = 3,
                 lstm_units: int = 50,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """Inizializza forecaster ConvLSTM."""
        super().__init__("ConvLSTM")
        self.sequence_length = sequence_length
        self.filters = filters
        self.kernel_size = kernel_size
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """Costruisce architettura ibrida CNN-LSTM."""
        from keras.layers import Conv1D, MaxPooling1D, Flatten
        
        self.model = Sequential()
        
        # Layer convoluzionali per estrazione feature spazio-temporali
        self.model.add(Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation='relu',
            input_shape=input_shape,
            name="conv1d_1"
        ))
        self.model.add(MaxPooling1D(pool_size=2, name="maxpool1d_1"))
        self.model.add(Dropout(self.dropout_rate))
        
        # Secondo layer convoluzionale per feature gerarchiche
        self.model.add(Conv1D(
            filters=self.filters//2,
            kernel_size=self.kernel_size,
            activation='relu',
            name="conv1d_2"
        ))
        self.model.add(MaxPooling1D(pool_size=2, name="maxpool1d_2"))
        
        # LSTM per dipendenze temporali su feature estratte
        self.model.add(LSTM(
            self.lstm_units,
            return_sequences=False,
            name="lstm_layer"
        ))
        self.model.add(Dropout(self.dropout_rate))
        
        # Layer denso finale
        self.model.add(Dense(1, name="prediction_output"))
        
        # Compila modello ibrido CNN-LSTM
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), # type: ignore
            loss='mse',
            metrics=['mae']
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: int = 0) -> None:
        """Addestra il modello ConvLSTM."""
        # Implementazione identica alle altre varianti LSTM
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
            
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self._build_model(input_shape)
            
        assert self.model is not None
        
        callbacks = [
            EarlyStopping(monitor='loss', patience=15, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, min_lr=1e-7, verbose=0)
        ]
        
        validation_data = None
        if X_val is not None and y_val is not None:
            if len(X_val.shape) == 2:
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            if len(y_val.shape) == 1:
                y_val = y_val.reshape(-1, 1)
            validation_data = (X_val, y_val)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=str(verbose),
            shuffle=True
        )
        
        self.is_fitted = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Genera previsioni utilizzando ConvLSTM."""
        assert self.model is not None
        if not self.is_fitted:
            raise ValueError("Il modello deve essere addestrato prima della predizione")
        
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        predictions = self.model.predict(X, verbose='0')
        return predictions.flatten() if len(predictions.shape) > 1 else predictions