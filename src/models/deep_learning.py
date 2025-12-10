"""
Deep Learning Models for Stock Prediction

LSTM, CNN, and Transformer neural networks.
These are more advanced than the sklearn models.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# try to import tensorflow (might not be installed)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, Conv1D, MaxPooling1D, 
        Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
        Bidirectional, BatchNormalization
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    keras = None
    print("TensorFlow not installed - deep learning wont work")


class LSTMPredictor:
    """
    LSTM network for stock prediction.
    LSTMs are good for sequential data like time series.
    """
    
    def __init__(self, sequence_length=20, n_features=8, n_classes=3):
        self.sequence_length = sequence_length  # how many days to look back
        self.n_features = n_features  # number of indicators
        self.n_classes = n_classes  # BUY, SELL, HOLD
        self.model = None
        self.scaler_X = None
        
        if TF_AVAILABLE:
            self.model = self._build_model()
    
    def _build_model(self):
        """Create the LSTM architecture."""
        model = Sequential([
            # bidirectional looks at data forwards and backwards
            Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)), 
                         input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.3),  # prevent overfitting
            BatchNormalization(),
            Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001))),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(self.n_classes, activation='softmax')  # output probabilities
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def prepare_sequences(self, df, feature_cols=None):
        """
        Turn the data into sequences for the LSTM.
        We need [samples, timesteps, features] shape.
        """
        if feature_cols is None:
            feature_cols = ['daily_return', 'rsi_14', 'macd', 'macd_histogram', 
                           'bb_percent', 'volatility_20', 'momentum_10', 'stoch_k']
        
        available = [c for c in feature_cols if c in df.columns]
        df = df.copy().dropna()
        
        # scale the features (helps training)
        from sklearn.preprocessing import StandardScaler
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(df[available].values)
        
        # create target: 0=SELL, 1=HOLD, 2=BUY based on 5-day return
        df['future_return'] = df['close'].shift(-5) / df['close'] - 1
        df['signal'] = np.where(df['future_return'] > 0.02, 2,
                       np.where(df['future_return'] < -0.02, 0, 1))
        y = df['signal'].values[:-5]
        X_scaled = X_scaled[:-5]
        
        # create sequences - each sample is 20 days of data
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - self.sequence_length):
            X_seq.append(X_scaled[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length - 1])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, X, y, epochs=100, batch_size=32):
        """Train the LSTM."""
        if not TF_AVAILABLE:
            return {'error': 'TensorFlow not installed'}
        
        # split into train/validation
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # callbacks to help training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=1
        )
        
        # check how we did
        val_metrics = self.model.evaluate(X_val, y_val, verbose=0)
        y_pred = np.argmax(self.model.predict(X_val), axis=1)
        
        from sklearn.metrics import f1_score
        return {
            'model_type': 'lstm',
            'val_loss': float(val_metrics[0]),
            'val_accuracy': float(val_metrics[1]),
            'f1_macro': float(f1_score(y_val, y_pred, average='macro', zero_division=0)),
            'epochs_trained': len(history.history['loss'])
        }
    
    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)


class CNN1DPredictor:
    """
    1D CNN for pattern recognition.
    CNNs can find patterns in the data (like chart patterns).
    """
    
    def __init__(self, sequence_length=20, n_features=8, n_classes=3):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        
        if TF_AVAILABLE:
            self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential([
            # conv layers find patterns
            Conv1D(64, 3, activation='relu', padding='same',
                   input_shape=(self.sequence_length, self.n_features)),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.2),
            Conv1D(128, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.2),
            GlobalAveragePooling1D(),  # flatten for dense layers
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(self.n_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def prepare_sequences(self, df, feature_cols=None):
        # use the same method as LSTM
        lstm = LSTMPredictor(self.sequence_length, self.n_features)
        return lstm.prepare_sequences(df, feature_cols)
    
    def train(self, X, y, epochs=100):
        if not TF_AVAILABLE:
            return {'error': 'TensorFlow not installed'}
        
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
        
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=epochs, batch_size=32, callbacks=callbacks, verbose=1)
        
        val_metrics = self.model.evaluate(X_val, y_val, verbose=0)
        y_pred = np.argmax(self.model.predict(X_val), axis=1)
        
        from sklearn.metrics import f1_score
        return {
            'model_type': 'cnn_1d',
            'val_loss': float(val_metrics[0]),
            'val_accuracy': float(val_metrics[1]),
            'f1_macro': float(f1_score(y_val, y_pred, average='macro', zero_division=0))
        }
    
    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)


class TransformerPredictor:
    """
    Transformer model using self-attention.
    These are what power ChatGPT and other LLMs.
    """
    
    def __init__(self, sequence_length=20, n_features=8, n_classes=3, n_heads=4):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_heads = n_heads
        self.d_model = 64  # embedding dimension
        self.model = None
        
        if TF_AVAILABLE:
            self.model = self._build_model()
    
    def _build_model(self):
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # project features to d_model dimensions
        x = Dense(self.d_model)(inputs)
        
        # add position info (transformers need this)
        positions = tf.range(self.sequence_length)
        pos_embed = keras.layers.Embedding(self.sequence_length, self.d_model)(positions)
        x = x + pos_embed
        
        # transformer blocks - this is where the magic happens
        for _ in range(2):
            x = self._transformer_block(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.n_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def _transformer_block(self, x):
        """One transformer block with attention and feedforward."""
        # self-attention - lets model focus on important parts
        attn = MultiHeadAttention(num_heads=self.n_heads, 
                                   key_dim=self.d_model // self.n_heads)(x, x)
        attn = Dropout(0.1)(attn)
        x = LayerNormalization()(x + attn)  # residual connection
        
        # feedforward network
        ffn = Dense(self.d_model * 4, activation='relu')(x)
        ffn = Dense(self.d_model)(ffn)
        ffn = Dropout(0.1)(ffn)
        return LayerNormalization()(x + ffn)
    
    def train(self, X, y, epochs=100):
        if not TF_AVAILABLE:
            return {'error': 'TensorFlow not installed'}
        
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
        
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=epochs, batch_size=32, callbacks=callbacks, verbose=1)
        
        val_metrics = self.model.evaluate(X_val, y_val, verbose=0)
        y_pred = np.argmax(self.model.predict(X_val), axis=1)
        
        from sklearn.metrics import f1_score
        return {
            'model_type': 'transformer',
            'val_loss': float(val_metrics[0]),
            'val_accuracy': float(val_metrics[1]),
            'f1_macro': float(f1_score(y_val, y_pred, average='macro', zero_division=0))
        }
    
    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)


class HybridCNNLSTM:
    """
    Combines CNN and LSTM together.
    CNN finds patterns, LSTM remembers over time.
    """
    
    def __init__(self, sequence_length=20, n_features=8, n_classes=3):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        
        if TF_AVAILABLE:
            self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential([
            # CNN part - find patterns
            Conv1D(64, 3, activation='relu', padding='same',
                   input_shape=(self.sequence_length, self.n_features)),
            BatchNormalization(),
            MaxPooling1D(2),
            Conv1D(128, 3, activation='relu', padding='same'),
            BatchNormalization(),
            # LSTM part - remember over time
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(self.n_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self, X, y, epochs=100):
        if not TF_AVAILABLE:
            return {'error': 'TensorFlow not installed'}
        
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
        
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=epochs, batch_size=32, callbacks=callbacks, verbose=1)
        
        val_metrics = self.model.evaluate(X_val, y_val, verbose=0)
        y_pred = np.argmax(self.model.predict(X_val), axis=1)
        
        from sklearn.metrics import f1_score
        return {
            'model_type': 'hybrid_cnn_lstm',
            'val_loss': float(val_metrics[0]),
            'val_accuracy': float(val_metrics[1]),
            'f1_macro': float(f1_score(y_val, y_pred, average='macro', zero_division=0))
        }
    
    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)


def train_deep_learning_model(df, model_type='lstm', sequence_length=20, epochs=100):
    """Quick way to train a deep learning model."""
    if not TF_AVAILABLE:
        return None, {'error': 'TensorFlow not installed'}
    
    feature_cols = ['daily_return', 'rsi_14', 'macd', 'macd_histogram', 
                   'bb_percent', 'volatility_20', 'momentum_10', 'stoch_k']
    n_features = len([c for c in feature_cols if c in df.columns])
    
    # pick the model type
    if model_type == 'lstm':
        model = LSTMPredictor(sequence_length, n_features, 3)
    elif model_type == 'cnn':
        model = CNN1DPredictor(sequence_length, n_features, 3)
    elif model_type == 'transformer':
        model = TransformerPredictor(sequence_length, n_features, 3)
    elif model_type == 'hybrid':
        model = HybridCNNLSTM(sequence_length, n_features, 3)
    else:
        raise ValueError(f"unknown model type: {model_type}")
    
    # prepare sequences
    X, y = model.prepare_sequences(df, feature_cols) if hasattr(model, 'prepare_sequences') else \
           LSTMPredictor(sequence_length, n_features).prepare_sequences(df, feature_cols)
    
    metrics = model.train(X, y, epochs=epochs)
    return model, metrics


# test it with data
if __name__ == "__main__":
    print("Testing Deep Learning Models with Real Stock Data...")
    
    if not TF_AVAILABLE:
        print("TensorFlow not installed - skipping")
        exit(1)
    
    # Import data collection and indicators
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.data_collection import StockDataCollector
    from src.indicators import calculate_all_indicators
    
    # Fetch AAPL data
    collector = StockDataCollector()
    df = collector.fetch_stock_data('AAPL', period='5y')
    
    if df.empty:
        print("Failed to fetch data, check internet connection")
        exit(1)
    
    # Calculate all technical indicators
    df = calculate_all_indicators(df)
    print(f"Loaded {len(df)} rows of real AAPL data\n")
    
    # Test LSTM with limited epochs
    print("Training LSTM model...")
    model, metrics = train_deep_learning_model(df, 'lstm', epochs=10)
    print(f"LSTM Validation Accuracy: {metrics.get('val_accuracy', 'N/A'):.4f}")
    print(f"LSTM F1 Macro: {metrics.get('f1_macro', 'N/A'):.4f}")
    
    print("\nDone!")

