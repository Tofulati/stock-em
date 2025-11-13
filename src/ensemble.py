import numpy as np
import torch
import torch.nn as nn

def combine_scores(rnn_pred, news_score, pe_z, pct_from_high, weights=None):
    """
    Legacy simple weighted combination (kept for backwards compatibility)
    """
    if weights is None:
        weights = dict(r=3.0, n=2.0, pe=1.0, h=-1.0)
    score = (weights['r'] * rnn_pred + 
             weights['n'] * news_score + 
             weights['pe'] * pe_z + 
             weights['h'] * pct_from_high)
    return score


class AdaptiveEnsemble:
    """
    Adaptive ensemble that learns optimal weights from recent performance
    Updates weights based on rolling accuracy window
    """
    def __init__(self, num_models=3, window_size=20):
        self.num_models = num_models
        self.window_size = window_size
        self.weights = np.ones(num_models) / num_models  # Start equal
        self.history = []  # Store (predictions, actual) for each model
        
    def update_weights(self, predictions, actual):
        """
        Update weights based on recent performance
        predictions: list of [model1_pred, model2_pred, model3_pred]
        actual: actual return value
        """
        self.history.append((predictions, actual))
        
        # Keep only recent history
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        if len(self.history) < 5:  # Need minimum history
            return
        
        # Calculate error for each model
        errors = np.zeros(self.num_models)
        for preds, act in self.history:
            for i, pred in enumerate(preds):
                errors[i] += abs(pred - act)
        
        # Convert errors to weights (inverse of error)
        # Add small epsilon to avoid division by zero
        inv_errors = 1.0 / (errors + 1e-6)
        self.weights = inv_errors / inv_errors.sum()
        
    def predict(self, predictions):
        """
        Combine predictions using learned weights
        predictions: list of [model1_pred, model2_pred, model3_pred]
        """
        return np.dot(self.weights, predictions)
    
    def get_weights(self):
        """Return current model weights"""
        return self.weights.copy()


class StackingEnsemble(nn.Module):
    """
    Neural network-based stacking ensemble
    Combines LSTM, GRU, and Transformer with meta-learner
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        # Base Model 1: LSTM (bidirectional)
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Base Model 2: GRU (bidirectional)
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Base Model 3: Transformer
        self.transformer_embed = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Meta-learner combines all base model outputs
        combined_size = hidden_size * 2 * 3  # 3 bidirectional models
        self.meta_learner = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output heads (same structure as AdvancedLSTM)
        self.return_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 3)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Base Model 1: LSTM
        lstm_out, _ = self.lstm(x)
        lstm_feat = lstm_out[:, -1, :]
        
        # Base Model 2: GRU
        gru_out, _ = self.gru(x)
        gru_feat = gru_out[:, -1, :]
        
        # Base Model 3: Transformer
        trans_embed = self.transformer_embed(x)
        trans_out = self.transformer(trans_embed)
        trans_feat = trans_out[:, -1, :]
        
        # Pad transformer to match LSTM/GRU size
        trans_feat = nn.functional.pad(
            trans_feat, 
            (0, lstm_feat.size(1) - trans_feat.size(1))
        )
        
        # Concatenate all features
        combined = torch.cat([lstm_feat, gru_feat, trans_feat], dim=1)
        
        # Meta-learner processes combined features
        meta_features = self.meta_learner(combined)
        
        # Generate predictions
        pred_return = self.return_head(meta_features).squeeze(-1)
        action_logits = self.action_head(meta_features)
        confidence = self.confidence_head(meta_features).squeeze(-1)
        
        return pred_return, action_logits, confidence