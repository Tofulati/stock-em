import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedLSTM(nn.Module):
    """
    Enhanced LSTM with:
    - Multi-head attention
    - Residual connections
    - Three-way classification (buy/hold/sell or short)
    """
    def __init__(self, input_size, hidden_size=128, num_layers=3, num_heads=4, dropout=0.3):
        super().__init__()
        
        # Bidirectional LSTM with more capacity
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout,
            bidirectional=True
        )
        
        lstm_output_size = hidden_size * 2  # bidirectional
        
        # Multi-head self-attention
        self.num_heads = num_heads
        self.head_dim = lstm_output_size // num_heads
        assert lstm_output_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.q_linear = nn.Linear(lstm_output_size, lstm_output_size)
        self.k_linear = nn.Linear(lstm_output_size, lstm_output_size)
        self.v_linear = nn.Linear(lstm_output_size, lstm_output_size)
        self.out_linear = nn.Linear(lstm_output_size, lstm_output_size)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(lstm_output_size)
        self.ln2 = nn.LayerNorm(lstm_output_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network with residual
        self.ff1 = nn.Linear(lstm_output_size, hidden_size * 2)
        self.ff2 = nn.Linear(hidden_size * 2, lstm_output_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Output heads
        # 1. Regression head: predict next-day return
        self.return_head = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        # 2. Classification head: buy(1)/hold(0)/sell(-1)
        self.action_head = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3)  # 3 classes
        )
        
        # 3. Confidence head: how confident in the prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def multi_head_attention(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_linear(attn_output)
        
        return output, attn_weights
    
    def forward(self, x, return_attention=False):
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        
        # Multi-head attention with residual
        attn_out, attn_weights = self.multi_head_attention(lstm_out)
        x1 = self.ln1(lstm_out + self.dropout1(attn_out))
        
        # Feed-forward with residual
        ff_out = self.ff2(F.relu(self.ff1(x1)))
        x2 = self.ln2(x1 + self.dropout2(ff_out))
        
        # Use last timestep for prediction
        final_hidden = x2[:, -1, :]  # (batch, hidden*2)
        
        # Multiple output heads
        pred_return = self.return_head(final_hidden).squeeze(-1)
        action_logits = self.action_head(final_hidden)
        confidence = self.confidence_head(final_hidden).squeeze(-1)
        
        if return_attention:
            return pred_return, action_logits, confidence, attn_weights
        
        return pred_return, action_logits, confidence


class CombinedLoss(nn.Module):
    """
    Multi-task loss combining:
    - Return prediction (MSE)
    - Action classification (CrossEntropy)
    - Confidence calibration
    """
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.3):
        super().__init__()
        self.alpha = alpha  # return prediction weight
        self.beta = beta    # action classification weight
        self.gamma = gamma  # confidence weight
        
    def forward(self, pred_return, action_logits, confidence, true_return, true_action):
        # 1. Return prediction loss
        return_loss = F.mse_loss(pred_return, true_return)
        
        # 2. Action classification loss
        action_loss = F.cross_entropy(action_logits, true_action)
        
        # 3. Confidence calibration: confidence should correlate with accuracy
        # Higher confidence when return prediction is accurate
        return_error = torch.abs(pred_return - true_return)
        confidence_loss = F.mse_loss(confidence, torch.exp(-return_error))
        
        total_loss = (
            self.alpha * return_loss + 
            self.beta * action_loss + 
            self.gamma * confidence_loss
        )
        
        return total_loss, {
            'return_loss': return_loss.item(),
            'action_loss': action_loss.item(),
            'confidence_loss': confidence_loss.item()
        }
