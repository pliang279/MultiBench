# https://github.com/brentyi/multimodalfilter/blob/master/crossmodal/push_models/lstm.py

import torch
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, units: int = 64):
        super().__init__()

        self.state_dim = 2
        self.lstm_hidden_dim = 512
        self.lstm_num_layers = 2
        self.units = units

        # LSTM layers
        self.lstm = nn.LSTM(units, self.lstm_hidden_dim, self.lstm_num_layers)

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, units),
            nn.ReLU(inplace=True),
            nn.Linear(units, self.state_dim),
        )

    def forward(self, fused_features, training=False):
        lstm_out, _ = self.lstm(fused_features)
        predicted_states = self.output_layers(lstm_out)
        return predicted_states

class GentlePushLateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, x, training=False):
        x, _ = self.lstm(x)
        return x
