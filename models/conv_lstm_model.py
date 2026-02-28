"""
models/conv_lstm_model.py
-------------------------
Conv1D + BiLSTM hybrid for breathing irregularity classification.
Captures local waveform features (CNN) and temporal dynamics (LSTM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMBreathing(nn.Module):
    """
    Architecture:
        Conv feature extractor  →  BiLSTM  →  attention pooling  →  classifier

    Args:
        n_channels : input signal channels
        seq_len    : samples per window
        n_classes  : output classes
        lstm_hidden: BiLSTM hidden units (per direction)
        lstm_layers: number of stacked LSTM layers
        dropout    : dropout rate
    """

    def __init__(self, n_channels=2, seq_len=960, n_classes=3,
                 lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()

        # ── CNN feature extractor ────────────────────────────────────────────
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )
        # After 3× MaxPool(2): seq_len → seq_len // 8
        cnn_out_len = seq_len // 8

        # ── BiLSTM ───────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── self-attention pooling ────────────────────────────────────────────
        lstm_dim = lstm_hidden * 2   # bidirectional
        self.attn = nn.Linear(lstm_dim, 1)

        # ── classifier ───────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x : (batch, n_channels, seq_len)
        returns logits : (batch, n_classes)
        """
        # CNN
        h = self.conv(x)                         # (B, 128, L/8)
        h = h.permute(0, 2, 1)                   # (B, L/8, 128) for LSTM

        # BiLSTM
        h, _ = self.lstm(h)                      # (B, L/8, lstm_dim)

        # Attention pooling
        scores = self.attn(h).squeeze(-1)        # (B, L/8)
        weights = torch.softmax(scores, dim=-1)  # (B, L/8)
        context = (h * weights.unsqueeze(-1)).sum(dim=1)  # (B, lstm_dim)

        context = self.dropout(context)
        return self.classifier(context)          # (B, n_classes)


def build_conv_lstm(n_channels, seq_len, n_classes,
                    lstm_hidden=128, dropout=0.3):
    return ConvLSTMBreathing(n_channels=n_channels,
                             seq_len=seq_len,
                             n_classes=n_classes,
                             lstm_hidden=lstm_hidden,
                             dropout=dropout)


if __name__ == "__main__":
    model = build_conv_lstm(n_channels=2, seq_len=960, n_classes=3)
    dummy = torch.randn(4, 2, 960)
    out   = model(dummy)
    print("ConvLSTM output shape:", out.shape)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
