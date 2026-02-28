"""
models/cnn_model.py
-------------------
1D CNN for breathing irregularity classification.
Accepts raw or filtered signal windows as input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DBlock(nn.Module):
    """Conv → BN → ReLU → Dropout block."""
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, dropout=0.25):
        super().__init__()
        self.conv  = nn.Conv1d(in_ch, out_ch, kernel_size,
                               stride=stride, padding=kernel_size // 2)
        self.bn    = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(F.relu(self.bn(self.conv(x))))


class BreathingCNN(nn.Module):
    """
    1D CNN for classifying 30-second breathing windows.

    Args:
        n_channels  : number of input signal channels
                      (2 = nasal + thoracic, 3 = + SpO2 resampled)
        seq_len     : samples per window (default 960 = 30 s × 32 Hz)
        n_classes   : number of output classes
        dropout     : dropout probability in conv blocks
    """

    def __init__(self, n_channels=2, seq_len=960, n_classes=3, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            Conv1DBlock(n_channels, 32,  kernel_size=9,  dropout=dropout),
            nn.MaxPool1d(2),                              # → seq/2
            Conv1DBlock(32,         64,  kernel_size=7,  dropout=dropout),
            nn.MaxPool1d(2),                              # → seq/4
            Conv1DBlock(64,        128,  kernel_size=5,  dropout=dropout),
            nn.MaxPool1d(2),                              # → seq/8
            Conv1DBlock(128,       256,  kernel_size=3,  dropout=dropout),
            nn.AdaptiveAvgPool1d(16),                    # fixed 256×16 regardless of seq_len
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        """
        x : (batch, n_channels, seq_len)
        returns logits : (batch, n_classes)
        """
        h = self.features(x)
        return self.classifier(h)


# ── convenience factory ───────────────────────────────────────────────────────

def build_cnn(n_channels, seq_len, n_classes, dropout=0.3):
    return BreathingCNN(n_channels=n_channels,
                        seq_len=seq_len,
                        n_classes=n_classes,
                        dropout=dropout)


if __name__ == "__main__":
    # quick sanity check
    model = build_cnn(n_channels=2, seq_len=960, n_classes=3)
    dummy = torch.randn(4, 2, 960)
    out   = model(dummy)
    print("CNN output shape:", out.shape)   # (4, 3)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
