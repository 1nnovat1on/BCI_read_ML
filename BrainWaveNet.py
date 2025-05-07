import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """
    Additive attention over the sequence dimension.

    Input:  (B, T, H)       – LSTM/Transformer outputs
    Output: (B, H)          – context vector
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.context = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (B, T, H)
        scores = torch.tanh(self.linear(x))            # (B, T, H)
        scores = torch.matmul(scores, self.context)    # (B, T)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = torch.sum(weights * x, dim=1)         # (B, H)
        return pooled


class BrainWaveNet(nn.Module):
    """
    Spatial‑temporal word‑classification network for single‑channel EEG.
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_filters: int = 32,
        lstm_hidden_size: int = 64,
        num_classes: int = 50,
        dropout_p: float = 0.4
    ):
        super().__init__()

        # ---- Convolutional stack (2 blocks) ----
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        conv_out_channels = num_filters * 2

        # ---- Temporal modelling ----
        self.lstm = nn.LSTM(
            input_size=conv_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        lstm_out_size = lstm_hidden_size * 2  # bidirectional

        # ---- Attention pooling ----
        self.attn = AttentionPool(lstm_out_size)

        # ---- Classification head ----
        self.fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(lstm_out_size, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            logits: (batch, num_classes)
        """
        x = self.conv_block1(x)              # (B, C1, L/2)
        x = self.conv_block2(x)              # (B, C2, L/4)

        # reshape for LSTM: (B, L/4, C2)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)           # (B, L/4, 2*H)

        pooled = self.attn(lstm_out)         # (B, 2*H)

        logits = self.fc(pooled)             # (B, num_classes)
        return logits


if __name__ == "__main__":
    # Example: batch of 8 trials, 1 channel, 512 time steps
    example_input = torch.randn(8, 1, 512)
    model = BrainWaveNet()
    output = model(example_input)
    print("Model output shape:", output.shape)  # (8, 50)
