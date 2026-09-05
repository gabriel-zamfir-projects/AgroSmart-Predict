import torch
import torch.nn as nn


class SoilPredictorNet(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for predicting soil moisture.
    Takes weather metrics and current moisture as input, outputs next day's moisture.
    """

    def __init__(self, input_size: int):
        super(SoilPredictorNet, self).__init__()

        # We build a fully connected network with dropout to prevent overfitting
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)  # Output layer: 1 continuous value (predicted soil moisture %)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)