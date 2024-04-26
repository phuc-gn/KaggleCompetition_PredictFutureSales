import torch.nn as nn


def config_lstm():
    return {
        'input_size': 33,
        'hidden_size': 16,
        'num_layers': 2,
        'dropout': 0.2
    }


def config_gru():
    return {
        'input_size': 33,
        'hidden_size': 16,
        'num_layers': 2,
        'dropout': 0.2
    }


class SalesLSTM(nn.Module):
    """
    LSTM-based model for sales prediction.

    Args:
        input_size (int): The number of expected features in the input x
        hidden_size (int): The number of features in the hidden state h
        num_layers (int): Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a `stacked LSTM`, with the second LSTM taking in outputs of the first LSTM and producing the final results.
        dropout (float): If non-zero, introduces a `Dropout` layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to `dropout`.

    Attributes:
        lstm (nn.LSTM): LSTM layer for sequence modeling
        fc (nn.Linear): Fully connected layer for output prediction

    Methods:
        __call__(x): Forward pass of the model.

    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout = dropout, batch_first = True)
        self.fc = nn.Linear(hidden_size, 1)

    def __call__(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length)

        """
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out.reshape(-1)

  
class SalesGRU(nn.Module):
    """
    A class representing a GRU-based sales prediction model.

    Args:
        input_size (int): The number of expected features in the input x
        hidden_size (int): The number of features in the hidden state h
        num_layers (int): Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two GRUs together to form a `stacked GRU`, with the second GRU taking in outputs of the first GRU and producing the final results.
        dropout (float): If non-zero, introduces a `Dropout` layer on the outputs of each GRU layer except the last layer, with dropout probability equal to `dropout`.

    Attributes:
        gru (nn.GRU): The GRU layer used for sequence modeling
        fc (nn.Linear): The fully connected layer used for prediction

    Methods:
        __call__(x): Forward pass of the model.

    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def __call__(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length).

        """
        out, _ = self.gru(x)
        out = self.fc(out)
        return out.reshape(-1)