import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.device = device

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)  # [1, 1, hidden]
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class SimpleDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, device, batch_size=1, num_layers=1, encoder_bidirectional=False):
        super(SimpleDecoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)

        if encoder_bidirectional:
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers*2)
        else:
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device
        self.batch_size = batch_size
        self.num_layers = num_layers

    def forward(self, input, hidden):
        # Your code here #
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)


class EncoderLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        device,
        batch_size=1,
        num_layers=1,
        dropout_rate=0.25,
        is_bidirectional=False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=is_bidirectional,
        )
        self.dropout = nn.Dropout(dropout_rate)

        self.device = device
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional

    def forward(self, input, hidden):
        # input: (1, input_size)
        embedded = self.embedding(input).view(
            1, 1, -1
        )  # [1, hidden] --> [1, 1, hidden]
        output, hidden = self.lstm(embedded, hidden)

        if self.is_bidirectional: # only for GRU decoder
            hidden = hidden[0]
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=0)

        return output, hidden

    def initHidden(self):
        return (
            torch.zeros(
                self.num_layers,
                self.batch_size,
                self.lstm.hidden_size,
                device=self.device,
            ),
            torch.zeros(
                self.num_layers,
                self.batch_size,
                self.lstm.hidden_size,
                device=self.device,
            ),
        )


class DecoderLSTM(nn.Module):
    def __init__(
        self,
        hidden_size,
        output_size,  # vocab size
        device,
        num_layers=1,
        batch_size=1,
        dropout_rate=0.25,
        is_bidirectional=False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=is_bidirectional,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device
        self.batch_size = batch_size
        self.num_layers = num_layers

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(
            1, 1, -1
        )  # [1, hidden] --> [1, 1, hidden]
        output, hidden = self.lstm(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return (
            torch.zeros(
                self.num_layers,
                self.batch_size,
                self.lstm.hidden_size,
                device=self.device,
            ),
            torch.zeros(
                self.num_layers,
                self.batch_size,
                self.lstm.hidden_size,
                device=self.device,
            ),
        )
