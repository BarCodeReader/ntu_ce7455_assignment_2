import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import *
import math

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

class EncoderTF(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderTF, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(d_model=512)
        tflayer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.tfenc = nn.TransformerEncoder(tflayer, num_layers=6)
        self.d_model = 512

        self.device = device

    def forward(self, input, mask):
        #[seq_len, batch_size, embedding_dim]
        embedded = self.embedding(input) * math.sqrt(self.d_model)
        embedded = self.pos_encoder(embedded)
        output = self.tfenc(embedded, mask)
        return output

def generate_square_subsequent_mask(sz):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe = pe

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        self.pe = self.pe.to(x.device)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

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

        return output, hidden

    def initHidden(self):
        return (
            torch.zeros(
                self.num_layers if not self.is_bidirectional else 2,
                self.batch_size,
                self.lstm.hidden_size,
                device=self.device,
            ),
            torch.zeros(
                self.num_layers if not self.is_bidirectional else 2,
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
        self.out = nn.Linear(hidden_size, output_size) if not is_bidirectional else nn.Linear(2*hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)  # [1, hidden] --> [1, 1, hidden]
        embedded = F.relu(embedded)
        output, hidden = self.lstm(embedded, hidden) # output shape [1,1,512]
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return (
            torch.zeros(
                self.num_layers if not self.is_bidirectional else 2,
                self.batch_size,
                self.lstm.hidden_size,
                device=self.device,
            ),
            torch.zeros(
                self.num_layers if not self.is_bidirectional else 2,
                self.batch_size,
                self.lstm.hidden_size,
                device=self.device,
            ),
        )

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # print(attn_weights.shape, encoder_outputs.shape)
        # torch.Size([1, 15]) torch.Size([15, 512])
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        # torch.Size([1, 1, 512]) torch.Size([1, 1, 512]) torch.Size([15, 512])
        # print(attn_applied.shape, embedded.shape, encoder_outputs.shape)
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class TFDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device, dropout_p=0.1, max_length=MAX_LENGTH):
        super(TFDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # here encoder_outputs are already weighted
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # torch.Size([15, 512]) torch.Size([1, 1, 512])
        #print('---', encoder_outputs.shape, embedded.shape)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        #print(attn_weights.shape, encoder_outputs.shape)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)