import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import time


class AutoEncoderSequence2Sequence(nn.Module):

    def __init__(self, input_dim,
                 output_dim,
                 encoder_emb_dim,
                 decoder_emb_dim,
                 hidden_dim,
                 n_layers,
                 encoder_dropout=0.5,
                 decoder_dropout=0.5,
                 device='cpu'):
        super().__init__()

        self.encoder = self.Encoder(input_dim, encoder_emb_dim, hidden_dim, n_layers, encoder_dropout)
        self.decoder = self.Decoder(output_dim, decoder_emb_dim, hidden_dim, n_layers, decoder_dropout)
        self.device = device

        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.CrossEntropyLoss()

        self.apply(self.init_weights)
        torch.device(device)

        self.to(device)

    def train_batch(self, src, trg, clip=1):
        src, trg = torch.LongTensor(src).t(), torch.LongTensor(trg).t()
        self.optimizer.zero_grad()

        output = self.forward(src, trg)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = self.criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)

        self.optimizer.step()

        return loss.item()

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)

        return outputs

    def init_weights(self, m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    class Encoder(nn.Module):
        def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
            super().__init__()

            self.input_dim = input_dim
            self.emb_dim = emb_dim
            self.hid_dim = hid_dim
            self.n_layers = n_layers
            self.dropout = dropout

            self.embedding = nn.Embedding(input_dim, emb_dim)

            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

            self.dropout = nn.Dropout(dropout)

        def forward(self, src):
            # src = [src sent len, batch size]

            embedded = self.dropout(self.embedding(src))

            # embedded = [src sent len, batch size, emb dim]

            outputs, (hidden, cell) = self.rnn(embedded)

            # outputs = [src sent len, batch size, hid dim * n directions]
            # hidden = [n layers * n directions, batch size, hid dim]
            # cell = [n layers * n directions, batch size, hid dim]

            # outputs are always from the top hidden layer

            return hidden, cell

    class Decoder(nn.Module):
        def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
            super().__init__()

            self.emb_dim = emb_dim
            self.hid_dim = hid_dim
            self.output_dim = output_dim
            self.n_layers = n_layers
            self.dropout = dropout

            self.embedding = nn.Embedding(output_dim, emb_dim)

            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

            self.out = nn.Linear(hid_dim, output_dim)

            self.dropout = nn.Dropout(dropout)

        def forward(self, input, hidden, cell):
            # input = [batch size]
            # hidden = [n layers * n directions, batch size, hid dim]
            # cell = [n layers * n directions, batch size, hid dim]

            # n directions in the decoder will both always be 1, therefore:
            # hidden = [n layers, batch size, hid dim]
            # context = [n layers, batch size, hid dim]

            input = input.unsqueeze(0)

            # input = [1, batch size]

            embedded = self.dropout(self.embedding(input))

            # embedded = [1, batch size, emb dim]

            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

            # output = [sent len, batch size, hid dim * n directions]
            # hidden = [n layers * n directions, batch size, hid dim]
            # cell = [n layers * n directions, batch size, hid dim]

            # sent len and n directions will always be 1 in the decoder, therefore:
            # output = [1, batch size, hid dim]
            # hidden = [n layers, batch size, hid dim]
            # cell = [n layers, batch size, hid dim]

            prediction = self.out(output.squeeze(0))

            # prediction = [batch size, output dim]

            return prediction, hidden, cell
