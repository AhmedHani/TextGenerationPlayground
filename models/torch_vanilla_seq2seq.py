import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import random


# Tutorial: https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, device='cpu'):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        torch.device(device)

        self.to(device)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, device='cpu'):
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

        torch.device(device)

        self.to(device)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        prediction = self.out(output.squeeze(0))

        return prediction, hidden, cell


class VanillaSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

        #self.init_weights()

        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.CrossEntropyLoss()

        self.to(device)
        torch.device(device)

    def train_batch(self, batch_x, batch_y):
        batch_x = Variable(torch.from_numpy(np.asarray(batch_x)).long().t().to(self.device))
        batch_y = Variable(torch.from_numpy(np.asarray(batch_y)).long().t().to(self.device))

        self.train()

        self.optimizer.zero_grad()

        output = self.forward(batch_x, batch_y)

        output = output[1:].view(-1, output.shape[-1])
        batch_y = batch_y[1:].contiguous().view(-1)

        loss = self.criterion(output, batch_y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)

        self.optimizer.step()

        return loss.item()

    def evaluate_batch(self, batch_x, batch_y):
        batch_x = Variable(torch.from_numpy(np.asarray(batch_x)).long().t().to(self.device))
        batch_y = Variable(torch.from_numpy(np.asarray(batch_y)).long().t().to(self.device))

        self.eval()

        with torch.no_grad():
            output = self.forward(batch_x, batch_y)
            output = output[1:].view(-1, output.shape[-1])
            batch_y = batch_y[1:].contiguous().view(-1)

            loss = self.criterion(output, batch_y)

        return loss.item()

    def test_batch(self, batch_x, batch_y):
        batch_x = Variable(torch.from_numpy(np.asarray(batch_x)).long().t().to(self.device))
        batch_y = Variable(torch.from_numpy(np.asarray(batch_y)).long().t().to(self.device))

        self.eval()

        with torch.no_grad():
            output = self.forward_test(batch_x, batch_y)

        return output

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

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

    def forward_test(self, src, trg):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        r = [trg[0, :].data.numpy()]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            top1 = output.max(1)[1]
            r.append(top1.data.numpy())
            input = top1

        return np.transpose(np.asarray(r)).tolist()

    def count_parameters(self, ):
        p = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print('The model has {} trainable parameters'.format(p))

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
