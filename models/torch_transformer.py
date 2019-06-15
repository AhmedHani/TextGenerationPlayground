import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import transformer
from transformer import Transformer


# Resource: https://github.com/phohenecker/pytorch-transformer


class CustomizedTransformer(object):

    def __init__(self, input_size, embedding_dim, pad_index, max_seq_length, device='cpu'):
        self.device = device

        embedding_layer = nn.Embedding(input_size, embedding_dim)
        embedding_layer.reset_parameters()

        self.model = Transformer(embedding_layer, pad_index, input_size, max_seq_len=max_seq_length)
        self.model.to(self.device)

        self.optimizer = optim.Adam((param for param in self.model.parameters() if param.requires_grad), lr=0.0001)
        self.loss = nn.CrossEntropyLoss()

    def train_batch(self, batch_x, batch_y):
        batch_x = Variable(torch.from_numpy(np.asarray(batch_x)).to(self.device))
        batch_y = Variable(torch.from_numpy(np.asarray(batch_y)).to(self.device))

        predictions = self.model(batch_x, batch_y)

        self.optimizer.zero_grad()

        current_loss = self.loss(
            predictions.view(predictions.size(0) * predictions.size(1), predictions.size(2)),
            batch_y.view(-1)
        )
        current_loss.backward()
        self.optimizer.step()

        return current_loss.item()

    def evaluate_batch(self, batch_x, batch_y):
        batch_x = Variable(torch.from_numpy(np.asarray(batch_x)).to(self.device))
        batch_y = Variable(torch.from_numpy(np.asarray(batch_y)).to(self.device))

        predictions = self.model(batch_x, batch_y)

        current_loss = self.loss(
            predictions.view(predictions.size(0) * predictions.size(1), predictions.size(2)),
            batch_y.view(-1)
        )

        return current_loss.item()

    def test_batch(self, batch_x, batch_y):
        batch_x = Variable(torch.from_numpy(np.asarray(batch_x)).to(self.device))
        batch_y = Variable(torch.from_numpy(np.asarray(batch_y)).to(self.device))

        sampled_output = transformer.sample_output(self.model, batch_x, 1, 2, batch_y.size(1))

        return sampled_output.cpu().data.numpy()


