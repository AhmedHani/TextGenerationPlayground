import math
import argparse
import matplotlib.pyplot as plt
from common.batcher import Batcher
from data_loader import NewsDataset
from utils.text_utils import TextEncoder
from common.transformations import TextTransformations
from models.torch_vanilla_seq2seq import Encoder, Decoder, VanillaSeq2Seq


parser = argparse.ArgumentParser(description='Encoder-Decoder Text Generation')

parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--embedding_dim', type=int, default=128, help='embedding dimension')
parser.add_argument('--hidden_dim', type=int, default=512, help='hidden layer dimension')
parser.add_argument('--n_layers', type=int, default=2, help='number of stacked rnn layers')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs
embedding_dim = args.embedding_dim
hidden_dim = args.hidden_dim
n_layers = args.n_layers
device = 'cpu' if not args.no_cuda is False else 'cuda'

news = NewsDataset()
info = news.info()

batcher = Batcher(data=news.data_list())

transformations = TextTransformations(
    TextTransformations.WordPad(size=info.avg_n_words),
    TextTransformations.WordTruncate(size=info.avg_n_words),
    TextTransformations.AddStartEndTokens()
)

text_encoder = TextEncoder(vocab2indexes=info.words2index, modelname='word_index')

encoder = Encoder(input_dim=text_encoder.encoding_size(),
                  emb_dim=embedding_dim,
                  hid_dim=hidden_dim,
                  n_layers=n_layers,
                  dropout=0.5,
                  device=device)

decoder = Decoder(output_dim=text_encoder.encoding_size(),
                  emb_dim=embedding_dim,
                  hid_dim=hidden_dim,
                  n_layers=n_layers,
                  dropout=0.5,
                  device=device)

model = VanillaSeq2Seq(encoder, decoder, device)

learning_rate = []

for epoch in range(1, epochs + 1):
    epoch_average_error = 0.0
    batch_count = 0

    while batcher.hasnext('train'):
        batch_count += 1
        batch = batcher.nextbatch('train')
        batchX, batchY = [item[0] for item in batch], [item[1] for item in batch]

        for transformation in transformations:
            batchX = transformation(batchX)
            batchY = transformation(batchY)

        batchX = text_encoder.encode(batchX)
        batchY = text_encoder.encode(batchY)

        error = model.train_batch(batchX, batchY)
        epoch_average_error += error

        print("Epoch {}/{} \t Batch {}/{} \t TrainLoss {} \t Perplexity {} ".format(
            epoch, epochs, batch_count, batcher.total_train_batches, error, round(math.exp(error), 3)))

    epoch_average_error /= batcher.total_train_batches

    learning_rate.append(epoch_average_error)

plt.title("Learning Curve using cross entropy cost function")
plt.xlabel("Number of Epochs")
plt.ylabel("Cost")
plt.plot(range(0, len(learning_rate)), learning_rate)
plt.savefig('./learning_rate.png')

average_valid_error = 0.0
batch_count = 0

while batcher.hasnext('valid'):
    batch_count += 1
    batch = batcher.nextbatch('valid')
    batchX, batchY = [item[0] for item in batch], [item[1] for item in batch]

    for transformation in transformations:
        batchX = transformation(batchX)
        batchY = transformation(batchY)

    batchX = text_encoder.encode(batchX)
    batchY = text_encoder.encode(batchY)

    error = model.evaluate_batch(batchX, batchY)
    average_valid_error += error

print("average valid loss: {}".format(average_valid_error / float(batcher.total_valid_batches)))

batch_count = 0


with open('./output.txt', 'w') as writer:
    while batcher.hasnext('test'):
        batch_count += 1
        batch = batcher.nextbatch('test')
        batchX, batchY = [item[0] for item in batch], [item[1] for item in batch]

        for transformation in transformations:
            batchX = transformation(batchX)
            batchY = transformation(batchY)

        batchX = text_encoder.encode(batchX)
        batchY = text_encoder.encode(batchY)

        output = model.test_batch(batchX, batchY)

        for i in range(0, len(batchY)):
            writer.write(str(batchY[i]))
            writer.write("\n")
            writer.write(str(output[i]))
            writer.write("\n\n")

