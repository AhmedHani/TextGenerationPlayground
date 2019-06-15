import math
import argparse
import matplotlib.pyplot as plt
from common.batcher import Batcher
from data_loader import NewsDataset, GYAFCEntertainmentFormal
from utils.text_utils import TextEncoder
from common.transformations import TextTransformations
from models.torch_transformer import CustomizedTransformer


parser = argparse.ArgumentParser(description='Encoder-Decoder Text Generation')

parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--embedding_dim', type=int, default=300, help='embedding dimension')
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

gyfac = GYAFCEntertainmentFormal()
info = gyfac.info()

batcher = Batcher(data=gyfac.data_list(), batch_size=batch_size)

transformations = TextTransformations(
    TextTransformations.WordPad(size=info.avg_n_words),
    TextTransformations.WordTruncate(size=info.avg_n_words),
    TextTransformations.AddStartEndTokens()
)

words2index = info.words2index
text_encoder = TextEncoder(vocab2indexes=info.words2index, modelname='word_index')

index2words = info.index2words

model = CustomizedTransformer(input_size=text_encoder.encoding_size(),
                              embedding_dim=embedding_dim,
                              pad_index=2,
                              max_seq_length=info.avg_n_words + 2,
                              device=device)

learning_rate = []

for epoch in range(1, epochs + 1):
    batch_count = 0
    average_epoch_error = 0.0

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

        average_epoch_error += error

        print("Epoch {}/{} \t Batch {}/{} \t TrainLoss {} \t Perplexity {} ".format(
            epoch, epochs, batch_count, batcher.total_train_batches, error, round(math.exp(error), 3)))

    learning_rate.append(average_epoch_error / batcher.total_train_batches)

    print()
    average_valid_error = 0.0
    while batcher.hasnext('valid'):
        batch = batcher.nextbatch('valid')
        batchX, batchY = [item[0] for item in batch], [item[1] for item in batch]

        for transformation in transformations:
            batchX = transformation(batchX)
            batchY = transformation(batchY)

        batchX = text_encoder.encode(batchX)
        batchY = text_encoder.encode(batchY)

        error = model.evaluate_batch(batchX, batchY)
        average_valid_error += error

    average_valid_error /= float(batcher.total_valid_batches)

    print("Average ValidLoss {} \t Average Perplexity {}".format(average_valid_error, round(math.exp(average_valid_error), 3)))

plt.title("Learning Curve using cross entropy cost function")
plt.xlabel("Number of Epochs")
plt.ylabel("Cost")
plt.plot(range(0, len(learning_rate)), learning_rate)
plt.savefig('./learning_rate.png')

#model.save('./saved_models/transformer_model.pt')

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

average_valid_error /= float(batcher.total_valid_batches)
print("Average ValidLoss {} \t Average Perplexity {}".format(average_valid_error, round(math.exp(average_valid_error), 3)))

batch_count = 0

with open('./saved_models/transformer_model_output.txt', 'w') as writer:
    while batcher.hasnext('test'):
        batch_count += 1
        batch = batcher.nextbatch('test')
        batchX, batchY = [item[0] for item in batch], [item[1] for item in batch]
        originalX, originalY = [item[0] for item in batch], [item[1] for item in batch]

        for transformation in transformations:
            batchX = transformation(batchX)
            batchY = transformation(batchY)

        batchX = text_encoder.encode(batchX)
        batchY = text_encoder.encode(batchY)

        output = model.test_batch(batchX, batchY)

        for i in range(0, len(batchY)):
            writer.write("Source: {}\n".format(originalY))
            writer.write("Output: {}\n\n".format(' '.join([info.index2words[index] for index in output[i]])))