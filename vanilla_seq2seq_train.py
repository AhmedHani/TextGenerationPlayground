import argparse
from common.batcher import Batcher
from data_loader import NewsDataset
from utils.text_utils import TextEncoder
from common.transformations import TextTransformations
from models.torch_vanilla_seq2seq import Encoder, Decoder, VanillaSeq2Seq


parser = argparse.ArgumentParser(description='Encoder-Decoder Text Generation')

parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--embedding_dim', type=int, default=128, help='embedding dimension')
parser.add_argument('--hidden_dim', type=int, default=512, help='hidden layer dimension')
parser.add_argument('--n_layers', type=int, default=2, help='number of stacked rnn layers')
parser.add_argument('--no-cuda', action='store_true', default=True, help='disables CUDA training')

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
    TextTransformations.Normalize(),
    TextTransformations.WordPad(size=info.avg_n_words),
    TextTransformations.WordTruncate(size=info.avg_n_words),
    TextTransformations.AddStartEndTokens()
)

text_encoder = TextEncoder(vocab2indexes=info.words2index, modelname='word_index')

encoder = Encoder(input_dim=text_encoder.encoding_size(),
                  emb_dim=embedding_dim,
                  hid_dim=hidden_dim,
                  n_layers=n_layers,
                  dropout=0.5)

decoder = Decoder(output_dim=text_encoder.encoding_size(),
                  emb_dim=embedding_dim,
                  hid_dim=hidden_dim,
                  n_layers=n_layers,
                  dropout=0.5)

model = VanillaSeq2Seq(encoder, decoder, device)

while batcher.hasnext('train'):
    batch = batcher.nextbatch('train')
    batchX, batchY = [item[0] for item in batch], [item[1] for item in batch]

    for transformation in transformations:
        batchX = transformation(batchX)
        batchY = transformation(batchY)

    batchX = text_encoder.encode(batchX)
    batchY = text_encoder.encode(batchY)

    error = model.train_batch(batchX, batchY)
    print(error)

