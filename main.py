# %%
import pandas as pd
from models.models_zoo import AutoEncoderSequence2Sequence


from sklearn.utils import shuffle
import numpy as np

news = pd.read_fwf('./data/paper-news/news.txt')
news.columns = ['News']

paper = pd.read_fwf('./data/paper-news/paper.txt')
paper.columns = ['Paper']

print("number of News samples: {}".format(news['News'].shape[0]))
print("number of Paper samples: {}".format(paper['Paper'].shape[0]))

news['News'] = news['News'].apply(lambda x: str(x))
paper['Paper'] = paper['Paper'].apply(lambda x: str(x))

max_char_length_paper = paper['Paper'].map(len).max()
max_char_length_news = news['News'].map(len).max()

max_word_length_paper = max([len(text.split()) for text in paper['Paper'].tolist()])
max_word_length_news = max([len(text.split()) for text in news['News'].tolist()])

print("[paper] maximum char length: {}".format(max_char_length_paper))
print("[paper] maximum word length: {}".format(max_word_length_paper))

print("[news] maximum char length: {}".format(max_char_length_news))
print("[news] maximum word length: {}".format(max_word_length_news))


# Padding Functions
def char_based_padding(text, size, token='%'):
    while len(text) < size:
        text += token

    return text


def word_based_padding(text, size, token='PAD'):
    text_tokens = text.split()

    while len(text_tokens) < size:
        text_tokens.append(token)

    return " ".join(text_tokens)


## Padding
news['Char Padded'] = news['News'].apply(lambda text: char_based_padding(text, max_char_length_news))
news['Word Padded'] = news['News'].apply(lambda text: word_based_padding(text, max_word_length_news))

paper['Char Padded'] = paper['Paper'].apply(lambda text: char_based_padding(text, max_char_length_paper))
paper['Word Padded'] = paper['Paper'].apply(lambda text: word_based_padding(text, max_word_length_paper))


## Shuffle dframe
paper = shuffle(paper, random_state=20)
news = shuffle(news, random_state=20)


# assign train/valid/testing
train = [1] * int((paper.shape[0] * 0.8) + 2)
valid = [2] * int(paper.shape[0] * 0.1)
test = [3] * int(paper.shape[0] * 0.1)
paper['Train/Valid/Test'] = train + valid + test

train = [1] * int((news.shape[0] * 0.8) + 1)
valid = [2] * int(news.shape[0] * 0.1)
test = [3] * int(news.shape[0] * 0.1)
news['Train/Valid/Test'] = train + valid + test

del train
del valid
del test


paper_word_vocab = set(paper['Word Padded'].str.cat(sep=' ').split())
paper_word2index = {word: i for i, word in enumerate(paper_word_vocab)}
paper_word2index['PAD'] = len(paper_word2index)
paper_word2index['<sos>'] = len(paper_word2index)

news_word_vocab = set(news['Word Padded'].str.cat(sep=' ').split())
news_word2index = {word: i for i, word in enumerate(news_word_vocab)}
news_word2index['PAD'] = len(news_word2index)
news_word2index['<sos>'] = len(news_word2index)


# Paper model
input_dim = len(paper_word_vocab) + 1
embedding_dim = 128
hidden_dim = 512

paper_model = AutoEncoderSequence2Sequence(
    input_dim=input_dim,
    output_dim=input_dim,
    encoder_emb_dim=embedding_dim,
    decoder_emb_dim=embedding_dim,
    hidden_dim=hidden_dim,
    n_layers=2
)


def get_total_batches(dataframe, column_name, batch_size, mode='train'):
    mapping = {'train': 1, 'valid': 2, 'test': 3}
    data = dataframe.loc[dataframe[column_name] == mapping[mode]]

    return len(data) // batch_size


def get_next_batch(dataframe, column_name, batch_size, batch_count, mode='train'):
    mapping = {'train': 1, 'valid': 2, 'test': 3}

    data = dataframe.loc[dataframe[column_name] == mapping[mode]]
    start, end = batch_size * batch_count, (batch_size * batch_count) + batch_size

    if end <= len(data):
        return data[batch_size * batch_count:(batch_size * batch_count) + batch_size]

    return data[batch_size * batch_count:]


def prepare_batch(batch, x_columns, y_columns):
    return batch[x_columns].tolist(), batch[y_columns].tolist()


def encode_batch(batch, word2numeric):
    matrix = []

    for item in batch:
        item_rep = []

        for word in item.split():
            item_rep.append(word2numeric[word])

        matrix.append(item_rep)

    return matrix


n_epochs = 100
batch_size = 32
total_train_batches = get_total_batches(paper, 'Train/Valid/Test', batch_size, mode='train')
total_valid_batches = get_total_batches(paper, 'Train/Valid/Test', batch_size, mode='valid')
total_test_batches = get_total_batches(paper, 'Train/Valid/Test', batch_size, mode='test')

for i in range(0, n_epochs):
    paper_model.train()

    batch_count = 0

    while batch_count < total_train_batches:
        next_batch = get_next_batch(paper, 'Train/Valid/Test', batch_size, batch_count, mode='train')
        x_batch, y_batch = prepare_batch(next_batch, 'Word Padded', 'Word Padded')
        x_batch, y_batch = encode_batch(x_batch, paper_word2index), encode_batch(y_batch, paper_word2index)

        error = paper_model.train_batch(x_batch, y_batch)

        print(error)