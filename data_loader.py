import pandas as pd
from utils.text_utils import TextAnalyzer


class NewsDataset(object):

    def __init__(self, data_path='./data/paper-news/news.txt'):
        if data_path.endswith('.txt'):
            self.news_df = pd.read_fwf(data_path)
            self.news_df.columns = ['News-X']
            self.news_df['News-X'] = self.news_df['News-X'].apply(lambda x: str(x))
            self.news_df['News-Y'] = self.news_df['News-X']
        else:
            self.news_df = pd.read_csv(data_path)

    def data_list(self):
        return list(zip(self.news_df['News-X'], self.news_df['News-Y']))

    def info(self):
        return TextAnalyzer(data=self.news_df['News-X'].tolist()).all(
                words_freqs=False,
                chars_freqs=False)
