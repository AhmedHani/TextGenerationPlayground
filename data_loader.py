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


class PaperDataset(object):
    def __init__(self, data_path='./data/paper-news/paper.txt'):
        if data_path.endswith('.txt'):
            self.paper_df = pd.read_fwf(data_path)
            self.paper_df.columns = ['Paper-X']
            self.paper_df['Paper-X'] = self.paper_df['Paper-X'].apply(lambda x: str(x))
            self.paper_df['Paper-Y'] = self.paper_df['Paper-X']
        else:
            self.paper_df = pd.read_csv(data_path)

    def data_list(self):
        return list(zip(self.paper_df['Paper-X'], self.paper_df['Paper-Y']))

    def info(self):
        return TextAnalyzer(data=self.paper_df['Paper-X'].tolist()).all(
                words_freqs=False,
                chars_freqs=False)


class GYFACEntertainmentFormal(object):
    pass


class GYFACEntertainmentInformal(object):
    pass


class GYFACFamilyFormal(object):
    pass


class GYFACFamilyInformal(object):
    pass

