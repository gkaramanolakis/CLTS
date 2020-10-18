import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import joblib
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.fr import French
from spacy.lang.it import Italian
from spacy.lang.de import German
from spacy.lang.ru import Russian
from spacy.lang.zh import Chinese
from spacy.lang.ja import Japanese
from spacy.lang.ca import Catalan
from spacy.lang.eu import Basque

from DataHandler import load_df_twitter_sent, load_df_lorelei
from util import clean_str as test_clean_str
from nltk.corpus import stopwords
from util import identity_fn, lang2id


language_dict = {
    'english': English(),
    'spanish': Spanish(),
    'french': French(),
    'italian': Italian(),
    'german': German(),
    'russian': Russian(),
    'chinese': Chinese(),
    'japanese': Japanese(),
    'catalan': Catalan(),
    'basque': Basque(),
}


class Tokenizer:
    def __init__(self, language, tokenizer_method='spacy', remove_stopwords=True, lowercase=True, strip_accents=None,
                 ngram_range=(1,1), min_freq=1, max_freq_perc=1.0, train_list=None, clean_text=True,
                 vocab_loadfolder=None, vocab_savefolder=None):
        self.language = language
        self.tokenizer_method = tokenizer_method
        self.remove_stopwords = remove_stopwords
        self.nlp = language_dict.get(language, None)
        if self.tokenizer_method == 'spacy':
            self.tokenizer_obj = self.nlp.Defaults.create_tokenizer(self.nlp)
            self.tokenize_fn = self.spacy_tokenize
            self.tokenizer = self.spacy_tokenize
        elif self.tokenizer_method == 'clean':
            self.tokenize_fn = self.clean_tokenize
            self.tokenizer = self.clean_tokenize
        else:
            raise(Exception("Tokenizer method not implemented: {}".format(self.tokenizer_method)))
        self.stopwords = self.get_stopwords()
        self.ngram_range = ngram_range
        self.min_freq = min_freq
        self.max_freq_perc = max_freq_perc

        self.vectorizer = None
        self.lowercase = lowercase
        self.clean_text = clean_text
        self.vocab_loadfolder = vocab_loadfolder
        self.vocab_savefolder = vocab_savefolder
        if strip_accents:
            raise(BaseException("strip accents not supported yet."))
        self.strip_accents = strip_accents
        self.oov = None
        if train_list:
            # If there is a non-empty list of documents then train vocabulary
            print("Training {} vocabulary".format(self.language))
            self.create_vocab(train_list, ngram_range=self.ngram_range, min_freq=self.min_freq, max_freq_perc=self.max_freq_perc)
            if self.vocab_savefolder:
                self.save_vocab(self.vocab_savefolder)
        elif self.vocab_loadfolder:
            print("Loading pre-trained {} vocabulary from {}".format(self.language, self.vocab_loadfolder))
            self.load_vocab(self.vocab_loadfolder)
        else:
            raise(BaseException("need to provide training data for the tokenizer..."))

    def get_stopwords(self):
        if self.language in ["chinese", "japanese", "catalan", "basque"]:
            return [lex.text.lower() for lex in self.nlp.vocab if lex.is_stop]
        else:
            try:
                return stopwords.words(self.language)
            except:
                return []

    def spacy_tokenize(self, text):
        # clean text
        text = text.strip()

        if self.lowercase:
            text = text.lower()

        if self.clean_text:
            text = self.clean_str(text)

        # tokenize
        tokens = self.tokenizer_obj(text)

        if self.remove_stopwords:
            tokens = [word for word in tokens if not word.is_stop]

        return [token.text for token in tokens]

    def clean_tokenize(self, text):
        text = text.strip()
        text = text.lower()
        text = self.clean_str(text)
        tokens = text.split()
        return tokens

    def clean_str(self, text):
        """
        Method to pre-process an text for training word embeddings.
        This is post by Sebastian Ruder: https://s3.amazonaws.com/aylien-main/data/multilingual-embeddings/preprocess.py
        and is used at this paper: https://arxiv.org/pdf/1609.02745.pdf
        """

        # replace all numbers with 0
        text = re.sub(r"[-+]?[-/.\d]*[\d]+[:,.\d]*", ' 0 ', text)
        text = re.sub(r'[,:;\.\(\)-/"<>]', " ", text)

        # separate exclamation marks and question marks
        text = re.sub(r"!+", " ! ", text)
        text = re.sub(r"\?+", " ? ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def get_vectors(self, text):
        return self.vectorizer.transform(text)

    def create_vocab(self, text_list, ngram_range=(1,1), min_freq=3, max_freq_perc=0.9):
        self.vectorizer = CountVectorizer(analyzer='word', tokenizer=identity_fn, preprocessor=identity_fn, token_pattern=None,
                                          ngram_range=ngram_range, min_df=min_freq, max_df=max_freq_perc) #strip_accents=self.strip_accents
        tokenized_text = [self.tokenizer(text) for text in text_list]
        self.vectorizer = self.vectorizer.fit(tokenized_text)
        self.word2ind = self.vectorizer.vocabulary_
        self.oov = np.max(list(self.word2ind.values()))
        print("{} vocab size: {}".format(self.language, len(self.word2ind)))
        return

    def load_vocab(self, loadfolder):
         self.vectorizer = joblib.load(os.path.join(loadfolder, "{}_vectorizer.pkl".format(self.language)))
         self.word2ind = joblib.load(os.path.join(loadfolder, "{}_word2ind.pkl".format(self.language)))
         self.oov = np.max(list(self.word2ind.values()))

    def save_vocab(self, savefolder):
        os.makedirs(savefolder, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(savefolder, "{}_vectorizer.pkl".format(self.language)))
        joblib.dump(self.word2ind, os.path.join(savefolder, "{}_word2ind.pkl".format(self.language)))


def train_tokenizer(dataset, savepath, train_size=10000):
    datapath = '../data/'
    #os.makedirs(savepath, exists_ok=True)
    if dataset == 'mldoc':
        train_tokenizer_mldoc(train_size=train_size, datapath=datapath, savepath=savepath)
    elif dataset == 'cls':
        train_tokenizer_cls(datapath=datapath, savepath=savepath)
    elif dataset == 'twittersent':
        train_tokenizer_twitter_sent(datapath=datapath, savepath=savepath)
    elif dataset == 'lorelei':
        train_tokenizer_lorelei(datapath=datapath, savepath=savepath)
    else:
        raise(BaseException('Unknown dataset: {}'.format(dataset)))
    return

def train_tokenizer_mldoc(train_size=1000, datapath='../data/', savepath=None):
    # Training a tokenizer for MLDoc dataset: create a vocabulary for each language in MLDoc
    languages = ["english", "german", "spanish", "french", "italian", "russian", "chinese", "japanese"]
    mldoc_folder = os.path.join(datapath, 'mldoc')
    savepath = os.path.join(mldoc_folder, 'vocab') if savepath is None else savepath
    for lang in languages:
        print("\n\n\t\t *** Training tokenizer for {} ***".format(lang))
        train_f = os.path.join(mldoc_folder, lang + '.train.{}'.format(train_size))
        print('loading data')
        train=pd.read_csv(train_f, delimiter='\t', header=None, names=["label","text"])
        print('data: {}'.format(train.shape))
        tokenizer = Tokenizer(language=lang, train_list=train['text'].tolist(), ngram_range=(1,1), min_freq=1, max_freq_perc=1.0,
                              vocab_savefolder=savepath)
        print("creating new tokenizer")
        tokenizer2 = Tokenizer(language=lang, vocab_loadfolder=savepath)
        print('loaded vocab: {}'.format(len(tokenizer2.word2ind)))

def train_tokenizer_cls(datapath='../data/', savepath=None):
    # Training a tokenizer for CLS dataset: create a vocabulary for each language and domain in CLS
    languages = ["english", "german", "french", "japanese"]
    domains = ['music', 'books', 'dvd']

    cls_folder = os.path.join(datapath, 'cls')
    savepath = os.path.join(cls_folder, 'vocab') if savepath is None else savepath

    for lang in languages:
        for domain in domains:
            print("\n\n\t\t *** Training tokenizer for {}-{} ***".format(lang, domain))
            dataset_folder = os.path.join(cls_folder, '{}-{}/'.format(lang2id(lang), domain))
            print('loading data from {}'.format(dataset_folder))
            fpath = os.path.join(dataset_folder, '{}.train.csv'.format(lang2id(lang)))

            print('loading data')
            train = pd.read_csv(fpath, header=None, names=["label", "title", "description"])
            train["text"] = train.apply(lambda row: row["title"] + ' . ' + row["description"], axis=1)

            print('data: {}'.format(train.shape))
            tokenizer = Tokenizer(language=lang, train_list=train['text'].tolist(), ngram_range=(1,1), min_freq=1, max_freq_perc=1.0,
                                  vocab_savefolder=os.path.join(savepath, domain))
            print("creating new tokenizer")
            tokenizer2 = Tokenizer(language=lang, vocab_loadfolder=os.path.join(savepath, domain))
            print('loaded vocab: {}'.format(len(tokenizer2.word2ind)))


def train_tokenizer_twitter_sent(datapath='../data/', savepath=None):
    languages = ['arabic', 'bulgarian', 'german', 'english', 'spanish', 'persian', 'croatian', 'hungarian',
                 'polish', 'portuguese', 'russian', 'slovak', 'slovenian', 'swedish', 'uyghur', 'chinese']
    savepath = os.path.join(datapath, 'twitter_sent/vocab') if savepath is None else savepath
    for lang in languages:
        print("\n\n\t\t *** Training tokenizer for {} ***".format(lang))
        train = load_df_twitter_sent(method='train', language=lang, print_fn=print)
        print('data: {}'.format(train.shape))
        # NOTE: We use tokenizer_method = 'clean' because data are already tokenized
        tokenizer = Tokenizer(language=lang, train_list=train['text'].tolist(), tokenizer_method='clean', remove_stopwords=False, ngram_range=(1,1), min_freq=5, max_freq_perc=1.0,
                              vocab_savefolder=savepath)
        print("creating new tokenizer")
        tokenizer2 = Tokenizer(language=lang, vocab_loadfolder=savepath, tokenizer_method='clean', remove_stopwords=False)
        print('loaded vocab: {}'.format(len(tokenizer2.word2ind)))
    return

def train_tokenizer_lorelei(datapath='../data/', savepath=None):
    languages = ['english', 'uyghur', 'sinhala']
    savepath = os.path.join(datapath, 'lorelei/vocab') if savepath is None else savepath
    for lang in languages:
        print("\n\n\t\t *** Training tokenizer for {} ***".format(lang))
        train = load_df_lorelei(method='train', language=lang, print_fn=print)
        print('data: {}'.format(train.shape))
        # NOTE: We use tokenizer_method = 'clean' because data are already tokenized
        tokenizer = Tokenizer(language=lang, train_list=train['text'].tolist(), tokenizer_method='clean', remove_stopwords=False, ngram_range=(1,1), min_freq=1, max_freq_perc=1.0,
                              vocab_savefolder=savepath)
        print("creating new tokenizer")
        tokenizer2 = Tokenizer(language=lang, vocab_loadfolder=savepath, tokenizer_method='clean', remove_stopwords=False)
        print('loaded vocab: {}'.format(len(tokenizer2.word2ind)))
    return

if __name__=='__main__':
    datapath='../data/'
    train_tokenizer_cls(datapath=datapath)
    train_tokenizer_mldoc(train_size=10000, datapath=datapath)
    train_tokenizer_twitter_sent(datapath=datapath)
    train_tokenizer_lorelei(datapath=datapath)
