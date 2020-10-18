import pandas as pd
import os
import sys
import re
import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import numpy as np
from util import identity_fn
from scipy.special import softmax


def postprocess_scores(pred):
    if np.sum(pred) == 0:
        return -1
    elif np.sum(pred == np.max(pred)) > 1:
        print("ambiguous: {}".format(pred))
        return -1
    else:
        return np.argmax(pred)
    return


class Teacher:
    def __init__(self, word2ind, aspect2ind, seed_word_dict, use_seed_weights=True, verbose=False, general_ind=None,
                 general_thres=0, teacher_args=None, tokenizer=None, stopwords=[]):
        self.word2ind = word2ind
        self.id2word = {word2ind[w]: w for w in word2ind}
        self.aspect2ind = aspect2ind
        self.ind2aspect = {aspect2ind[a]: a for a in aspect2ind}
        self.verbose = verbose
        self.general_ind = general_ind
        self.args = teacher_args
        self.tokenizer = tokenizer
        self.oov = np.max(list(self.word2ind.values()))
        if not self.tokenizer:
            raise (Exception("Need to define tokenizer for teacher logreg={}".format(self.name)))
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, analyzer='word', tokenizer=identity_fn, preprocessor=identity_fn, token_pattern=None,
                                          ngram_range=(1, 1), stop_words=stopwords, vocabulary=self.word2ind)
        self.clf = LogisticRegression()

    def train(self, df, teacher_weight_dict=None, teacher_bias=None, num_classes=2):
        # Extract seed words from training set
        tokenized_text = df['text'].map(lambda x: self.tokenizer(x))
        _ = self.vectorizer.fit_transform(tokenized_text).toarray()
        self.word2ind = self.vectorizer.vocabulary_
        self.seedwords = set(teacher_weight_dict)
        self.clf.classes_ = np.arange(num_classes)
        if num_classes == 2:
            # binary classification: single weight vector
            self.clf.intercept_ = np.zeros(1)
            self.clf.coef_ = np.zeros((1, len(self.word2ind)))
        else:
            # multiclass classification: weight matrix
            self.clf.intercept_ = np.zeros(len(self.aspect2ind))
            self.clf.coef_ = np.zeros( (len(self.aspect2ind), len(self.word2ind)) )

        self.softmax_intercept = softmax(self.clf.intercept_)
        for word, weight in teacher_weight_dict.items():
            if not word in self.word2ind:
                raise(BaseException("Error: {} not in vocab".format(word)))
            self.clf.coef_[:, self.word2ind[word]] = weight

        return

    def postprocess_scores(self, pred):
        if np.abs(pred - self.softmax_intercept).sum() < 1e-06:
            return -1
        elif np.sum(pred == np.max(pred)) > 1:
            return -1
        else:
            return np.argmax(pred)
        return

    def predict(self, df):
        tokenized_text = df['text'].map(lambda x: self.tokenizer(x))
        features = self.vectorizer.transform(tokenized_text).toarray()
        pred = self.clf.predict_proba(features)
        predictions = np.array([self.postprocess_scores(x) for x in pred])
        return predictions