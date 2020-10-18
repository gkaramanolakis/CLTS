import pandas as pd
import os
from os.path import expanduser
home = expanduser("~")
import sys
import re
import time
import joblib
from datetime import datetime
import numpy as np
from collections import defaultdict
from Logger import get_logger, close
from util import lang2id
import glob
import json

mldoc_folder = "/home/gkaraman/data2/multilingual/mldoc/preprocessed_data/"
cls_folder = "/home/gkaraman/data2/multilingual/cls/multifit_preprocessed/"
twitter_folder = "/home/gkaraman/data1/data/senti_transfer/labeled_data/"
lorelei_folder = "/home/gkaraman/data2/multilingual/lorelei/preprocessed_data/"

label2ind_dict = {
    'mldoc': {'CCAT': 0, 'ECAT': 1, 'GCAT': 2, 'MCAT': 3},
    'cls': {'NEG': 0, 'POS': 1},
    'twittersent': {'negative': 0, 'neutral': 1, 'positive': 2},
    'lorelei': {'non-med':0, 'med':1}
    }


def load_df_mldoc(method='train', language='english', train_size=1000, print_fn=None):
    if method == 'train':
        fpath = os.path.join(mldoc_folder, language + '.train.{}'.format(train_size))
    elif method == 'dev':
        fpath = os.path.join(mldoc_folder, language + '.dev')
    elif method == 'test':
        fpath = os.path.join(mldoc_folder, language + '.test')
    elif method == 'unlabeled':
        fpath = os.path.join(mldoc_folder, language + '.train.{}'.format(train_size))
    print_fn('Loading {} {} data'.format(method, language))
    df = pd.read_csv(fpath, delimiter='\t', header=None, names=["label", "text"])
    return df

def load_df_cls(method='train', language='english', domain='books', print_fn=None, dev_ratio=0.2):
    if method == 'dev':
        method = 'test'
    dataset_folder = os.path.join(cls_folder, '{}-{}/'.format(lang2id(language), domain))
    if method =='unlabeled':
        method='unsup'
    fpath = os.path.join(dataset_folder, '{}.{}.csv'.format(lang2id(language), method))
    print_fn('loading data: {}'.format(fpath))
    df = pd.read_csv(fpath, header=None, names=["label", "title", "description"])
    df["text"] = df.apply(lambda row: row["title"] + ' . ' + row["description"], axis=1)
    label2ind = label2ind_dict['cls']
    ind2label = {label2ind[a]: a for a in label2ind}
    df["label"] = df['label'].map(lambda x: ind2label.get(x, 'UNK'))
    return df


def load_df_twitter_sent(language='english', method='train', print_fn=None):
    if method == 'unlabeled':
        method = 'train'
    if language == 'chinese' and method == 'dev':
        method = 'test'
    if method == 'test':
        method = 'test_clean'
    if language == 'uyghur' and method == 'train':
        print_fn("USING LORELEI UNLABELED DATA FOR UYGHUR")
        train_path = os.path.join(twitter_folder, "{}/uyghur_unlabeled.csv".format(method))
        df = pd.read_csv(train_path)
        print(df.label.value_counts())
        return df
    language = lang2id(language)
    train_path = os.path.join(twitter_folder, "{}/{}".format(method, language))
    print_fn("loading data: {}".format(train_path))

    with open(train_path, 'r') as f:
        lines = [line for line in f.readlines()]
        texts = [line.strip().split('\t')[0].replace('"','') for line in lines]
        labels = [line.strip().split('\t')[1] for line in lines]
    df = pd.DataFrame()
    df['text'] = texts
    df['label'] = labels
    print_fn(df.label.value_counts())
    return df


def load_df_lorelei(language='english', method='train', print_fn=None, binary=True):
    if method == 'train' and language != 'english':
        method = 'unlabeled'
    fpath = os.path.join(lorelei_folder, "{}_{}.csv".format(language, method))
    print_fn("loading data: {}".format(fpath))
    df = pd.read_csv(fpath)
    df['text'] = df['text'].map(lambda x: str(x))
    if binary:
        df['label'] = df['label'].map(lambda x: 'med' if x=='med' else 'non-med')
    print_fn(df.label.value_counts())
    return df


class DataHandler:
    def __init__(self, dataset, logger=None, domain=None, binary=False):
        self.logger = logger
        self.dataset = dataset
        self.domain = domain
        self.binary = binary

        if 'cls' in dataset:
            self.dataset = dataset.split('_')[0]
            self.domain= dataset.split('_')[1]
        self.label2ind = label2ind_dict[self.dataset]
        self.ind2label = {self.label2ind[a]: a for a in self.label2ind}

    def printstr(self, s):
        if self.logger:
            self.logger.info(s)
        else:
            print(s)

    def load_df(self, method='train', language='english', train_size=1000):
        if self.dataset == 'mldoc':
            return load_df_mldoc(method=method, language=language, train_size=train_size, print_fn=self.printstr)
        if self.dataset == 'cls':
            return load_df_cls(method=method, language=language, domain=self.domain, print_fn=self.printstr)
        elif self.dataset == 'twittersent':
            return load_df_twitter_sent(method=method, language=language, print_fn=self.printstr)
        elif self.dataset == 'lorelei':
            return load_df_lorelei(method=method, language=language, print_fn=self.printstr)
        else:
            raise(Exception('dataset not supported: {}'.format(self.dataset)))