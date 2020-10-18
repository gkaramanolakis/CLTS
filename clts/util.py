import pandas as pd
import os
import re
import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from scipy.special import rel_entr
import numpy as np
from numpy import log
from collections import defaultdict

def clean_str(text, language='english'):
    """
    Method to pre-process an text for training word embeddings.
    This is post by Sebastian Ruder: https://s3.amazonaws.com/aylien-main/data/multilingual-embeddings/preprocess.py
    and is used at this paper: https://arxiv.org/pdf/1609.02745.pdf
    """
    """
    Cleans an input string and prepares it for tokenization.
    :type text: unicode
    :param text: input text
    :return the cleaned input string
    """
    text = text.lower()

    # replace all numbers with 0
    text = re.sub(r"[-+]?[-/.\d]*[\d]+[:,.\d]*", ' 0 ', text)

    # English-specific pre-processing
    if language == 'english':
        text = re.sub(r"\'s", " \'s", text)
        text = re.sub(r"\'ve", " \'ve", text)
        text = re.sub(r"n\'t", " n\'t", text)
        text = re.sub(r"\'re", " \'re", text)
        text = re.sub(r"\'d", " \'d", text)
        text = re.sub(r"\'ll", " \'ll", text)

    elif language == 'french':
        # French-specific pre-processing
        text = re.sub(r"c\'", " c\' ", text)
        text = re.sub(r"l\'", " l\' ", text)
        text = re.sub(r"j\'", " j\' ", text)
        text = re.sub(r"d\'", " d\' ", text)
        text = re.sub(r"s\'", " s\' ", text)
        text = re.sub(r"n\'", " n\' ", text)
        text = re.sub(r"m\'", " m\' ", text)
        text = re.sub(r"qu\'", " qu\' ", text)

    elif language == 'spanish':
        # Spanish-specific pre-processing
        text = re.sub(r"Â¡", " ", text)

    elif language == 'chinese':
        # Chinese text needs to be segmented before pre-processing
        pass

    # remove commas, colons, semicolons, periods, brackets, hyphens, <, >, and quotation marks
    text = re.sub(r'[,:;\.\(\)-/"<>]', " ", text)

    # separate exclamation marks and question marks
    text = re.sub(r"!+", " ! ", text)
    text = re.sub(r"\?+", " ? ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text, word2ind):
    indices = [word2ind[w] for w in clean_str(text).split() if w in word2ind]
    return indices

def identity_fn(doc):
    return doc

def split_to_sentences(text):
    return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)

def lang2id(lang):
    lang_dict = {
        'english': 'en',
        'spanish': 'es',
        'french': 'fr',
        'italian': 'it',
        'german': 'de',
        'russian': 'ru',
        'japanese': 'ja',
        'chinese': 'zh',
        'catalan': 'ca',
        'basque': 'eu',
        'uyghur': 'ug',
        'sinhalese': 'si',
        'sinhala': 'si',
        'arabic': 'ar',
        'bulgarian': 'bg',
        'persian': 'fa',
        'croatian': 'hr',
        'hungarian': 'hu',
        'polish': 'pl',
        'portuguese': 'pt',
        'slovak': 'sk',
        'slovenian': 'sl',
        'swedish': 'sv'
    }
    return lang_dict[lang]

def report_res(res, savefolder, languages):
    writefile = os.path.join(savefolder, 'results.txt')
    f = open(writefile, 'w')

    def print_and_write(text, n=True):
        if n:
            print(text)
            f.write(text + "\n")
        else:
            print(text, end='')
            f.write(text)

    def print_performance_string(d):
        for lang in languages:
            if not lang in d:
                print_and_write("{:.1f},".format(0), n=False)
                continue
            print_and_write("{:.1f},".format(np.average(d[lang])), n=False)
        print_and_write("")

    avg_teacher_train = defaultdict(list)
    avg_teacher_dev = defaultdict(list)
    avg_teacher_test = defaultdict(list)
    avg_student_train = defaultdict(list)
    avg_student_dev = defaultdict(list)
    avg_student_test = defaultdict(list)

    for run in res:
        for lang in run:
            avg_teacher_train[lang].append(run[lang]['teacher_train_res']['acc'])
            avg_teacher_dev[lang].append(run[lang]['teacher_dev_res']['acc'])
            avg_teacher_test[lang].append(run[lang]['teacher_test_res']['acc'])
            avg_student_dev[lang].append(run[lang]['student_dev_res']['acc'])
            avg_student_test[lang].append(run[lang]['student_test_res']['acc'])
            if 'student_train_res' in run[lang] and 'acc' in run[lang]['student_train_res']:
                avg_student_train[lang].append(run[lang]['student_train_res']['acc'])

    print_and_write("Languages: {}\n".format(languages, n=True))
    print_and_write("\nTeacher Test:\n\t", n=False)
    print_performance_string(avg_teacher_test)

    print_and_write("\nStudent Test:\n\t", n=False)
    print_performance_string(avg_student_test)

    print_and_write("-----------------------------")

    print_and_write("\nTeacher Train:\n\t", n=False)
    print_performance_string(avg_teacher_train)

    print_and_write("\nTeacher Dev:\n\t", n=False)
    print_performance_string(avg_teacher_dev)

    print_and_write("\nStudent Dev:\n\t", n=False)
    print_performance_string(avg_student_dev)

    if len(avg_student_train) > 0:
        print_and_write("\nStudent Train:\n\t", n=False)
        print_performance_string(avg_student_train)
    return

