import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from util import identity_fn
from scipy.special import rel_entr
from numpy import log
from Tokenizer import Tokenizer
from sklearn.svm import l1_min_c
from time import time
from util import lang2id


class SeedwordExtractor:
    def __init__(self, source_language, target_language, num_seeds, label2ind, alignment_method='google_translate', missing_translation_strategy='delete',
                 translation_dict_path=None, tokenizer=None, word2ind=None, logger=None):
        self.source_language = source_language
        self.target_language = target_language
        self.num_seeds = num_seeds  # refers to source language
        self.alignment_method = alignment_method
        self.tokenizer = tokenizer
        self.word2ind = word2ind
        self.logger = logger
        self.label2ind = label2ind
        self.ind2label = {label2ind[a]: a for a in label2ind}
        # if strategy is 'delete' then words without translations are deleted
        # if strategy is 'keep' then words without translations are kept as is (useful for Twitter "emoji" words)
        self.missing_translation_strategy = missing_translation_strategy
        self.translation_dict_path = translation_dict_path
        translation_methods = ['google_translate', 'muse_dict']
        if (self.alignment_method in translation_methods) and (not self.translation_dict_path) and (self.source_language != self.target_language):
            raise(Exception("SeedwordExtractor alignment_method = {} but self.translation_dict_path = empty.".format(self.alignment_method)))

    def printstr(self, s):
        if self.logger:
            self.logger.info(s)
        else:
            print(s)

    def extract_seedwords(self, df, language, tokenizer_method='spacy', seedword_savefolder='./seedwords', min_freq=5, max_freq_perc=0.9,
                          tokenizer=None, stopwords=[], random_state=1000):
        if not tokenizer:
            source_tokenizer = Tokenizer(language=language, tokenizer_method=tokenizer_method, remove_stopwords=True, ngram_range=(1, 1),
                                     min_freq=1, max_freq_perc=1.0, train_list=df['text'].tolist())
            tokenizer = source_tokenizer.tokenizer
            stopwords = source_tokenizer.stopwords

        vectorizer = TfidfVectorizer(sublinear_tf=True, analyzer='word', tokenizer=identity_fn, preprocessor=identity_fn, token_pattern=None,
                                     min_df=2, ngram_range=(1, 1), stop_words=stopwords)
        tokenized_text = df['text'].map(lambda x: tokenizer(x))
        features = vectorizer.fit_transform(tokenized_text)
        labels = df['label'].map(lambda x: self.label2ind[x])

        clf = LogisticRegression(penalty='l1', random_state=random_state, tol=1e-6, max_iter=int(1e6), solver='liblinear', multi_class='ovr')
        # Regularization path: tune C so that we get the #coefficients we ask for.
        best_c = tune_C_regularization_path(clf, features, labels, ind2label=self.ind2label, num_steps=50, num_seeds=self.num_seeds)
        self.printstr("best C: {}".format(best_c))
        clf.set_params(C=best_c)

        self.printstr("SOURCE CLF: #features={}".format(features.shape[1]*len(set(labels))))
        clf.fit(features, labels)

        word2ind = vectorizer.vocabulary_
        ind2word = {i: w for w, i in word2ind.items()}

        # Get scores provided by LogReg
        if len(clf.classes_) == 2:
            # binary classification: classifier is a single vector
            aspect2indices = {
                1: np.argsort(clf.coef_[0])[::-1], # pos class (rank highest->lowest)
                0: np.argsort(clf.coef_[0])        # neg class (rank lowest->highest)
            }
            aspect_score_dict = {
                self.ind2label[0]: [(ind2word[sw_ind], clf.coef_[0][sw_ind]) for sw_ind in aspect2indices[0] if clf.coef_[0][sw_ind] < 0],
                self.ind2label[1]: [(ind2word[sw_ind], clf.coef_[0][sw_ind]) for sw_ind in aspect2indices[1] if clf.coef_[0][sw_ind] > 0]}
            weight_dict = {ind2word[i]: clf.coef_[:, i] for aspect in aspect2indices for i in aspect2indices[aspect][:self.num_seeds]}
        else:
            aspect2indices = {i: np.argsort(clf.coef_[i])[::-1] for i in clf.classes_}
            aspect_score_dict = {self.ind2label[i]: [(ind2word[sw_ind], clf.coef_[i][sw_ind]) for sw_ind in aspect2indices[i] if clf.coef_[i][sw_ind]>0] for i in clf.classes_}
            # Get weights for the top <num_seeds> words per aspect.
            weight_dict = {ind2word[i]: clf.coef_[:, i] for aspect in aspect2indices for i in aspect2indices[aspect][:self.num_seeds]}
        os.makedirs(seedword_savefolder, exist_ok=True)
        save_seedwords(aspect_score_dict, seedword_savefolder, language)
        joblib.dump(clf, os.path.join(seedword_savefolder, '{}_seedword_clf.pkl'.format(language)))
        return weight_dict, clf.intercept_

    def get_target_seedwords(self, source_seedword_dict):
        if source_seedword_dict is None:
            raise(Exception('To extract target seed words, you first need to provide source seed words...'))
        if self.source_language == self.target_language:
            target_seedword_dict = {sw:weight for sw,weight in source_seedword_dict.items() if sw in self.word2ind}
            return target_seedword_dict
        else:
            self.printstr("Loading multilingual seed words...")
            translated_seedwords_dict = self.get_translations(source_seedword_dict)
            tokenized_sw_dict = {}
            missing_tokenizations= []
            for target_sw, target_score in translated_seedwords_dict.items():
                tokenized_sw = self.tokenizer(target_sw)
                if len(tokenized_sw) == 1 and tokenized_sw[0] in self.word2ind:
                    tokenized_sw = tokenized_sw[0]
                    tokenized_sw_dict[tokenized_sw] = target_score
                else:
                    missing_tokenizations.append(target_sw)
            return tokenized_sw_dict

    def get_translations(self, source_seedword_dict):
        self.printstr("Loading translation dict: {}".format(self.translation_dict_path))
        translation_dict = joblib.load(self.translation_dict_path)
        translation_dict = translation_dict[(self.source_language, self.target_language)]

        translated_seedwords_dict = {}
        for source_seedword, source_seed_score in source_seedword_dict.items():
            if not source_seedword in translation_dict:
                continue
            if self.alignment_method == 'google_translate':
                translated_sw = translation_dict[source_seedword]
                translated_seedwords_dict[translated_sw] = source_seed_score
            elif self.alignment_method == 'muse_dict':
                # MUSE dict has potentially more than one translations. Add them all.
                translated_sw_list = translation_dict[source_seedword]
                for translated_sw in translated_sw_list:
                    translated_seedwords_dict[translated_sw] = source_seed_score

        if self.missing_translation_strategy == 'keep':
            for source_seedword, source_seed_score in source_seedword_dict.items():
                if not source_seedword in translation_dict:
                    translated_seedwords_dict[source_seedword] = source_seed_score
        return translated_seedwords_dict

def save_seedwords(aspect_score_dict, savefolder, language):
    print("Saving results to {}".format(savefolder))
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)
    dict_savefile = '{}/{}.pkl'.format(savefolder, language)
    joblib.dump(aspect_score_dict, dict_savefile)
    for aspect, scores in aspect_score_dict.items():
        savefile = '{}/{}_{}.scores.txt'.format(savefolder, language, aspect)
        fout = open(savefile, 'w')
        for term, cla in scores[:100]:
            if (cla > 0) or (cla < 0):
                try:
                    fout.write('{0:.5f} {1}\n'.format(cla, term))
                except:
                    print("[ERROR] Could not write non-ascii character")
        pkl_savefile = '{}/{}_{}.scores.pkl'.format(savefolder, language, aspect)
        joblib.dump(scores, pkl_savefile)
        fout.close()
    return


def tune_C_regularization_path(clf, features, labels, ind2label, num_steps=16, num_seeds=50):
    # select the top <num_seeds> seed words for each aspect
    clf.set_params(warm_start=True)
    min_c = 1  # 10^0
    max_c = 8  # 10^7
    # Regularization path
    # Return the lowest bound for C such that for C in (l1_min_C, infinity) the model is guaranteed not to be empty.
    # cs: a list of possible c values to try
    cs = l1_min_c(features, labels, loss='log') * np.logspace(min_c, max_c, num_steps)
    print("Computing regularization path ...")
    start = time()
    coefs_ = []
    for c in cs:
        clf.set_params(C=c)
        clf.fit(features, labels)
        coefs_.append(clf.coef_.copy())
        pos = np.sum(clf.coef_ > 0, axis=1)
        if len(ind2label) == 2:
            # binary classification
            flags = [pos[0] > num_seeds]
        else:
            # multiclass classification
            flags = [pos[i] > num_seeds for i in ind2label]
        if not False in flags:
            print("This took %0.3fs" % (time() - start))
            return c
    print("This took %0.3fs" % (time() - start))
    return c

