
from simpletransformers.classification import ClassificationModel
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from util import split_to_sentences, identity_fn
import pandas as pd
import shutil
import numpy as np
import joblib
import os

monolingualbert = {
    'english': 'bert-base-cased',
    'spanish': 'dccuchile/bert-base-spanish-wwm-cased',
    'french': 'camembert-base',
    'german': 'bert-base-german-cased',
    'italian': 'dbmdz/bert-base-italian-xxl-cased',
    'russian': 'DeepPavlov/rubert-base-cased',
    'chinese': 'bert-base-chinese',
    'japanese': 'bert-base-japanese'
}

class Student:
    def __init__(self, language, label2ind, student_args, tokenizer=None, manual_seed=1993):
        print("Compiling Student")
        self.name = student_args['model_name']
        self.num_labels = len(label2ind)
        self.label2ind = label2ind
        self.manual_seed = manual_seed
        self.language = language

        if self.name == 'bert':
            bert_name = 'camembert' if language == 'french' else 'bert'
            student_args['model_name'] = bert_name
            if self.language == 'japanese':
                student_args["use_multiprocessing"] = False
            self.clf = ClassificationModel(bert_name, monolingualbert[self.language], num_labels=self.num_labels, args=student_args)
            if self.language == 'japanese':
                from transformers import AutoTokenizer, AutoModel
                japanese_tokenizer = AutoTokenizer.from_pretrained('bert-base-japanese')
                japanese_model = AutoModel.from_pretrained('bert-base-japanese')
                self.clf.tokenizer = japanese_tokenizer
                self.clf.model.bert = japanese_model
                model_to_save = self.clf.model.module if hasattr(self.clf.model, "module") else self.clf.model
                model_to_save.save_pretrained(self.clf.args["output_dir"])
                self.clf.tokenizer.save_pretrained(self.clf.args["output_dir"])
        elif self.name == 'mbert':
            student_args['model_name'] = 'bert'
            self.clf = ClassificationModel('bert', 'bert-base-multilingual-cased', num_labels=self.num_labels, args=student_args)
        elif self.name == 'logreg':
            self.tokenizer = tokenizer
            if not self.tokenizer:
                raise(Exception("Need to define tokenizer for student={}".format(self.name)))
            self.vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.9, norm='l2', ngram_range=(1, 2),
                                              analyzer='word', tokenizer=identity_fn, preprocessor=identity_fn, token_pattern=None)
            self.clf = LogisticRegression(random_state=self.manual_seed, max_iter=int(1e6))

    def postprocess_df(self, df, label_name='label'):
        text = df['text'].map(lambda x: " ".join(split_to_sentences(x)[:2])).tolist()
        labels = df[label_name].map(lambda x: self.label2ind[x]).tolist()
        return pd.DataFrame(list(zip(text, labels)))

    def train(self, df, eval_df=None, label_name='label', eval_label_name='label', drop_sw=None):
        if self.name in ['bert', 'mbert']:
            self.train_bert(df, eval_df, label_name, eval_label_name)
        elif self.name == 'logreg':
            self.train_logreg(df, eval_df, label_name, eval_label_name, drop_sw=drop_sw)
        return

    def eval(self, df, label_name='label'):
        if self.name in ['bert', 'mbert']:
            return self.eval_bert(df, label_name)
        else:
            return self.eval_logreg(df, label_name)

    def train_bert(self, df, eval_df=None, label_name='label', eval_label_name='label'):
        train_df = self.postprocess_df(df, label_name=label_name)
        if eval:
            eval_df = self.postprocess_df(eval_df, label_name=eval_label_name)

        self.clf.train_model(train_df, eval_df=eval_df)

    def train_logreg(self, df, eval_df=None, label_name='label', eval_label_name='label', drop_sw=None):
        tokenized_text = df['text'].map(lambda x: self.tokenizer(x))
        if drop_sw is not None:
            def drop_all_sw(text, seedwords):
                return [x if x not in seedwords else "<UNK>" for x in text]
            print("\n\nSTUDENT: DROPPING {} SEED WORDS".format(len(drop_sw)))
            tokenized_text = tokenized_text.map(lambda x: drop_all_sw(x, drop_sw))
        features = self.vectorizer.fit_transform(tokenized_text).toarray()
        labels = df[label_name].map(lambda x: self.label2ind[x])
        self.clf.fit(features, labels)
        print("logreg features: {}".format(self.clf.coef_.shape[1] * self.clf.coef_.shape[0]))


    def eval_bert(self, df, label_name='label'):
        test_df = self.postprocess_df(df, label_name=label_name)
        result, model_outputs, wrong_predictions = self.clf.eval_model(test_df, acc=sklearn.metrics.accuracy_score)
        pred = np.argmax(model_outputs, axis=1)
        return pred

    def eval_logreg(self, df, label_name='label'):
        tokenized_text = df['text'].map(lambda x: self.tokenizer(x))
        features = self.vectorizer.transform(tokenized_text).toarray()
        print("logreg features: {}".format(features.shape))
        pred = self.clf.predict(features)
        return pred

    def save(self, savefolder):
        joblib.dump(self.clf, os.path.join(savefolder, 'student_clf.pkl'))
        if self.name == 'logreg':
            joblib.dump(self.vectorizer, os.path.join(savefolder, 'student_vectorizer.pkl'))