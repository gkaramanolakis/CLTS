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
from Evaluator import Evaluator
from Teacher import Teacher
from Student import Student
from DataHandler import DataHandler
from Logger import get_logger, close
from Tokenizer import Tokenizer, train_tokenizer
from SeedwordExtractor import SeedwordExtractor
import glob
import json
from simpletransformers.classification import ClassificationModel
from copy import deepcopy
np.random.seed(42)


def cotrain(dataset, source_language, target_language, translation_dict_path, seed_words_path, vocab_path,
            teacher_args, student_args, train_size=10000, num_seeds=30, remove_stopwords=True, use_seed_weights=True,
            manual_seed=0, sw_train_size=10000, logger=None, metric='acc',
            alignment_method='google_translate', tokenizer_method='spacy', num_iter=2):

    def printstr(s):
        if logger:
            logger.info(s)
        else:
            print(s)

    dh = DataHandler(dataset=dataset, logger=logger)
    num_labels = len(dh.label2ind)

    # Load training data in target language
    printstr('Loading train {} data'.format(target_language))
    train_df = dh.load_df(method='unlabeled', language=target_language, train_size=train_size)

    # Load tokenizer for target language
    tokenizer = Tokenizer(language=target_language, tokenizer_method=tokenizer_method, vocab_loadfolder=vocab_path,
                          remove_stopwords=remove_stopwords, ngram_range=(1, 1), min_freq=1, max_freq_perc=1.0)
    word2ind = tokenizer.word2ind


    # Seed Word Extraction
    printstr("Extracting Seed words")
    missing_translation_strategy = 'keep' if dataset == 'twittersent' else 'delete'
    seedword_extractor = SeedwordExtractor(source_language=source_language, target_language=target_language,
                                           num_seeds=num_seeds, label2ind=dh.label2ind, alignment_method=alignment_method,
                                           translation_dict_path=translation_dict_path, missing_translation_strategy=missing_translation_strategy,
                                           tokenizer=tokenizer.tokenizer, word2ind=word2ind, logger=logger)
    source_df = dh.load_df(method='train', language=source_language, train_size=sw_train_size)
    source_tokenizer = Tokenizer(language=source_language, vocab_loadfolder=vocab_path, tokenizer_method=tokenizer_method)
    source_weight_dict, intercept = seedword_extractor.extract_seedwords(df=source_df, language=source_language, seedword_savefolder=seed_words_path,
                                                                         tokenizer=source_tokenizer.tokenizer, tokenizer_method=tokenizer_method,
                                                                         stopwords=source_tokenizer.stopwords)
    seed_word_dict = seedword_extractor.get_target_seedwords(source_seedword_dict=source_weight_dict)

    if len(seed_word_dict) == 0:
        printstr("ERROR: There are 0 target seed words. Skipping experiment")
        return {
            'teacher_args': teacher_args,
            'student_args': student_args,
            'teacher_train_res': defaultdict(int),
            'teacher_dev_res': defaultdict(int),
            'teacher_test_res': defaultdict(int),
            'student_dev_res': defaultdict(int),
            'student_test_res': defaultdict(int),
            'student_train_res': defaultdict(int),
        }

    # Initialize Teacher in Target Language
    # report_seedwords(seed_word_dict, ind2label=dh.ind2label, print_fn=printstr)
    teacher = Teacher(word2ind=word2ind, aspect2ind=dh.label2ind, seed_word_dict=seed_word_dict, verbose=True, teacher_args=teacher_args,
                      tokenizer=tokenizer.tokenizer, stopwords=tokenizer.stopwords)
    teacher.train(df=train_df, teacher_weight_dict=seed_word_dict, teacher_bias=intercept, num_classes=num_labels)

    # Initialize Student in Target Language:
    if num_iter > 1:
        # For the first iteration, use LogReg
        actual_student_name = student_args['model_name']
        student_args['model_name'] = 'logreg'
    student = Student(language=target_language, label2ind=dh.label2ind, student_args=student_args, tokenizer=tokenizer.tokenizer, manual_seed=manual_seed)

    ev = Evaluator(aspect2ind=dh.label2ind, logger=logger, metric=metric)

    # Test Teacher in Target Language
    test_df = dh.load_df(method='test', language=target_language)
    test_df['teacher_pred_id'] = teacher.predict(test_df)
    teacher_test_res = ev.evaluate(test_df, pred_col='teacher_pred_id', true_col='label', descr='Teacher Test {}'.format(target_language))

    printstr('Loading dev {} data'.format(target_language))
    dev_df = dh.load_df(method='dev', language=target_language)
    dev_df['teacher_pred_id'] = teacher.predict(dev_df)
    teacher_dev_res = ev.evaluate(dev_df, pred_col='teacher_pred_id', true_col='label', descr='Teacher Dev {}'.format(target_language))

    # Apply Teacher on unlabeled Train examples (Target Language)
    printstr("Applying Teacher on {} train data ({})".format(train_df.shape[0], target_language))
    train_df['teacher_pred_id'] = teacher.predict(train_df)
    teacher_train_res = ev.evaluate(train_df, pred_col='teacher_pred_id', true_col='label', descr='Teacher Train {}'.format(target_language))

    # Ignore "bad" Teacher's predictions...
    train_df_teacher = train_df[train_df['teacher_pred_id'] != -1]
    train_df_teacher['teacher_label'] = train_df_teacher['teacher_pred_id'].map(lambda x: dh.ind2label[x])

    # balance classes
    min_class_support = train_df_teacher['teacher_label'].value_counts().min()
    train_df_teacher = train_df_teacher.groupby('teacher_label', group_keys=False).apply(lambda x: x.sample(min_class_support, random_state=42))

    
    # Train Student on Target Language
    printstr("Training Student using Teacher's predictions on {} documents ({})".format(train_df_teacher.shape[0], target_language))
    student.train(train_df_teacher, eval_df=dev_df, label_name='teacher_label', eval_label_name='label')

    # Apply Student as Teacher on target language
    if num_iter == 2:
        train_df['student_pred_id'] = student.eval(train_df, label_name='label')
        printstr("Applying Student on train data...")
        student_train_res = ev.evaluate(train_df, pred_col='student_pred_id', true_col='label', descr='Student Train {}'.format(target_language))

        # Retrain Student on target language
        if actual_student_name != 'logreg':
            student_args['model_name'] = actual_student_name
            student = Student(language=target_language, label2ind=dh.label2ind, student_args=student_args, tokenizer=tokenizer.tokenizer, manual_seed=manual_seed)

        printstr("Re-training student on training data using student's predictions")
        train_df['student_label'] = train_df['student_pred_id'].map(lambda x: dh.ind2label[x])
        if dh.dataset == 'cls':
            # balance positive and negative class
            min_class_support = train_df['student_label'].value_counts().min()
            train_df = train_df.groupby('student_label', group_keys=False).apply(lambda x: x.sample(min_class_support, random_state=42))
        if actual_student_name == 'mbert':
            source_df['student_label'] = source_df['label']
            crosslingual_df = pd.concat([source_df, train_df])
            crosslingual_df.sample(frac=1)
            student.train(crosslingual_df, eval_df=dev_df, label_name='student_label', eval_label_name='label')
        else:
            student.train(train_df, eval_df=dev_df, label_name='student_label', eval_label_name='label')
    elif num_iter == 1:
        student_train_res = {}
    else:
        raise(BaseException('not implemented: num_iter={}'.format(num_iter)))

    # Evaluate Student on Target Language
    printstr("Evaluating Student on Dev set ({})".format(target_language))
    dev_df['student_pred_id'] = student.eval(dev_df, label_name='label')
    student_dev_res = ev.evaluate(dev_df, pred_col='student_pred_id', true_col='label', descr='Student Dev {}'.format(target_language))

    printstr("Evaluating Student on Test set ({})".format(target_language))
    test_df['student_pred_id'] = student.eval(test_df, label_name='label')
    student_test_res = ev.evaluate(test_df, pred_col='student_pred_id', true_col='label', descr='Student Test {}'.format(target_language))

    # Save student
    student.save(seed_words_path)
    return {
        'teacher_args': teacher_args,
        'student_args': student_args,
        'teacher_train_res': teacher_train_res,
        'teacher_dev_res': teacher_dev_res,
        'teacher_test_res': teacher_test_res,
        'student_dev_res': student_dev_res,
        'student_test_res': student_test_res,
        'student_train_res': student_train_res,
    }


def cotrain_wrapper(dataset='mldoc', source_language='english', target_language='spanish', num_epochs=3, num_seeds=50, train_size=1000,
                    sw_train_size=10000, manual_seed=0, student_name='logreg', experiments_dir='./',
                    alignment_method='google_translate', tokenizer_method='spacy', num_iter=2, cuda_device=0):
    now = datetime.now()
    basedatapath = '../data'

    # Define dataset-specific paths and metrics
    if dataset == 'mldoc':
        datapath = os.path.join(basedatapath, dataset)
        vocab_path = os.path.join(datapath, 'vocab/')
        seed_words_path = os.path.join(experiments_dir, '{}/seedwords/'.format(target_language))
        checkpoint_path = os.path.join(experiments_dir, target_language)
        metric = 'acc'
    elif 'cls' in dataset:
        dataset_name = dataset.split('_')[0]
        domain = dataset.split('_')[1]
        datapath = os.path.join(basedatapath, dataset_name)
        #vocab_path = os.path.join(datapath, 'vocab/')
        #datapath = os.path.join(home, 'data2/multilingual/{}/'.format(dataset_name))
        vocab_path = os.path.join(datapath, 'vocab/{}/'.format(domain))
        seed_words_path = os.path.join(experiments_dir, '{}_{}/seedwords/'.format(target_language, domain))
        checkpoint_path = os.path.join(experiments_dir, '{}_{}'.format(target_language, domain))
        metric = 'acc'
    elif dataset == 'twittersent':
        datapath = os.path.join(basedatapath, dataset)
        vocab_path = os.path.join(datapath, 'vocab/')
        seed_words_path = os.path.join(experiments_dir, '{}/seedwords/'.format(target_language))
        checkpoint_path = os.path.join(experiments_dir, target_language)
        metric = 'f1'
    elif dataset == 'lorelei':
        datapath = os.path.join(basedatapath, dataset)
        vocab_path = os.path.join(datapath, 'vocab/')
        seed_words_path = os.path.join(experiments_dir, '{}/seedwords/'.format(target_language))
        checkpoint_path = os.path.join(experiments_dir, target_language)
        metric = 'acc'
    else:
        raise (BaseException('dataset not supported: {}'.format(dataset)))

    if student_name == 'logreg' and not os.path.exists(vocab_path):
        print("Training tokenizer...")
        #tok_dataset = dataset if not 'cls' in dataset else dataset_name
        train_tokenizer(dataset, train_size=train_size, savepath=vocab_path)

    if alignment_method == 'google_translate':
        translation_dict_path = os.path.join(basedatapath, 'googletranslate_dict.pkl')
    elif alignment_method == 'muse_dict':
        # to use the MUSE dict, make sure you've ran "python prepare_muse_dict.py" under the "data" directory
        translation_dict_path = os.path.join(basedatapath, 'muse_dict.pkl')
    else:
        raise(BaseException('translation method not implemented: {}'.format(alignment_method)))

    # Define save paths
    print('Experiments dir: {}'.format(experiments_dir))
    os.makedirs(checkpoint_path, exist_ok=True)
    results_savefile = os.path.join(checkpoint_path, 'results.pkl')
    loggerfile = os.path.join(checkpoint_path, 'log.log')
    logger = get_logger(logfile=loggerfile)

    # Define model args
    # more arguments: https://github.com/ThilinaRajapakse/simpletransformers/blob/master/simpletransformers/config/global_args.py
    teacher_args = {'name': 'SeedCLF'}
    student_args = {
        'model_name': student_name,
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'evaluate_during_training': True,
        'num_train_epochs': num_epochs,
        'output_dir': checkpoint_path,
        'fp16': False,
        "manual_seed": manual_seed,
        "no_cache": True,
        "use_cached_eval_features": False,
        "tensorboard_dir": os.path.join(experiments_dir, 'tensorboard/'),
        "cache_dir": os.path.join(experiments_dir, 'cache/'),
        "use_early_stopping": True,
        "early_stopping_patience": 3,
        "n_gpu": 1,
        "save_model_every_epoch": False,
        "cuda_device": cuda_device
    }

    # Run co-training
    res = cotrain(dataset=dataset, source_language=source_language, target_language=target_language,
                  train_size=train_size, num_seeds=num_seeds, sw_train_size=sw_train_size,
                  translation_dict_path=translation_dict_path, seed_words_path=seed_words_path,
                  vocab_path=vocab_path, student_args=student_args, teacher_args=teacher_args, manual_seed=manual_seed,
                  logger=logger, alignment_method=alignment_method, tokenizer_method=tokenizer_method,
                  num_iter=num_iter, metric=metric)

    # Save results
    print("Finished experiments: {}".format(datetime.now()-now))
    print("Saved results at {}".format(results_savefile))
    joblib.dump(res, results_savefile)
    close(logger)
    return res


def report_seedwords(seed_word_dict, ind2label, print_fn=None):
    print_fn("Available target seedwords: {}".format(len(seed_word_dict)))
    overlap = np.array(list(seed_word_dict.keys()))[np.sum(np.array(list(seed_word_dict.values())) > 0, axis=1) == 2]
    print_fn("Overlapping seedwords: {}".format(overlap))
    for sw in overlap:
        print_fn("{}\t{}".format(sw, seed_word_dict[sw]))
    print_fn("Seed words per aspect:")
    if len(ind2label) == 2:
        # binary classification
        # positive class
        swlist = [sw for sw in seed_word_dict if seed_word_dict[sw][0] > 0]
        print_fn("{} ({}):\t{}".format(ind2label[1], len(swlist), swlist))
        for sw in swlist:
            print_fn("{}\t{}".format(sw, seed_word_dict[sw]))

        # negative class
        swlist = [sw for sw in seed_word_dict if seed_word_dict[sw][0] < 0]
        print_fn("{} ({}):\t{}".format(ind2label[0], len(swlist), swlist))
        for sw in swlist:
            print_fn("{}\t{}".format(sw, seed_word_dict[sw]))
    else:
        # multi-class classification
        for ind in ind2label:
            swlist = [sw for sw in seed_word_dict if seed_word_dict[sw][ind] > 0]
            print_fn("{} ({}):\t{}".format(ind2label[ind], len(swlist), swlist))
            for sw in swlist:
                print_fn("{}\t{}".format(sw, seed_word_dict[sw]))

