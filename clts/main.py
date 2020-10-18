import pandas as pd
import os
import sys
import re
import time
import joblib
from datetime import datetime
import numpy as np
import glob
import json
from Cotraining import cotrain_wrapper
import argparse
from collections import defaultdict
from util import report_res

def run_trainer(args, target_language):
    print("Running for {}".format(target_language))
    args.target_language = target_language
    res = cotrain_wrapper(dataset=args.dataset, source_language=args.source_language, target_language=args.target_language,
                          num_epochs=args.num_epochs, num_seeds=args.num_seeds, train_size=args.train_size, sw_train_size=args.sw_train_size,
                          manual_seed=args.manual_seed, student_name=args.student_name, experiments_dir=args.logdir,
                          alignment_method=args.alignment_method, num_iter=args.num_iter, cuda_device=args.cuda_device, tokenizer_method=args.tokenizer_method)
    return res


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', help="Dataset: mldoc/cls/lorelei", type=str, default='cls')
    parser.add_argument('--source_language', help="Source language", type=str, default='english')
    parser.add_argument('--target_language', help="Source language", type=str, default='french')
    parser.add_argument('--num_epochs', help="Number of epochs for BERT Student only (default: 2)", type=int, default=2)
    parser.add_argument('--num_iter', help="Number of iterations for Teacher-Student co-training in target language", type=int, default=2)
    parser.add_argument('--num_seeds', help="Translation budget, i.e., number of seed words to translate per class (default: 50)", type=int, default=50)
    parser.add_argument('--train_size', help="Training size for Student training in target language (default: 1000)", type=int, default=1000)
    parser.add_argument('--sw_train_size', help="Training size for seed word extraction in source language (default: 1000)", type=int, default=1000)
    parser.add_argument('--student_name', help="Name of the student. Supported names: logreg, bert", type=str, default='logreg')
    parser.add_argument('--comment', help="Comment used for logging reasons", type=str, default='cotraining')
    parser.add_argument('--alignment_method', help="Method to align words across languages: google_translate/muse_dict", type=str, default='muse_dict')
    parser.add_argument('--tokenizer_method', help="Method to tokenize strings to tokens: spacy/clean", type=str, default='spacy')
    parser.add_argument('--cuda_device', help="GPU Device ID", type=int, default=0)
    parser.add_argument('--logdir', help="log directory", type=str, default='./')
    parser.add_argument('--debug', help="Enable debug mode", action='store_true')
    parser.add_argument('--supervised', help="Enable debug mode", action='store_true')
    args = parser.parse_args()

    if not args.dataset in ['mldoc', 'cls', 'twittersent', 'lorelei']:
        raise (BaseException('dataset not supported: {}'.format(args.dataset)))

    datapath = '../experiments/{}'.format(args.dataset)
    language = args.target_language
    args.logdir = os.path.abspath(os.path.join(datapath, '{}'.format(args.logdir)))
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d-%H_%M")
    manual_seeds = [20, 7, 1993, 42, 127]
    args.logdir = args.logdir +  "/" + \
                  date_time + \
                  "_{}".format(args.comment) + \
                  "_{}".format(args.student_name) + \
                  "_SRC{}".format(args.source_language) + \
                  "_TGT{}".format(args.target_language) + \
                  "_sw{}".format(args.num_seeds) + \
                  "_ITER{}".format(args.num_iter)

    # RUN EXPERIMENT
    original_logdir = args.logdir
    os.makedirs(args.logdir, exist_ok=True)
    print('\t\tEXPERIMENT with dataset={}\nargs: {}\nlogdir: {}'.format(args.dataset, args, args.logdir))
    all_results_savefile = os.path.join(args.logdir, 'results.pkl')
    all_res = []
    if args.dataset in ['mldoc', 'twittersent', 'lorelei']:
        for i, manual_seed in enumerate(manual_seeds):
            args.manual_seed = manual_seed
            res = {}
            args.logdir = os.path.join(original_logdir, 'run_{}'.format(i))
            target_language = args.target_language
            if args.supervised:
                args.source_language = target_language
            elif target_language == args.source_language:
                continue
            print("\n\n\t\t *** {}->{} ***".format(args.source_language, target_language))
            res[target_language] = run_trainer(args, target_language)
            all_res.append(res)
        # Save results
        print("Saved results at {}".format(all_results_savefile))
        joblib.dump(all_res, all_results_savefile)
        report_res(all_res, original_logdir, languages=[target_language])
    elif args.dataset == 'cls':
        original_dataset = args.dataset
        for i, manual_seed in enumerate(manual_seeds):
            args.manual_seed = manual_seed
            res = {}
            args.logdir = os.path.join(original_logdir, 'run_{}'.format(i))
            domains = ['books', 'dvd', 'music']
            target_language = args.target_language
            for domain in domains:
                if args.supervised:
                    args.source_language = target_language
                elif target_language == args.source_language:
                    continue
                args.dataset = original_dataset + '_{}'.format(domain)
                print("\n\n\t\t *** [{}] {}->{} ***".format(args.dataset, args.source_language, target_language))
                res[target_language] = run_trainer(args, target_language)
                res[target_language + '_' + domain] = res.pop(target_language)
                all_res.append(res)
        # Save results
        print("Saved results at {}".format(all_results_savefile))
        joblib.dump(all_res, all_results_savefile)
        report_res(all_res, original_logdir, languages=list(all_res[0].keys()))
