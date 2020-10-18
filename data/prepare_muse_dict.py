import os
import joblib
import numpy as np
from collections import defaultdict
from numpy import log
from time import time

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


def create_muse_dict_mldoc_cls(muse_dict_folder, savefolder):
    # Creating MUSE dictionaries: https://github.com/facebookresearch/MUSE
    # First need to download the repo. and follow their commands to download dictionaries.
    # In summary you need to run: ./get_evaluation.sh script of this repo.
    # Then, assume that you have downloaded the dictionaries in the following folder
    print("Creating MUSE dict...")
    source_languages = ["english"]
    target_languages = ["german", "spanish", "french", "italian", "russian", "chinese", "japanese"]
    muse_dict = defaultdict()
    for source_language in source_languages:
        for target_language in target_languages:
            if source_language == target_language:
                continue
            print("{}->{}".format(source_language, target_language))
            dict_path = os.path.join(muse_dict_folder, "{}-{}.txt".format(lang_dict[source_language],lang_dict[target_language],))
            if not os.path.exists(dict_path):
                print("ERROR: Could not load {}".format(dict_path))
                continue
            translation_dict = defaultdict(list)
            with open(dict_path, 'r') as f:
                for line in f.readlines():
                    try:
                        src,tgt = line.strip().split()
                    except:
                        print("ERROR: could not read line: {}".format(line.strip()))
                        continue
                    # NOTE: the same src word may have many tgt words in different entries
                    # This is why we use .append(tgt) to keep them all...
                    translation_dict[src].append(tgt)
            muse_dict[(source_language, target_language)] = translation_dict
    print('Saving muse dict into {}'.format(savefolder))
    joblib.dump(muse_dict, savefolder)
    print("DONE.")
    return muse_dict


def create_muse_dict_twitter_sentiment(muse_dict_folder, savefile='./muse_dict.pkl'):
    # Creating MUSE dictionaries: https://github.com/facebookresearch/MUSE
    # First need to download the repo. and follow their commands to download dictionaries.
    # In summary you need to run: ./get_evaluation.sh script of this repo.
    # Then, assume that you have downloaded the dictionaries in the following folder
    languages = ['arabic', 'bulgarian', 'german', 'english', 'spanish', 'persian', 'croatian', 'hungarian',
                 'polish', 'portuguese', 'russian', 'slovak', 'slovenian', 'swedish', 'chinese']
    muse_dict = defaultdict()
    source_language = 'english'
    for target_language in languages:
        if source_language == target_language:
            continue
        print("{}->{}".format(source_language, target_language))
        dict_path = os.path.join(muse_dict_folder, "{}-{}.txt".format(lang_dict[source_language],lang_dict[target_language]))
        if not os.path.exists(dict_path):
            print("ERROR: Could not load {}".format(dict_path))
            continue
        translation_dict = defaultdict(list)
        with open(dict_path, 'r') as f:
            for line in f.readlines():
                try:
                    src,tgt = line.strip().split()
                except:
                    print("ERROR: could not read line: {}".format(line.strip()))
                    continue
                # NOTE: the same src word may have many tgt words in different entries
                # This is why we use .append(tgt) to keep them all...
                translation_dict[src].append(tgt)
        muse_dict[(source_language, target_language)] = translation_dict
    print('Saving muse dict into {}'.format(savefile))
    joblib.dump(muse_dict, savefile)
    print("DONE.")
    return muse_dict


if __name__=='__main__':
    datafolder = "../data/"
    muse_dict_folder = os.path.join(datafolder, "MUSE/data/crosslingual/dictionaries")
    savefolder = os.path.join(datafolder, 'muse_dict.pkl')
    create_muse_dict_mldoc_cls(muse_dict_folder, 'muse_dict.pkl')

    # twitter_sent_file = '/home/gkaraman/data2/multilingual/twitter_sent/translation_resources/muse_dict.pkl'
    # create_muse_dict_twitter_sentiment(muse_dict_folder, twitter_sent_file)