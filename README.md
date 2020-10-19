# CLTS: Cross-lingual Teacher-Student

This repo holds the code for our Findings of EMNLP'20 paper: "[Cross-Lingual Text Classification with Minimal Resources by Transferring a Sparse Teacher](https://arxiv.org/pdf/2010.02562.pdf)" 

CLTS trains a classifier in a target language (e.g., French) with minimal resources by transferring weak supervision from a different source language (e.g., English). Given a limited translation budget, CLTS transfers a sparse classifier (Teacher) across languages and trains a more powerful classifier (Student) in the target language.

## Install Requirements (Python 3.7)
```
pip install -r requirements.txt
```

## Download Data
Download the MUSE dictionary:
```
cd data
bash download_muse_dict.sh
```

Download the CLS dataset:
```
cd data/cls
bash prepare_cls.sh
```

Download the MLDoc dataset: 
```
cd data/mldoc
python prepare_mldoc.py
```

## Run CLTS 
```
cd clts
python main.py --dataset cls --source_language english --target_language french --num_seeds 10 
```
Experimental results are stored under the 'experiments' directory. 

## Run CLTS for all languages and domains in the CLS dataset
```
cd scripts
bash run_cls.sh
```

