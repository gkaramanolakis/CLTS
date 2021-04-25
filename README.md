# CLTS: Cross-lingual Teacher-Student

This repo holds the code for our Findings of EMNLP'20 paper: "[Cross-Lingual Text Classification with Minimal Resources by Transferring a Sparse Teacher](https://www.aclweb.org/anthology/2020.findings-emnlp.323.pdf)" 

## Overview of CLTS

While most NLP models and datasets have been developed in English, it is important to consider more languages out of the 4,000 written languages in the world. However, it would be expensive or sometimes impossible to obtain labeled training data across all languages. CLTS is a cross-lingual text classification framework that trains neural networks for a target language **without labeled data** in the target language. 

CLTS (Cross-Lingual Teacher Student) **automatically generates labeled training data** in a target language (e.g., French) by transferring weak supervision from a different source language (e.g., English) for which labeled data are available: 

![alt text](https://github.com/gkaramanolakis/CLTS/blob/main/CLTS.jpg?raw=true)

Main Components: 
* **Seed Word Extractor**: extracts the _B_ most indicative keywords, or seed words, using training data in the source language. The _B_ seed words are then translated to the target language. (_B_ is the translation budget, e.g., _B_=20 word translations.)
* **Teacher**: a simple classifier that uses translated seed words to automatically annotate unlabeled data in the target language.
* **Student**: a base model (e.g., a BERT-based classifier) that considers both seed words and their context in unlabeled documents. 

The core advantage of CLTS is that it can transfer supervision across languages **efficiently**, i.e., without requiring expensive cross-lingual resources such as machine translation systems or high-coverage bilingual dictionaries. 

With as few as _B_=20 seed-word translations, CLTS is **effective across 18 languages** and 4 tasks and sometimes outperforms even more expensive state-of-the-art methods. 

#### News document classification (MLDoc) results:

Method | De | Es | Fr | It | Ru | Zh | Ja 
--- | --- | --- | --- |--- |--- |--- |--- 
MultiBERT | 79.8 | 72.1 | 73.5 | 63.7 | 73.7 | 76.0 | 72.8
MultiCCA (_B_=20K) | 81.2 | 72.5 | 72.4 | 69.4 | 60.8 | 74.7 | 67.6
**CLTS (_B_=160)** | **90.4** | **86.3** | **91.2** | **74.7** | **75.6** | **84.0** | **72.6**

#### Review sentiment classification (CLS) results: 

Method | De | Fr | Ja 
--- | --- | --- | --- 
MultiBERT | 72.0 | 75.4 | 66.9 
VECMAP | 75.3 | 78.2 | 55.9
CL-SCL (_B_=450) | 78.1 | 78.4 | 73.1 
**CLTS (_B_=20)** | **80.1** | **83.4** | **77.6** 

Our [EMNLP Findings '20 paper](https://www.aclweb.org/anthology/2020.findings-emnlp.323.pdf) describes CLTS in detail and provides more experimental results. 

## Installing and Running CLTS
### Install Requirements (Python 3.7)
```
pip install -r requirements.txt
```

### Download Data
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

### Run CLTS 
```
cd clts
python main.py --dataset cls --source_language english --target_language french --num_seeds 10 
```
Experimental results are stored under the 'experiments' directory. 

### Run CLTS for all languages and domains in the CLS dataset
```
cd scripts
bash run_cls.sh
```

## Citation 

```
@inproceedings{karamanolakis2020cross,
  title={Cross-Lingual Text Classification with Minimal Resources by Transferring a Sparse Teacher},
  author={Karamanolakis, Giannis and Hsu, Daniel and Gravano, Luis},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings},
  pages={3604--3622},
  year={2020}
}
```
