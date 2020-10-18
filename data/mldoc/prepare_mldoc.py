# MLDoc dataset: A Corpus for Multilingual Document Classification in Eight Languages
# paper: https://github.com/facebookresearch/MLDoc
import os
import glob
import argparse
import logging
import os
import xml.etree.ElementTree as ET

# First, you need to download the RCV1 and RCV2 corpora:
# Credentials are required!
# Type in a terminal:
#   !curl --insecure -u <CREDENTIALS> https://ir.nist.gov/reuters/rcv1.tar.xz -o ./rcv1.tar.xz
#   !curl --insecure -u <CREDENTIALS> https://ir.nist.gov/reuters/rcv2.tar.xz -o ./rcv2.tar.xz
if not os.path.exists('./rcv1') or not os.path.exists('./rcv2'):
    print("You need credentials to download the MLDoc dataset...\nFirst download rcv1 and rcv2 and put them under 'mldoc'.")
    print("curl --insecure -u <CREDENTIALS> https://ir.nist.gov/reuters/rcv1.tar.xz -o ./rcv1.tar.xz")
    print("curl --insecure -u <CREDENTIALS> https://ir.nist.gov/reuters/rcv2.tar.xz -o ./rcv2.tar.xz")
    exit()

# Then, you need to run the pre-processing code below
# source: https://github.com/facebookresearch/MLDoc/blob/master/generate_documents.py
# My version below fixes a small bug of the source version
global ch
def generate_documents(indices_file, output_filename, rcv_dir):
    global ch
    delim_str = '\t'
    sentence_delim = ' '
    code_class = 'bip:topics:1.0'
    labels = ['C', 'E', 'G', 'M']
    target_topics = ['{}CAT'.format(label) for label in labels]
    with open(indices_file, 'r') as indices_f, \
            open(output_filename, 'w') as output_f:
        for line in indices_f:
            sub_corpus, file_name = line.strip().split('-')
            sub_corpus_path = os.sep.join([rcv_dir, sub_corpus])
            doc_path = os.sep.join(
                [sub_corpus_path, '{}.xml'.format(file_name)]
            )
            data_str = open(doc_path).read()
            xml_parsed = ET.fromstring(data_str)
            topics = [
                topic.attrib['code'] for topic in xml_parsed.findall(
                    ".//codes[@class='{}']/code".format(code_class)
                ) if topic.attrib['code'] in target_topics
            ]
            assert len(topics) == 1, 'More than one class label found.'
            doc = sentence_delim.join(
                [p.text for p in xml_parsed.findall(".//p")]
            )
            output_f.write( '{}{}{}\n'.format(topics[0], delim_str, doc) )
    return


rcv1dir = "mldoc/rcv1/"
rcv2dir = "mldoc/rcv2/"
indices_dir = "mldoc-indices"
indices_files = glob.glob(indices_dir + "/*")
output_folder = "mldoc/preprocessed_data"
if len(indices_files) == 0:
    raise(BaseException('could not load indices from {}'.format(indices_dir)))

for indices_file in indices_files:
    fname = indices_file.split('/')[-1]
    print(fname)
    language = fname.split('.')[0]
    output_filename = os.path.join(output_folder, fname)
    if language == 'english':
        rcv_dir = rcv1dir
    else:
        rcv_dir = os.path.join(rcv2dir, language)
    generate_documents(indices_file, output_filename, rcv_dir)