#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-tfidf-keywords.py
#        \author   chenghuige  
#          \date   2017-10-28 11:13:50.023262
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import sys
import os
import dill
import numpy as np

import gezi
from gezi import Segmentor
segmentor = Segmentor()


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('valid_resource_dir', '/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic/valid', '')

test_dir = FLAGS.valid_resource_dir
document_frequency_path = os.path.join(test_dir, 'valid_refs_document_frequency.dill')
assert os.path.exists(document_frequency_path), document_frequency_path
ref_len_path = os.path.join(test_dir, 'valid_ref_len.txt')
assert os.path.exists(ref_len_path), ref_len_path
document_frequency = dill.load(open(document_frequency_path))
ref_len = float(open(ref_len_path).readline().strip())
print('document_frequency {} ref_len {}'.format(len(document_frequency), ref_len), file=sys.stderr)

def calc_tfidf(word, term_freq):
  df = np.log(max(1.0, document_frequency[(word, )]))
  #print(word, term_freq, (ref_len - df), float(term_freq) * (ref_len - df))
  return float(term_freq) * (ref_len - df)
 
tfidf_map = {}
num_lines = 0
for line in sys.stdin:
  img, captions_str = line.strip().split('\t')

  captions = captions_str.split('\x01')

  count_map = {}
  for text in captions:
    text = gezi.norm(text)
    words = segmentor.Segment(text)
    # right now only consider onegram
    for word in words:
      count_map.setdefault(word, 0)
      count_map[word] += 1
  
  tfidf_list = [(calc_tfidf(word, count), word) for word, count in count_map.items()]
  tfidf_list.sort(reverse=True)

  for tfidf, word in tfidf_list:
    tfidf_map.setdefault(word, 0)
    tfidf_map[word] += tfidf

  num_lines += 1

tfidf_list = [(tfidf, word) for word, tfidf in tfidf_map.items()]
tfidf_list.sort(reverse=True)

for tfidf, word in tfidf_list:
  print(word, tfidf / num_lines , sep='\t')
