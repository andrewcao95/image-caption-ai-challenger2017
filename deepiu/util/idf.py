#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   idf.py
#        \author   chenghuige  
#          \date   2017-11-23 17:55:20.548347
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import dill

import numpy as np

import melt 
logging = melt.logging

from deepiu.util import vocabulary
from deepiu.image_caption.conf import NUM_RESERVED_IDS

import tensorflow as tf

flags = tf.app.flags
#import gflags as flags
FLAGS = flags.FLAGS

def get_idf():
  logging.info('cider using global idf info from valid data')
  vocab_size = vocabulary.vocab_size
  if not vocab_size:
    vocabulary.init()
  test_dir = FLAGS.valid_resource_dir
  document_frequency_path = os.path.join(test_dir, 'valid_refs_document_frequency.dill')
  print('document_frequency_path', document_frequency_path, file=sys.stderr)
  assert os.path.exists(document_frequency_path), document_frequency_path
  ref_len_path = os.path.join(test_dir, 'valid_ref_len.txt')
  assert os.path.exists(ref_len_path), ref_len_path
  document_frequency = dill.load(open(document_frequency_path))
  ref_len = float(open(ref_len_path).readline().strip())
  logging.info('document_frequency {} ref_len {}'.format(len(document_frequency), ref_len))

  def calc_idf(word):
    # NOTICE! easy to mistake...
    df = np.log(max(1.0, document_frequency[(word, )]))
    term_freq = 1
    return float(term_freq) * (ref_len - df)
  idf = [calc_idf(vocabulary.vocab.key(id)) for id in range(NUM_RESERVED_IDS, vocab_size)]
  idf = [0.] * NUM_RESERVED_IDS + idf
  print('idf 0, 1, 2, 100', idf[0], idf[1], idf[2], idf[100], file=sys.stderr)
  return idf
