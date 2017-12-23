#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   calc-idf.py
#        \author   chenghuige  
#          \date   2017-12-02 14:30:20.776964
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('type', 'valid', '')
flags.DEFINE_string('input_file', 'ensemble.evaluate-inference.txt', '')
flags.DEFINE_string('result_file', None, '')

import operator
import cPickle as pickle
from collections import namedtuple
import gezi

from deepiu.util import vocabulary
vocabulary.init('/home/gezi/mount/temp/image-caption/ai-challenger/tfrecord/seq-basic/vocab.txt')
from deepiu.util import idf 

idf_weights = idf.get_idf()

vocab = vocabulary.vocab

print('type', FLAGS.type)

if FLAGS.type == 'test':
  FLAGS.input_file = './ensemble.inference.txt'
  ofile = './ensemble.test.idf.txt'
  opkl = './ensemble.test.idf.pkl'
else:
  ofile = './ensemble.valid.idf.txt'
  opkl = './ensemble.valid.idf.pkl'

print('infile:', FLAGS.input_file, 'ofile:', ofile)
timer = gezi.Timer('load segged_caption', True)
with open('./segged_caption.pkl') as f:
  segged_captions_map = pickle.load(f)
timer.print()

m = {}
Result = namedtuple('Result', ['words', 'scores'])
with open(ofile, 'w') as out:
  for i, line in enumerate(open(FLAGS.input_file)):
    if i % 1000 == 0:
      print('num imgs done %d' % i, end='\r')
    l = line.strip().split('\t')
    img, captions = l[0], l[-2]
    captions = captions.split(' ')
    for caption in captions:
      segged_caption = segged_captions_map[caption]
      word_ids = [vocab.id(x) for x in segged_caption]
      weights = [idf_weights[x] for x in word_ids]
      caption_weights = zip(segged_caption, weights)
      caption_weights.sort(key=operator.itemgetter(1), reverse=True)
      results = zip(*caption_weights)
      m[img][caption] = Result(*results)
      captions = ' '.join(results[0])
      scores = ' '.join(map(str, results[1]))
      print(img, caption, captions, scores, sep='\t', file=out)
    
with open(opkl, 'w') as out:
  pickle.dump(m, out) 
