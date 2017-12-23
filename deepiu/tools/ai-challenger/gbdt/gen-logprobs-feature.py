#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-attention-feature.py
#        \author   chenghuige  
#          \date   2017-12-02 20:08:52.122757
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import glob 
import numpy as np
import cPickle as pickle
import gezi
import math

from deepiu.util import idf

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', 'ensemble.train.detect.scene.lm.attention.txt', '')
flags.DEFINE_string('type', 'valid', 'valid or test')
flags.DEFINE_string('img_dir', '/home/gezi/data2/data/ai_challenger/image_caption/pic/', '')

print('type', FLAGS.type)
if FLAGS.type == 'test':
  FLAGS.input_file = './ensemble.inference.feature.detect.scene.lm.attention.txt'
ofile = FLAGS.input_file.replace('.txt', '.logprobs.txt')
print('infile:', FLAGS.input_file, 'ofile:',ofile, file=sys.stderr)

logprobs_files = glob.glob('./*model*.evaluate-inference.logprobs.txt') if FLAGS.type == 'valid' \
              else glob.glob('./*model*.inference.logprobs.txt')

assert len(logprobs_files) == 19

names = []
names += ['sum', 'min', 'max', 'mean']
names += ['idf_sum', 'idf_min', 'idf_max', 'idf_mean']
names += ['count_de', 'count_he']
names = ['logprobs_' + x for x in names]

from deepiu.util import vocabulary
vocabulary.init('/home/gezi/mount/temp/image-caption/ai-challenger/tfrecord/seq-basic/vocab.txt')
vocab = vocabulary.vocab

idf_weights = idf.get_idf()

m = {}
count_map = {}

def deal_logprobs(file):
  for i, line in enumerate(open(file)):
    # if i == 0:
    #   continue
    if i % 1000 == 0:
      print(i, end='\r', file=sys.stderr)
    l = line.strip().split('\t')
    img, caption = l[0], l[1]
    words = caption.split()
    words.append('</S>')
    caption = caption.replace(' ', '')
    logprobs = l[-1].split()
    logprobs = map(float, logprobs)
    # TODO FIXME whey some(small ratio) caption with -inf not as expected num_words + 1(end mark) ?
    logprobs = [x for x in logprobs if x > -1000]
    if len(logprobs) < len(words):
      words = words[:-1]
    logprobs = np.array(logprobs)
    if img not in m:
      m[img] = {}
      count_map[img] = {}
    if caption not in m[img]:
      m[img][caption] = np.array([0.] * len(names))
      count_map[img][caption] = 0
    fe = m[img][caption]
    count_map[img][caption] += 1

    fe[0] += logprobs.sum() 
    fe[1] += logprobs.min()
    fe[2] += logprobs.max() 
    fe[3] += logprobs.mean()

    idf_logprobs = []
    for i, word in enumerate(words):
      idf_logprobs.append(idf_weights[vocab.id(word)] * math.exp(logprobs[i]))
    idf_logprobs = np.array(idf_logprobs)
    fe[4] += idf_logprobs.sum()
    fe[5] += idf_logprobs.min()
    fe[6] += idf_logprobs.max()
    fe[7] += idf_logprobs.mean()

    fe[8] += words.count('的')
    fe[9] += words.count('和')
      
for i, file in enumerate(logprobs_files):
  timer = gezi.Timer('deal %d %s' % (i, file))
  deal_logprobs(file)
  timer.print()

with open(ofile, 'w') as out:
  timer = gezi.Timer('merge previous features')
  is_header = True
  default_fe = [0.] * len(names)
  for line in open(FLAGS.input_file):
    l = line.strip().split('\t')
    if is_header:
      names = l + names 
      print('\t'.join(names), file=out)
      with open('./feature_name.txt', 'w') as out_fname:
        for name in names[3:]:
          print(name, file=out_fname)
      is_header = False
      continue
    img, caption = l[0], l[1]
    
    if caption in m[img]:
      fe = m[img][caption]
      fe = [x / count_map[img][caption] for x in fe]
    else:
      fe = default_fe
    l += map(str, fe)
    print('\t'.join(l), file=out)
  timer.print()

      