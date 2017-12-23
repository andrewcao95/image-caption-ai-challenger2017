#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   lm-inference.py
#        \author   chenghuige  
#          \date   2017-11-30 19:06:34.154922
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import cPickle as pickle

import gezi

from deepiu.util.lm_predictor import LanguageModelPredictor as Predictor
from deepiu.util.text2ids import words2ids 
from deepiu.util import text2ids

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('type', 'valid', '')
flags.DEFINE_string('result_file', None, '')

text2ids.init('/home/gezi/temp/image-caption/ai-challenger/tfrecord/seq-basic/vocab.txt')

batch_size = 1024

timer = gezi.Timer('load segged_caption', True)
with open('./segged_caption.pkl') as f:
  segged_captions_map = pickle.load(f)
timer.print()

model_dir = '/home/gezi/mount/temp/image-caption/ai-challenger/model.v5/lm'

timer = gezi.Timer('load lm predictor', True)
predictor = Predictor(model_dir)
timer.print()

lm_scores_map = {}

captions = []
segged_captions = []

def predict(captions, segged_captions):
  scores = predictor.predict(segged_captions)
  for caption, score in zip(captions, scores):
    lm_scores_map[caption] = score
  print('captions done %d %f' % (len(lm_scores_map), len(lm_scores_map) / len(segged_captions_map)), end='\r')

for caption, segged_caption in segged_captions_map.items():
  captions.append(caption)
  segged_captions.append(words2ids(segged_caption))
  if len(captions) == batch_size:
    predict(captions, segged_captions)
    captions = []
    segged_captions = []

if captions:
  predict(captions, segged_captions)

timer = gezi.Timer('dump lm result', True)
result_file = FLAGS.result_file or 'lm.pkl' 
with open(result_file, 'w') as out:
  pickle.dump(lm_scores_map, out)

