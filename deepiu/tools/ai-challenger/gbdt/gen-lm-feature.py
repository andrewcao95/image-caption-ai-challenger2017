#!/usr/bin/env python
#encoding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', 'ensemble.train.detect.scene.txt', '')
flags.DEFINE_string('type', 'valid', 'valid or test')

import sys
import cPickle as pickle
import numpy as np

import gezi

print('type', FLAGS.type)
if FLAGS.type == 'test':
  FLAGS.input_file = './ensemble.inference.feature.detect.scene.txt'

lm_file = './lm.pkl' 

ofile = FLAGS.input_file.replace('.txt', '.lm.txt')
print('infile:', FLAGS.input_file, 'ofile:',ofile, file=sys.stderr)
names = []
names += ['lm_score']

with open(lm_file) as f:
  lm_results = pickle.load(f)

with open(ofile, 'w') as out:
  is_header = True
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
    lm_score = lm_results[caption]
    
    l += map(str, [lm_score])
   
    print('\t'.join(l), file=out)
      

