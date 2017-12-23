#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-imagenet-feature.py
#        \author   chenghuige  
#          \date   2017-12-03 13:50:33.562133
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', 'ensemble.train.detect.scene.lm.attention.logprobs.txt', '')
flags.DEFINE_string('type', 'valid', 'valid or test') 

names = []
names += ['count', 'sum', 'mean']
names = ['imagenet_' + x for x in names]

import gezi 
import numpy as np 
import cPickle as pickle

print('type', FLAGS.type)

if FLAGS.type == 'test':
  FLAGS.input_file = './ensemble.inference.feature.detect.scene.lm.attention.logprobs.txt'

imagenet_file = './imagenet.%s.pkl' % FLAGS.type

ofile = FLAGS.input_file.replace('.txt', '.imagenet.txt')
print('infile:', FLAGS.input_file, 'ofile:',ofile, file=sys.stderr)

timer = gezi.Timer('load segged caption file')
with open('./segged_caption.pkl') as f:
  segged_captions = pickle.load(f)
timer.print()

class_names = open('/data2/data/imagenet/raw-data/label-caption.txt').readlines()
class_names = [x.strip().split(',')[-1] for x in class_names]
class_names.insert(0, 'None')
print(class_names[0], class_names[559])


timer = gezi.Timer('load imagenet_file %s' % imagenet_file, True)
with open(imagenet_file) as f:
  imagenet_results = pickle.load(f)
timer.print()

def get_imagenet_score(imagenet_result, words):
  logits = 0. 
  hit_set = set()
  words = set(words)
  for cid, logit in zip(imagenet_result.top_class, imagenet_result.top_logit):
    if logit < 1.7:
      break
    cnames = class_names[cid].split()
    for cname in cnames:
      if cname in words:
        hit_set.add(cname)
        logits += logit
        
  return len(hit_set), logits
      
      
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
    words = segged_captions[caption]
    
    imagenet_result = imagenet_results[img]
    count, score = get_imagenet_score(imagenet_result, words)
    l += map(str, [count, score, score / (count + 1)])
    
    print('\t'.join(l), file=out)
      
