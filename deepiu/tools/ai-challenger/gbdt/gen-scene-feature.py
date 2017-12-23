#!/usr/bin/env python
#encoding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', 'ensemble.train.detect.txt', '')
flags.DEFINE_string('type', 'valid', 'valid or test')

import sys
import cPickle as pickle
import numpy as np

import gezi 

print('type', FLAGS.type)

if FLAGS.type == 'test':
  FLAGS.input_file = './ensemble.inference.feature.detect.txt'

scene_file = './scene.%s.pkl' % FLAGS.type
#scene_inceptionV4_file = './scene.inceptionV4.%s.pkl' % FLAGS.type

scene_names = open('/data2/data/ai_challenger/scene/ai_challenger_scene_train_20170904/scene_classes_name.txt').readlines()
scene_names = [x.strip().split(',')[1] for x in scene_names]
print(scene_names[0])

def get_scene_score(scene_result, caption):
  for i in range(len(scene_result.top_logit)):
    scene_name = scene_names[scene_result.top_class[i]]
    scenes = scene_name.split()
    for scene in scenes:
      if scene in caption:
        return scene_result.top_logit[i], i 
  return 0, len(scene_result.top_logit)

ofile = FLAGS.input_file.replace('.txt', '.scene.txt')
print('infile:', FLAGS.input_file, 'ofile:',ofile, file=sys.stderr)
names = []
#names += ['scene_logit', 'scene_pos', 'scene_ensemble_logit', 'scene_ensemble_pos']
names += ['scene_logit', 'scene_pos']

timer = gezi.Timer('load scene_file %s' % scene_file, True)
with open(scene_file) as f:
  scene_results = pickle.load(f)
timer.print()
#timer = gezi.Timer('load scene_inceptionV4_file', True)
#with open(scene_inceptionV4_file) as f:
#  scene_inceptionV4_results = pickle.load(f)
#timer.print()

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
    
    scene_result = scene_results[img]
    logit, pos = get_scene_score(scene_result, caption)
    l += map(str, [logit, pos])
    
    #scene_inceptionV4_result = scene_inceptionV4_results[img]
    #logit2, pos2 = get_scene_score(scene_inceptionV4_result, caption)
    ###l += map(str, [logit, pos])
    #
    #logit = (logit + logit2) / 2
    #pos = (pos + pos2) / 2
    #l += map(str, [logit, pos])
    
    print('\t'.join(l), file=out)
      

