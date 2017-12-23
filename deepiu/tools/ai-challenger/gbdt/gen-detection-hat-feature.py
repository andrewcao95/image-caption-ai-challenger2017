#!/usr/bin/env python
#encoding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', 'ensemble.train.txt', '')

import sys
import pickle

from deepiu.util.detector import Detector

class_names = open('/home/gezi/mine/hasky/deepiu/object_detection/data/oid_label_caption.txt').readlines()
class_names = [x.strip() for x in class_names]
print(class_names[0], class_names[1], class_names[10], 'num_classes:', len(class_names))
assert(len(class_names) == 545)

hat_names = set(['帽子', '太阳帽', '牛仔帽', '草帽'])

hat_ids = []
for i in range(len(class_names)):
  if class_names[i] in hat_names:
    hat_ids.append(i) 
hat_ids = set(hat_ids)

print('hat_ids', hat_ids, file=sys.stderr)

detection_file = '/home/gezi/mine/mount/temp/image-caption/ai-challenger/detection/openimage/faster-rcnn.lowproposals/valid.pkl'

def get_hat_score(detection_result):
  score = 0.
  MAX_INDEX = 1000
  pos = MAX_INDEX
  for i in range(detection_result.num):
    if detection_result.classes[i] in hat_ids:
      score = detection_result.scores[i]
      pos = i
      break
  return score, pos

ofile = FLAGS.input_file.replace('.txt', '.detect_hat.txt')
print('infile:', FLAGS.input_file, 'ofile:',ofile, file=sys.stderr)
names = ['caption_hat', 'detect_hat_score', 'detect_hat_pos']

with open(detection_file) as f:
  detection_results = pickle.load(f)

with open(ofile, 'w') as out:
  is_header = True
  for line in open(FLAGS.input_file):
    l = line.strip().split('\t')
    if is_header:
      names = l + names 
      print('\t'.join(names), file=out)
      is_header = False
      continue
    img, caption = l[0], l[1]
    
    detection_result = detection_results[img]
    score, pos = get_hat_score(detection_result)
    # make from (0, 1) to (-1, 1)
    #score = (score - 0.5) * 2  
    if '帽' in caption:
      has_hat = 1.
    else:
      has_hat = 0. 
    #score *= has_hat
    l += map(str, [has_hat, score, pos])
    print('\t'.join(l), file=out)
      
