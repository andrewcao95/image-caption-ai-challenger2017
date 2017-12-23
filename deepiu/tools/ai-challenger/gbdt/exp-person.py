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
import numpy as np

import gezi

from deepiu.util.detector import Detector

class_names = open('/home/gezi/mine/hasky/deepiu/object_detection/data/oid_label_caption.txt').readlines()
class_names = [x.strip() for x in class_names]
print(class_names[0], class_names[1], class_names[10], 'num_classes:', len(class_names))
assert(len(class_names) == 545)

person_names = ['男人', '女人', '男孩', '女孩']

hat_ids = []
for i in range(len(class_names)):
  if class_names[i] in person_names:
    hat_ids.append(i) 
hat_ids = set(hat_ids)

print('hat_ids', hat_ids, file=sys.stderr)

detection_file = '/home/gezi/mine/mount/temp/image-caption/ai-challenger/detection/openimage/faster-rcnn.lowproposals/valid.pkl'

def get_person_score(detection_result):
  person_detection_scores = [0.] * 4
  for i in range(detection_result.num):
    index = person_names.index(detection_result.classes[i])
    if index >= 0:
      person_detection_scores[index] = detection_result.scores[i]
  return np.array(person_detection_scores)

ofile = FLAGS.input_file.replace('.txt', '.detect_person.txt')
print('infile:', FLAGS.input_file, 'ofile:',ofile, file=sys.stderr)
names = ['detection_person_score']

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
    person_detection_scores = get_hat_score(detection_result)
    person_caption_scores = [0.] * 4 
    
    for i, item in enumerate(person_names):
      if item in caption:
        person_caption_scores[0] = 1.0 
    person_caption_scores = np.array(person_caption_scores)

    score = gezi.cosine(person_detection_scores, person_caption_scores)

    l += map(str, [score])
    print('\t'.join(l), file=out)
      

