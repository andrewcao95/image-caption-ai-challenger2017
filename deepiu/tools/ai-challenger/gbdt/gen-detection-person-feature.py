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

detection_file = '/home/gezi/mine/mount/temp/image-caption/ai-challenger/detection/openimage/faster-rcnn.lowproposals/valid.pkl'

def get_person_score(detection_result):
  person_detection_scores = [0.] * 4
  num_matchs = 0
  for i in range(detection_result.num):
    index = -1
    
    # FIXME >= 545 ?
    try:
      class_name = class_names[int(detection_result.classes[i])] 
      if class_name in person_names:
        index = person_names.index(class_name)
    except Exception:
      pass

    # therer might be many match detection for one class like girl, here just choose the max prob one
    if index >= 0 and not person_detection_scores[index]:
      person_detection_scores[index] = detection_result.scores[i]
      num_matchs += 1
      if num_matchs == len(person_detection_scores):
        break
  return np.array(person_detection_scores)

ofile = FLAGS.input_file.replace('.txt', '.detect_person.txt')
print('infile:', FLAGS.input_file, 'ofile:',ofile, file=sys.stderr)
#names = ['detection_person_score']
names = ['detect_man', 'detect_woman', 'detect_boy', 'detect_girl']
names += ['caption_man', 'caption_woman', 'caption_boy', 'caption_girl']


with open(detection_file) as f:
  detection_results = pickle.load(f)

with open(ofile, 'w') as out:
  is_header = True
  for line in open(FLAGS.input_file):
    l = line.strip().split('\t')
    if is_header:
      names = l + names 
      print('\t'.join(names), file=out)
      with open('./feature_name.txt', 'w') as out:
        for name in names[3:]:
          print(name, file=out)
      is_header = False
      continue
    img, caption = l[0], l[1]
    
    detection_result = detection_results[img]
    person_detection_scores = get_person_score(detection_result)
    person_caption_scores = [0.] * 4 
    
    for i, item in enumerate(person_names):
      if item in caption:
        person_caption_scores[i] = 1.0 
    person_caption_scores = np.array(person_caption_scores)

    # if using cosine need to add additional 1 maybe to avoid all zero [0, 0, 0, 0] nan
    #score = gezi.dist(person_detection_scores, person_caption_scores)
    
    # if img == '673ab73ecc9f4fee5e0616d6e645d7485c1d596a':
    #   print('should be 女人 only')
    #   print(person_detection_scores, person_caption_scores, score)
      
    #l += map(str, [score])

    l += map(str, person_detection_scores)
    l += map(str, person_caption_scores)
    print('\t'.join(l), file=out)
      

