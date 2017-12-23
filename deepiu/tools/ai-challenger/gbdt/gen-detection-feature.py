#!/usr/bin/env python
#encoding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', 'ensemble.train.txt', '')
flags.DEFINE_string('type', 'valid', 'valid or test')
flags.DEFINE_string('detector', 'faster-rcnn', 'faster-rcnn.lowproposals or faster-rcnn')

import sys
import cPickle as pickle
import numpy as np

import gezi

from deepiu.util.detector import Detector

print('type', FLAGS.type)

if FLAGS.type == 'test':
  FLAGS.input_file = './ensemble.inference.feature.txt'

detection_file = '/home/gezi/mine/mount/temp/image-caption/ai-challenger/detection/openimage/%s/%s.pkl' % (FLAGS.detector, FLAGS.type)

print('detection_file', detection_file)

class_names = open('/home/gezi/mine/hasky/deepiu/object_detection/data/oid_label_caption.txt').readlines()
class_names = [x.strip() for x in class_names]
print(class_names[0], class_names[1], class_names[10], 'num_classes:', len(class_names))
assert(len(class_names) == 545)
class_names += ['None'] * 10000

hat_names = set(['帽子', '太阳帽', '牛仔帽', '草帽'])

hat_ids = []
for i in range(len(class_names)):
  if class_names[i] in hat_names:
    hat_ids.append(i) 
hat_ids = set(hat_ids)

print('hat_ids', hat_ids, file=sys.stderr)

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

person_names = ['人', '男人', '女人', '男孩', '女孩', '孩子']

def get_person_score(detection_result):
  person_detection_scores = [0.] * 5
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
  return person_detection_scores

def get_person_pos(detection_result):
  person_pos = [0.] * 4
  for i in range(detection_result.num):
    class_name = class_names[int(detection_result.classes[i])] 
    if class_name in person_names:
      person_pos = detection_result.boxes[i]
      return person_pos
  return person_pos

def get_face_pos(detection_result):
  face_pos = [0.] * 4
  for i in range(detection_result.num):
    class_name = class_names[int(detection_result.classes[i])] 
    if class_name == '脸':
      face_pos = detection_result.boxes[i]
      return face_pos
  return face_pos

def get_num_persons(detection_result):
  persons_score = [0.] * 4
  num_persons = 0
  for i in range(detection_result.num):
    class_name = class_names[int(detection_result.classes[i])]  
    if class_name == '人':
      persons_score[num_persons] = detection_result.scores[i]
      num_persons += 1 
      if num_persons == len(persons_score):
        break 
  return persons_score

def get_num_faces(detection_result):
  faces_score = [0.] * 4
  num_faces = 0
  for i in range(detection_result.num):
    class_name = class_names[int(detection_result.classes[i])]  
    if class_name == '脸':
      faces_score[num_faces] = detection_result.scores[i]
      num_faces += 1 
      if num_faces == len(faces_score):
        break 
  return faces_score

# TODO no detection result for phone
# def get_phone_score(detection_result):
#   score = 0.
#   MAX_INDEX = 1000
#   pos = MAX_INDEX
#   for i in range(detection_result.num):
#     if class_names[int(detection_result.classes[i])] == '手机' :
#       score = detection_result.scores[i]
#       pos = i
#       break
#   return score, pos

ofile = FLAGS.input_file.replace('.txt', '.detect.txt')
print('infile:', FLAGS.input_file, 'ofile:',ofile, file=sys.stderr)
names = []
names += ['detect_has_hat', 'detect_hat_score', 'detect_hat_pos']
names += ['detect_has_person', 'detect_has_man', 'detect_has_woman', 'detect_has_boy', 'detect_has_girl', 'detect_has_child']
names += ['detect_person', 'detect_man', 'detect_woman', 'detect_boy', 'detect_girl']
# names += ['detect_has_phone', 'detect_phone_score', 'detect_phone_pos']
names += ['detect_has_lhand', 'detect_has_rhand']
names += ['detect_person_pos0', 'detect_person_pos1', 'detect_person_pos2', 'detect_person_pos3']
names += ['detect_face_pos0', 'detect_face_pos1', 'detect_face_pos2', 'detect_face_pos3']
names += ['detect_num_persons0', 'detect_num_persons1', 'detect_num_persons2', 'detect_num_persons3']
names += ['detect_num_faces0', 'detect_num_faces1', 'detect_num_faces2', 'detect_num_faces3']

timer = gezi.Timer('load detection_file', True)
with open(detection_file) as f:
  detection_results = pickle.load(f)
timer.print()

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
    
    detection_result = detection_results[img]
    
    #---hat
    has_hat = 0.
    for item in hat_names:
      if item in caption:
        has_hat = 1.

    score, pos = get_hat_score(detection_result)
    l += map(str, [has_hat, score, pos])

    #---person
    person_caption_stats = [0.] * len(person_names)
    for i, item in enumerate(person_names):
      if item in caption:
        person_caption_stats[i] = 1.0 

    person_detection_scores = get_person_score(detection_result) 

    l += map(str, person_caption_stats)
    l += map(str, person_detection_scores)

    # has_phone = float('手机' in caption)
    # score, pos = get_phone_score(detection_result)
    # l += map(str, [has_phone, score, pos])


    has_lhand = float('左手' in caption)
    has_rhand = float('右手' in caption)
    l += map(str, [has_lhand, has_rhand])

    l += map(str, get_person_pos(detection_result))
    l += map(str, get_face_pos(detection_result))

    l += map(str, get_num_persons(detection_result))
    l += map(str, get_num_faces(detection_result))

    print('\t'.join(l), file=out)
      

