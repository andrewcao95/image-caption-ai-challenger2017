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

import skimage.transform
from scipy import ndimage

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', 'ensemble.train.detect.scene.lm.txt', '')
flags.DEFINE_string('type', 'valid', 'valid or test')
flags.DEFINE_string('img_dir', '/home/gezi/data2/data/ai_challenger/image_caption/pic/', '')

print('type', FLAGS.type)
if FLAGS.type == 'test':
  FLAGS.input_file = './ensemble.inference.feature.detect.scene.lm.txt'
ofile = FLAGS.input_file.replace('.txt', '.attention.txt')
print('infile:', FLAGS.input_file, 'ofile:',ofile, file=sys.stderr)

attention_files = glob.glob('./*model*.evaluate-inference.attention.txt') if FLAGS.type == 'valid' \
              else glob.glob('./*model*.inference.attention.txt')

#only consider the first occurence of letf hand and right hand
names = ['num_persons', \
         'img_height', 'img_width', 'img_hw_ratio', \
         'coverage_loss', \
         'lhand_entropy', 'lhand_peak_height', 'lhand_peak_width', 'lhand_box_start0', 'lhand_box_stop0', 'lhand_box_start1', 'lhand_box_stop1', 'lhand_box_center_height', 'lhand_box_center_width', \
         'rhand_entropy', 'rhand_peak_height', 'rhand_peak_width', 'rhand_box_start0', 'rhand_box_stop0', 'rhand_box_start1', 'rhand_box_stop1', 'rhand_box_center_height', 'rhand_box_center_height']
names = ['attention_' + x for x in names]

timer = gezi.Timer('load segged caption file')
with open('./segged_caption.pkl') as f:
  segged_captions = pickle.load(f)
timer.print()

m = {}
count_map = {}

def get_num_persons(segged_caption):
  return segged_caption.count('一个') + segged_caption.count('两个') * 2 \
        + segged_caption.count('三个') * 3 + segged_caption.count('四个') * 4 \
        + segged_caption.count('五个') * 5 \
        + segged_caption.count('一群') * 6

def get_hand_feature(img, alignment):
  entropy = gezi.probs_entropy(alignment)
  peak_pos = alignment.argmax(-1)
  attention_size = alignment.shape[-1]
  dim = int(np.sqrt(attention_size))
  peak_height = (peak_pos % dim) / dim + 1 
  peak_width = (peak_pos / dim) / dim + 1

  # img_height = img.shape[0]
  # img_width = img.shape[1]
  # alpha_img = skimage.transform.resize((alignment > 0.005).reshape(dim, dim), [img_width, img_height])
  # objs = ndimage.find_objects(alpha_img.astype(int))
  # if not objs:
  box_start0, box_stop0 = 0., 0.
  box_start1, box_stop1 = 0., 0. 
  box_center_height, box_center_width = 0., 0.
  # else:
  #   obj = objs[0]
  #   box_start0, box_stop0 = obj[0].start / img_height, obj[0].stop / img_height
  #   box_start1, box_stop1 = obj[1].start / img_width, obj[1].stop / img_width   
  #   box_center_height, box_center_width =  (box_start0 + box_stop0) / 2 / img_height,  (box_start0 + box_stop0) / 2 / img_width
  return entropy, peak_height, peak_width, box_start0, box_stop0, box_start1, box_stop1, box_center_height, box_center_width

def set_hand_feature(fe, img, alignment, index):
  entropy, peak_height, peak_width, box_start0, box_stop0, box_start1, box_stop1, box_center_height, box_center_width  = get_hand_feature(img, alignment)
  fe[index] += entropy
  fe[index + 1] += peak_height
  fe[index + 2] += peak_width 
  fe[index + 3] += box_start0
  fe[index + 4] += box_stop0
  fe[index + 5] += box_start1
  fe[index + 6] += box_stop1
  fe[index + 7] += box_center_height
  fe[index + 8] += box_center_width


def deal_attention(file):
  pre_img = None
  img_data = None
  for i, line in enumerate(open(file)):
    if i % 1000 == 0:
      print(i, end='\r', file=sys.stderr)
    img, caption, num_steps, attention_size, alignment = line.strip().split('\t')
    if img not in m:
      m[img] = {}
      count_map[img] = {}
    if caption not in m[img]:
      m[img][caption] = np.array([0.] * len(names))
      count_map[img][caption] = 0
    fe = m[img][caption]
    count_map[img][caption] += 1
    num_steps = int(num_steps)
    attention_size = int(attention_size)
    dim = int(np.sqrt(attention_size))
    alignment = np.reshape(np.array(map(float, alignment.split())), [num_steps, attention_size])
    words = segged_captions[caption]

    fe[0] = get_num_persons(words)

    image_path = os.path.join(FLAGS.img_dir, img + '.jpg')
    if img != pre_img:
      img_data = ndimage.imread(image_path)
      pre_img = img
    img_height = img_data.shape[0]
    img_width = img_data.shape[1]
    img_hw_ratio = img_height / img_width 

    fe[1], fe[2], fe[3] = img_height, img_width, img_hw_ratio

    # the lower the better
    coverage_loss = np.sum((alignment.sum(0) - num_steps / attention_size) ** 2)
    fe[4] += coverage_loss
    lhand_count = words.count('左手')
    rhand_count = words.count('右手')
    if lhand_count == 1 or rhand_count == 1:
      for i, word in enumerate(words):
        if word == '左手' and lhand_count == 1:
          set_hand_feature(fe, img_data, alignment[i], 5)
        elif  word == '右手' and rhand_count == 1:
          set_hand_feature(fe, img_data, alignment[i], 14)

for i, file in enumerate(attention_files):
  timer = gezi.Timer('deal %d %s' % (i, file))
  deal_attention(file)
  timer.print()
  #if i == 2:
  #  break

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
      for i in range(4, len(fe)):
        fe[i] /= count_map[img][caption]
    else:
      fe = default_fe
    l += map(str, fe)
    print('\t'.join(l), file=out)
  timer.print()

      