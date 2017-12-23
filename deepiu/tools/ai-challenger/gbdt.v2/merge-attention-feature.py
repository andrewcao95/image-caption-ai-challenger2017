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

attention_files = glob.glob('./*model*.evaluate-inference.attention.feature.txt') if FLAGS.type == 'valid' \
              else glob.glob('./*model*.inference.attention.feature.txt')

assert len(attention_files) == 19

#only consider the first occurence of letf hand and right hand
names = []
names += ['num_persons', 'img_height', 'img_width', 'img_hw_ratio']
names += ['coverage_loss']

names += ['lhand_entropy', 'lhand_peak_height', 'lhand_peak_width', 'lobj_entropy', 'lobj_peak_width', 'lobj_peak_height', 'lobj_dist', \
         'rhand_entropy', 'rhand_peak_height', 'rhand_peak_width', 'robj_entropy', 'robj_peak_width', 'robj_peak_height', 'robj_dist', \
         'ls_entropy', 'ls_peak_height', 'ls_peak_width', 'lsobj_entropy', 'lsobj_peak_width', 'lsobj_peak_height', 'lsobj_dist',\
         'rs_entropy', 'rs_peak_height', 'rs_peak_width', 'rsobj_entropy', 'rsobj_peak_width', 'rsobj_peak_height', 'rsobj_dist']

names = ['attention_' + x for x in names]

m = {}
count_map = {}
image_info = {}

def get_num_persons(segged_caption):
  return segged_caption.count('一个') + segged_caption.count('两个') * 2 \
        + segged_caption.count('三个') * 3 + segged_caption.count('四个') * 4 \
        + segged_caption.count('五个') * 5 \
        + segged_caption.count('一群') * 6


def deal_attention(file):
  for i, line in enumerate(open(file)):
    if i == 0:
      continue
    if i % 1000 == 0:
      print(i, end='\r', file=sys.stderr)
    l = line.strip().split('\t')
    img, caption = l[0], l[1]
    words = caption.split()
    caption = caption.replace(' ', '')
    atten_fe = l[2:]
    atten_fe = map(float, atten_fe)
    #print('-------', len(atten_fe), len(names))
    if img not in m:
      m[img] = {}
      count_map[img] = {}
    if caption not in m[img]:
      m[img][caption] = np.array([0.] * len(names))
      count_map[img][caption] = 0
    fe = m[img][caption]
    count_map[img][caption] += 1
    
    fe[0] = get_num_persons(words)
    
    image_path = os.path.join(FLAGS.img_dir, img + '.jpg')
    if img not in image_info:
      img_data = ndimage.imread(image_path)
      
      img_height = img_data.shape[0]
      img_width = img_data.shape[1]
      img_hw_ratio = img_height / img_width 
      fe[1], fe[2], fe[3] = img_height, img_width, img_hw_ratio
      image_info[img] = {}
      image_info[img]['height'] = img_height
      image_info[img]['width'] = img_width
      image_info[img]['hw_ratio'] = img_hw_ratio
    else:
      fe[1], fe[2], fe[3] = image_info[img]['height'], image_info[img]['width'], image_info[img]['hw_ratio']
    index = 4
    for i in range(len(atten_fe)):
      fe[index + i] += atten_fe[i]

for i, file in enumerate(attention_files):
  timer = gezi.Timer('deal %d %s' % (i, file))
  deal_attention(file)
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
      for i in range(4, len(fe)):
        fe[i] /= count_map[img][caption]
    else:
      fe = default_fe
    l += map(str, fe)
    print('\t'.join(l), file=out)
  timer.print()

      
