#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-attention-feature-permodel.py
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
import melt

import tempfile

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', None, '')
flags.DEFINE_string('type', 'valid', 'valid or test')
flags.DEFINE_string('img_dir', '/home/gezi/data2/data/ai_challenger/image_caption/pic/', '')
flags.DEFINE_integer('img_id', 0, '')

ofile = FLAGS.input_file.replace('.txt', '.feature.txt')
print('infile:', FLAGS.input_file, 'ofile:',ofile, file=sys.stderr)

#only consider the first occurence of letf hand and right hand
names = []
names += ['coverage_loss']

names += ['lhand_entropy', 'lhand_peak_height', 'lhand_peak_width', \
         'rhand_entropy', 'rhand_peak_height', 'rhand_peak_width', \
         'ls_entropy', 'ls_peak_height', 'ls_peak_width', \
         'rs_entropy', 'rs_peak_height', 'rs_peak_width']

names = ['attention_' + x for x in names]

timer = gezi.Timer('load segged caption file')
with open('./segged_caption.pkl') as f:
  segged_captions = pickle.load(f)
timer.print()

def get_hand_feature(alignment):
  entropy = gezi.probs_entropy(alignment)
  peak_pos = alignment.argmax(-1)
  attention_size = alignment.shape[-1]
  dim = int(np.sqrt(attention_size))
  peak_height = (peak_pos / dim) / dim
  peak_width = (peak_pos % dim) / dim

  return entropy, peak_height, peak_width

def set_hand_feature(fe, alignment, index):
  entropy, peak_height, peak_width = get_hand_feature(alignment)
  fe[index] = entropy
  fe[index + 1] = peak_height
  fe[index + 2] = peak_width 

sess = tf.InteractiveSession()
def calc_imtxt_sim(image_path, text, obj):
  text = text.replace(' ', '')
  image = melt.read_image(image_path)
  image=tf.image.decode_jpeg(image)
  #image = tf.image.resize_images(image, size=(img.shape[0], img.shape[1]))
  image = tf.image.crop_to_bounding_box(image, obj[0].start, obj[1].start, obj[0].stop - obj[0].start, obj[1].stop - obj[1].start)
  return np.squeeze(predictor.predict([image], [text2ids.text2ids(text)]))

with open(ofile, 'w') as out:
  print('\t'.join(['#img', 'caption'] + names), file=out)

  for i, line in enumerate(open(FLAGS.input_file)):
    if i % 1000 == 0:
      print(i, end='\r', file=sys.stderr)
    img, caption, num_steps, attention_size, alignment = line.strip().split('\t')

    fe = [0.] * len(names)
    num_steps = int(num_steps)
    attention_size = int(attention_size)
    dim = int(np.sqrt(attention_size))
    alignment = np.reshape(np.array(map(float, alignment.split())), [num_steps, attention_size])
    words = segged_captions[caption]

    # the lower the better
    coverage_loss = np.sum((alignment.sum(0) - num_steps / attention_size) ** 2)
    fe[0] += coverage_loss
    lhand_count = words.count('左手')
    rhand_count = words.count('右手')
    ls_count = words.count('左肩')
    rs_count = words.count('右肩')
    if lhand_count == 1 or rhand_count == 1 or ls_count == 1 or rs_count == 1:
      for i, word in enumerate(words):
        if word == '左手' and lhand_count == 1:
          set_hand_feature(fe, alignment[i], 1)
        elif  word == '右手' and rhand_count == 1:
          set_hand_feature(fe, alignment[i], 4)
        elif word == '左肩' and ls_count == 1:
          set_hand_feature(fe, alignment[i], 7)
        elif word == '右肩' and rs_count == 1:
          set_hand_feature(fe, alignment[i], 10)
    
    l = map(str, fe)
    print(img, ' '.join(words), '\t'.join(l), sep='\t', file=out)


  