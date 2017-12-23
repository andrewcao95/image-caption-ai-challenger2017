#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-bidirectional-label-map.py
#        \author   chenghuige  
#          \date   2016-10-07 22:19:40.675048
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  
flags.DEFINE_string('img2text', '', '')
flags.DEFINE_string('text2id', '', '')
flags.DEFINE_string('all_distinct_text_strs', '', '')

import sys
import numpy as np 

all_distinct_text_strs = np.load(FLAGS.all_distinct_text_strs) 
img2text = text2id = {}

for i, text in enumerate(all_distinct_text_strs):
  text2id[text] = i

for line in sys.stdin:
  l = line.rstrip().split('\t')
  img = l[0]
  texts = l[1:]
  if img not in img2text:
    img2text[img] = set()
  m = img2text[img]
  for text in texts:
    if text not in text2id:
      continue
    id = text2id[text]
    m.add(id)

np.save(FLAGS.img2text, img2text)
np.save(FLAGS.text2id, text2id)
