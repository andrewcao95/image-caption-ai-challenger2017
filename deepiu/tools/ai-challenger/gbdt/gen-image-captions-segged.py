#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-image-captions-segged.py
#        \author   chenghuige  
#          \date   2017-12-03 13:53:41.347800
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import gezi

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('type', 'valid', 'valid or test')
  
timer = gezi.Timer('load segged caption file')
with open('./segged_caption.pkl') as f:
  segged_captions = pickle.load(f)
timer.print()

infile = './ensemble.evaluate-inference.txt' if FLAGS.type == 'valid' else './ensemble.inference.txt'
ofile = './%s.img-captions.segged.txt' % FLAGS.type

with open(ofile) as out:
  for line in open(infile):
    l = line.strip().split('\t')
    img, captions = l[0], l[2]
    captions = captions.split()
    captions = [' '.join(segged_captions[caption]) for caption in captions] 
    print(img, '\t'.join(captions), sep='\t', file=out)