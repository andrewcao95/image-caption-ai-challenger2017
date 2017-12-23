#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   read-dataset.py
#        \author   chenghuige  
#          \date   2017-12-22 15:56:00.221213
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time
import tensorflow as tf
from collections import defaultdict

import numpy as np

from gezi import Timer
import melt 
import gezi

from deepiu.util import text2ids

import input

flags = tf.app.flags
#import gflags as flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', '/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic/valid/test-*', '')


def dataset_input_fn():
  filenames = gezi.list_files(FLAGS.input)
  dataset = tf.data.TFRecordDataset(filenames)

  dataset = dataset.map(input.decode_example)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(32)
  #dataset = dataset.padded_batch(32, padded_shapes=[None])
  dataset = dataset.repeat(2)
  iterator = dataset.make_one_shot_iterator()

  ops = iterator.get_next()
  return ops

def main(_):
  FLAGS.pre_calc_image_feature = True

  ops = dataset_input_fn()
  print('---------------ops', ops)

  sess = tf.InteractiveSession()

  print(sess.run(ops))

if __name__ == '__main__':
  tf.app.run()
  
