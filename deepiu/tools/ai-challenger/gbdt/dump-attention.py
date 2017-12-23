#!/usr/bin/env python
#encoding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('type', 'valid', 'valid or test')
flags.DEFINE_string('model', None, '')

import sys
import cPickle as pickle
import numpy as np

import gezi 

print('type', FLAGS.type)

model = FLAGS.model or sys.argv[1]

input_file = '%s.evaluate-inference.txt' % model if FLAGS.type == 'valid' else '%s.inference.txt' % model
attention_file = './%s.%s.alignments.pkl' % (model, FLAGS.type)

output_file = input_file.replace('.txt', '.attention.txt')

print('infile:', input_file, 'ofile:', output_file)

timer = gezi.Timer('load attention file')
with open(attention_file) as f:
  alignments = pickle.load(f)
timer.print()

with open(output_file, 'w') as out:
  for i, line in enumerate(open(input_file)):
    if i % 100 == 0:
      print('%d pics done' % i, end='\r', file=sys.stderr)
    l = line.strip().split('\t')
    img, captions = l[0], l[-2]
    for caption in captions.split(' '):
      alignment = alignments[img][caption]
      num_alignments = alignment.shape[0]
      alignments_length = alignment.shape[1]
      alignment = np.reshape(alignment, -1)
      alignment = ' '.join(map(str, alignment))
      print(img, caption, num_alignments, alignments_length, alignment, sep='\t', file=out)
  


