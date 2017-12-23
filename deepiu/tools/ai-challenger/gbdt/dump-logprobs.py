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
logprobs_file = './%s.%s.logprobs.pkl' % (model, FLAGS.type)

output_file = input_file.replace('.txt', '.logprobs.txt')

print('infile:', input_file, 'ofile:', output_file)

timer = gezi.Timer('load logprobs file')
with open(logprobs_file) as f:
  logprobs = pickle.load(f)
timer.print()

timer = gezi.Timer('load segged caption')
with open('./segged_caption.pkl') as f:
  segged_captions = pickle.load(f)
timer.print()

with open(output_file, 'w') as out:
  for i, line in enumerate(open(input_file)):
    if i % 100 == 0:
      print('%d pics done' % i, end='\r', file=sys.stderr)
    l = line.strip().split('\t')
    img, captions = l[0], l[-2] 
    for caption in captions.split(' '):
      logprob = logprobs[img][caption]
      words = segged_captions[caption]
      logprob = logprob[:len(words) + 1]
      logprob = ' '.join(map(str, logprob))
      caption = ' '.join(words)
      print(img, caption, logprob, sep='\t', file=out)
  


