#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   classification-inference.py
#        \author   chenghuige  
#          \date   2017-11-10 15:32:56.988358
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

from deepiu.util.classifier import Classifier

import melt

import glob

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', None, '')
flags.DEFINE_string('test_image_dir', None, '')
flags.DEFINE_integer('buffer_size', 512, '')
flags.DEFINE_string('result_file', None, '')

# HACK for nasnet
done_imgs = set()

def predict(predictor, imgs, out):
  result = predictor.predict(imgs)
  for img, top_class, top_logit, top_prediction, logit, prediction \
      in zip(imgs, result.top_classes,
             result.top_logits, result.top_predictions,
             result.logits, result.predictions):
    if img not in done_imgs:
      print(os.path.basename(img), ' '.join(map(str, top_class)),
            ' '.join(map(str, top_logit)), ' '.join(map(str, top_prediction)),
            ' '.join(map(str, logit)), ' '.join(map(str, prediction)), sep='\t', file=out)
      done_imgs.add(img)

def main(_):
  model_dir = FLAGS.model_dir or sys.argv[1]
  assert model_dir
  model_path = melt.get_model_path(model_dir)
  print('model_path:', model_path, file=sys.stderr)

  result_file = FLAGS.result_file or model_path + '.inference.txt'
  print('result file is:', result_file, file=sys.stderr)
  out = open(result_file, 'w')

  predictor = Classifier(model_dir)

  imgs = []

  files = glob.glob(FLAGS.test_image_dir + '/*')
  num_files = len(files)
  assert num_files, FLAGS.test_image_dir
  print('num_files to inference', num_files)
  finished = 0
  for img_ in files:
    imgs.append(img_)
    if len(imgs) == FLAGS.buffer_size:
      predict(predictor, imgs, out)
      finished += len(imgs)
      print('finished:%f' % (finished / float(num_files)), file=sys.stderr, end='\r')
      imgs = []

  if imgs:
    # HACK for nasnet
    while len(imgs) != FLAGS.buffer_size:
      imgs.append(imgs[0])

    predict(predictor, imgs, out)
    imgs = []


if __name__ == '__main__':
  tf.app.run()
