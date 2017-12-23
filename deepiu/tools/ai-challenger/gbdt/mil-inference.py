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

import melt
import glob

from collections import namedtuple

from deepiu.util.sim_predictor import SimPredictor

import pickle

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', '/home/gezi/new/temp/image-caption/ai-challenger/model.v4/mil.idf.rnn2/epoch', '')
flags.DEFINE_string('type', 'valid', '')
flags.DEFINE_string('test_image_dir', '/data2/data/ai_challenger/ai_challenger_caption_validation_20170910/caption_validation_images_20170910/', '')
flags.DEFINE_integer('buffer_size', 1, '')
flags.DEFINE_string('result_file', None, '')

image_model_checkpoint_path = '/home/gezi/data/image_model_check_point/inception_resnet_v2.ckpt'

model_dir = FLAGS.model_dir

assert FLAGS.buffer_size == 1

m = {}
Result = namedtuple('Result', ['top_class', 'top_logit'])

image_model_name = melt.get_imagenet_from_checkpoint(image_model_checkpoint_path).name
image_model = None
if not melt.varname_in_checkpoint(image_model_name, model_dir):
  image_model = melt.image.ImageModel(image_model_checkpoint_path, 
                                      feature_name='attention')
      
print('image_model:', image_model)
predictor = SimPredictor(model_dir, image_model=image_model)

def predict(predictor, imgs):
  top_logits, top_classes = predictor.top_words(imgs, 200)
  for img, top_class, top_logit in zip(imgs, top_classes,top_logits):
    img = os.path.basename(img).replace('.jpg', '')
    if img not in m:
      result = Result(top_class, top_logit)
      m[img] = result

def main(_):
  model_dir = FLAGS.model_dir or sys.argv[1]
  assert model_dir
  model_path = melt.get_model_path(model_dir)
  print('model_path:', model_path, file=sys.stderr)

  result_file = './mil.%s.pkl' % FLAGS.type
  out = open(result_file, 'w')

  imgs = []
  
  print('type', FLAGS.type)
  if FLAGS.type == 'test':
    FLAGS.test_image_dir = '/data2/data/ai_challenger/ai_challenger_caption_test1_20170923/caption_test1_images_20170923/'

  print('test_image_dir', FLAGS.test_image_dir)

  files = glob.glob(FLAGS.test_image_dir + '/*')
  num_files = len(files)
  assert num_files, FLAGS.test_image_dir
  print('num_files to inference', num_files)
  finished = 0
  for img_ in files:
    imgs.append(img_)
    if len(imgs) == FLAGS.buffer_size:
      predict(predictor, imgs)
      finished += len(imgs)
      print('finished:%f' % (finished / float(num_files)), file=sys.stderr, end='\r')
      imgs = []

  if imgs:
    # HACK for nasnet
    while len(imgs) != FLAGS.buffer_size:
      imgs.append(imgs[0])

    predict(predictor, imgs)
    imgs = []

  pickle.dump(m, out)


if __name__ == '__main__':
  tf.app.run()
