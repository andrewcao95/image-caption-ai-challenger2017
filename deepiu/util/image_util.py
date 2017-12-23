#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   image_util.py
#        \author   chenghuige  
#          \date   2017-11-11 19:06:24.907563
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import melt
logging = melt.logging

import tensorflow as tf
flags = tf.app.flags
#import gflags as flags
FLAGS = flags.FLAGS 

def init():
  model_dir = FLAGS.model_dir 
  net = melt.image.get_imagenet_from_checkpoint(FLAGS.image_checkpoint_file)
  #assert net is not None, FLAGS.image_checkpoint_file
  assert net is not None or FLAGS.image_model_name
  print('image net', net, 'net.default_image_size', net.default_image_size if net else None)
  FLAGS.image_model_name = FLAGS.image_model_name or net.name
  FLAGS.image_height = FLAGS.image_height or net.default_image_size if net else melt.image.image_processing.info[FLAGS.image_model_name]['height']
  FLAGS.image_width = FLAGS.image_width or net.default_image_size if net else melt.image.image_processing.info[FLAGS.image_model_name]['width']
  FLAGS.image_attention_size = FLAGS.image_attention_size or melt.image.get_num_features(FLAGS.image_model_name)
  if FLAGS.image_endpoint_feature_name is None:
    FLAGS.image_attention_size = 1
  FLAGS.image_feature_len = FLAGS.image_feature_len or melt.image.get_feature_dim(FLAGS.image_model_name) * FLAGS.image_attention_size
  print('image_endpoint_feature_name', FLAGS.image_endpoint_feature_name)
  print('image_model_name', FLAGS.image_model_name, 'height', FLAGS.image_height, 'width', FLAGS.image_width, file=sys.stderr)
  print('image_attention_size', FLAGS.image_attention_size, 'image_feature_len', FLAGS.image_feature_len)
  image_model_name = FLAGS.image_model_name 
  image_checkpoint_file = FLAGS.image_checkpoint_file
  image_endpoint_feature_name = FLAGS.image_endpoint_feature_name
  pre_calc_image_feature = FLAGS.pre_calc_image_feature

  if melt.has_image_model(model_dir, image_model_name): 
    print('model_dir:%s already has image mode %s inside, force pre_calc_image_feature to False' \
        %(model_dir, image_model_name), file=sys.stderr)
    pre_calc_image_feature = False
  else:
    print('model_dir:%s not has image mode inside'%model_dir)
    has_image_model = image_checkpoint_file and melt.checkpoint_exists(image_checkpoint_file)
    print('outside model_dir has_image_model:', has_image_model, file=sys.stderr)
    assert not (image_checkpoint_file and not melt.checkpoint_exists(image_checkpoint_file)), \
      'set to use image_checkpoint_file%s but could not find it'%image_checkpoint_file
    if pre_calc_image_feature is None:
      pre_calc_image_feature = not has_image_model

  if not pre_calc_image_feature:
    melt.apps.image_processing.init(image_model_name, num_classes=FLAGS.num_pretrain_image_classes)
  # else:
  #   assert False, 'must has image_model for classify'

  print('pre_calc_image_feature:', pre_calc_image_feature, file=sys.stderr)
  FLAGS.pre_calc_image_feature = pre_calc_image_feature

  return image_model_name


def get_init_restore_fn():
  init_fn = None 
  restore_fn = None
  if not FLAGS.pre_calc_image_feature:
    if FLAGS.image_checkpoint_file:
      init_fn = melt.image.image_processing.create_image_model_init_fn(FLAGS.image_model_name, FLAGS.image_checkpoint_file)
      if melt.checkpoint_exists_in(FLAGS.model_dir):
        if not melt.varname_in_checkpoint(FLAGS.image_model_name, FLAGS.model_dir):
          restore_fn = init_fn
          logging.info('Model has no image model inside will init/restore image model from image checkpoint file %s' % FLAGS.image_checkpoint_file)
        else:
          logging.info('Model %s has image model inside' % FLAGS.model_dir)
      else:
        logging.info('No checkpoint in %s' % FLAGS.model_dir)
    else:
      logging.info('No image model finetune, train image model from scratch')
  return init_fn, restore_fn
