#!/usr/bin/env python
# ==============================================================================
#          \file   monitor_epoch.py
#        \author   chenghuige  
#          \date   2017-09-19 12:14:39.924016
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, glob, time, traceback
import pickle

import gezi, melt 
from deepiu.util import evaluator
from deepiu.util.text_predictor import TextPredictor

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', None, '')
flags.DEFINE_string('vocab', '/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic/vocab.txt', '')
flags.DEFINE_bool('write_step', True, 'if False will write epoch')

logging = melt.logging

def main(_):

  print('eval_rank:', FLAGS.eval_rank, 'eval_translation:', FLAGS.eval_translation) 

  model_dir = FLAGS.model_dir or sys.argv[1]
  model_dirs = None

  if ',' in model_dir:
    model_dirs = model_dir.split(',')
    model_dir = None
  print(model_dirs, model_dir)  
  model_dir_ = gezi.get_dir(model_dir or model_dirs[0])
  log_dir = model_dir_
  log_dir = log_dir or './'
  print('model log dir:', log_dir)
  logging.set_logging_path(log_dir)

  sess = tf.Session()
  summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
  
  Predictor = TextPredictor  

  image_model = None 
  if FLAGS.image_checkpoint_file and os.path.exists(FLAGS.image_checkpoint_file) \
    and not melt.has_image_model(model_dir, FLAGS.image_model_name):
    #feature_name = None, since in show and tell predictor will use gen_features not gen_feature
    image_model = melt.image.ImageModel(FLAGS.image_checkpoint_file, 
                                        FLAGS.image_model_name, 
                                        feature_name=None)
  print('image_model:', image_model)
  #hack here! TODO
  if melt.has_image_model(model_dir or model_dirs[0], FLAGS.image_model_name): 
    FLAGS.pre_calc_image_feature = False
  else:
    has_image_model = FLAGS.image_checkpoint_file and os.path.exists(FLAGS.image_checkpoint_file)
    FLAGS.pre_calc_image_feature = FLAGS.pre_calc_image_feature or (not has_image_model)
  print('pre_calc_image_feature:', FLAGS.pre_calc_image_feature)

  evaluator.init(image_model)

  checkpoint = None
  checkpoints = None
  if not model_dirs:
    checkpoint = melt.get_model_path(model_dir)
    epoch = melt.get_model_epoch(checkpoint)
    step = melt.get_model_step(checkpoint)
  else:
    checkpoints = [melt.get_model_path(model_dir) for model_dir in model_dirs]
    epoch = melt.get_model_epoch(checkpoints[0])
    step = melt.get_model_step(checkpoints[0])

  logging.info('epoch:{}, step:{}, checkpoint:{}'.format(epoch, step, checkpoints or checkpoint))

  predictor = Predictor(checkpoints or checkpoint, image_model=image_model)
  
  melt.set_global('epoch', epoch)
  melt.set_global('step', step)
  scores, metrics = evaluator.evaluate(predictor, eval_rank=FLAGS.eval_rank, eval_translation=FLAGS.eval_translation)

  summary = tf.Summary()
  if scores:
    melt.add_summarys(summary, scores, metrics, prefix='epoch')
    summary_writer.add_summary(summary, int(epoch))
    summary_writer.flush()

 
if __name__ == '__main__':
  tf.app.run()
