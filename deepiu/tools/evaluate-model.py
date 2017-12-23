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

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', None, '')
flags.DEFINE_string('vocab', '/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic/vocab.txt', '')
flags.DEFINE_bool('write_step', True, 'if False will write epoch')

import sys, os, glob, time, traceback
import pickle

import gezi, melt 
from deepiu.util import evaluator
from deepiu.util.text_predictor import TextPredictor

logging = melt.logging

def main(_):

  print('eval_rank:', FLAGS.eval_rank, 'eval_translation:', FLAGS.eval_translation) 

  model_dir = FLAGS.model_dir or sys.argv[1]

  model_dir_ = gezi.get_dir(model_dir)
  logging.set_logging_path(model_dir_)

  log_dir = model_dir_
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
  if melt.has_image_model(model_dir, FLAGS.image_model_name): 
    FLAGS.pre_calc_image_feature = False
  else:
    has_image_model = FLAGS.image_checkpoint_file and os.path.exists(FLAGS.image_checkpoint_file)
    FLAGS.pre_calc_image_feature = FLAGS.pre_calc_image_feature or (not has_image_model)
  print('pre_calc_image_feature:', FLAGS.pre_calc_image_feature)

  evaluator.init(image_model)

  checkpoint = melt.get_model_path(model_dir)
  epoch = melt.get_model_epoch(checkpoint)
  step = melt.get_model_step(checkpoint)
  logging.info('epoch:{}, step:{}, checkpoint:{}'.format(epoch, step, checkpoint))
  #will use predict_text in eval_translation , predict in eval_rank
  #predictor = Predictor(checkpoint, image_model=image_model, feature_name=melt.get_features_name(FLAGS.image_model_name)) 

  predictor = Predictor(checkpoint, image_model=image_model)
  summary = tf.Summary()
  melt.set_global('epoch', epoch)
  melt.set_global('step', step)
  scores, metrics = evaluator.evaluate(predictor, eval_rank=FLAGS.eval_rank, eval_translation=FLAGS.eval_translation)
  if scores:
    prefix = 'step' if FLAGS.write_step else 'epoch'
    melt.add_summarys(summary, scores, metrics, prefix=prefix)
    step = epoch if not FLAGS.write_step else step 
    summary_writer.add_summary(summary, step)
    summary_writer.flush()

 
if __name__ == '__main__':
  tf.app.run()
