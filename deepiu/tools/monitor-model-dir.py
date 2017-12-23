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

flags.DEFINE_string('model_dir', '/home/gezi/new/temp/image-caption/ai-challenger/model/showandtell', '')
flags.DEFINE_string('vocab', '/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic/vocab.txt', '')
flags.DEFINE_integer('start_epoch', 0, '')
flags.DEFINE_integer('epoch_interval', 1, '')
flags.DEFINE_bool('write_step', True, 'if False will write epoch')

import sys, os, glob, time, traceback
import pickle

import gezi, melt 
from deepiu.util import evaluator
from deepiu.util.text_predictor import TextPredictor

logging = melt.logging

def main(_):
  print('eval_rank:', FLAGS.eval_rank, 'eval_translation:', FLAGS.eval_translation)
  #epoch_dir = os.path.join(FLAGS.model_dir, 'epoch')
  epoch_dir = FLAGS.model_dir

  logging.set_logging_path(gezi.get_dir(epoch_dir))

  log_dir = epoch_dir
  sess = tf.Session()
  summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
  
  Predictor = TextPredictor  

  image_model = None 
  if FLAGS.image_checkpoint_file and os.path.exists(FLAGS.image_checkpoint_file) \
    and not melt.has_image_model(FLAGS.model_dir, FLAGS.image_model_name):
    #feature_name = None, since in show and tell predictor will use gen_features not gen_feature
    image_model = melt.image.ImageModel(FLAGS.image_checkpoint_file, 
                                        FLAGS.image_model_name, 
                                        feature_name=None)
  print('image_model:', image_model)
  #hack here! TODO
  if melt.has_image_model(FLAGS.model_dir, FLAGS.image_model_name): 
    FLAGS.pre_calc_image_feature = False
  else:
    has_image_model = FLAGS.image_checkpoint_file and os.path.exists(FLAGS.image_checkpoint_file)
    FLAGS.pre_calc_image_feature = FLAGS.pre_calc_image_feature or (not has_image_model)
  print('pre_calc_image_feature:', FLAGS.pre_calc_image_feature)

  evaluator.init(image_model)

  while True:
    #each time reload so we can run two monitor to same path at the same time
    visited_path = os.path.join(epoch_dir, 'visited.pkl')
    if not os.path.exists(visited_path):
      visited_checkpoints = set()
    else:
      visited_checkpoints = pickle.load(open(visited_path, 'rb'))
      
    visited_checkpoints = set([x.split('/')[-1] for x in visited_checkpoints])
  
    suffix = '.data-00000-of-00001'
    files = glob.glob(os.path.join(epoch_dir, 'model.ckpt*.data-00000-of-00001'))
    #from epoch 1, 2, ..
    #files.sort(key=os.path.getmtime)
    files = [file.replace(suffix, '') for file in files if not 'best' in file.split('/')[-1]]
    epochs = [int(float(file.split('/')[-1].split('-')[-2])) for file in files]
    #m = dict([(x, y) for (x, y) in zip(epochs, files)])
    files = [x for _, x in sorted(zip(epochs, files))]
    #print('--', files)
    for i, file in enumerate(files):
      file_ = file.split('/')[-1]
      if file_ not in visited_checkpoints:
        epoch = int(float(file_.split('-')[-2]))
        step = int(float(file_.split('-')[-1]))
        if FLAGS.start_epoch and epoch < FLAGS.start_epoch:
          continue
        if FLAGS.epoch_interval and epoch % FLAGS.epoch_interval != 0:
          continue
        logging.info('mointor_epoch:%d model:%s, %d model files has been done'%(epoch, file_, len(visited_checkpoints)))
        #will use predict_text in eval_translation , predict in eval_rank
        try:
          predictor = Predictor(file, image_model=image_model, feature_name=melt.get_features_name(FLAGS.image_model_name)) 
        except Exception:
          print(traceback.format_exc(), file=sys.stderr)
          continue
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
          visited_checkpoints.add(file_)
          pickle.dump(visited_checkpoints, open(visited_path, 'wb'))
    time.sleep(5)

 
if __name__ == '__main__':
  tf.app.run()
