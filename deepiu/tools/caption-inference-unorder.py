#!/usr/bin/env python
# ==============================================================================
#          \file   caption-inference.py
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
flags.DEFINE_string('test_image_dir', None, '') 
flags.DEFINE_integer('buffer_size', 512, '')

flags.DEFINE_string('result_file', None, '')

flags.DEFINE_boolean('full_text', False, '')

flags.DEFINE_string('text_key', None, '')
flags.DEFINE_string('score_key', None, '')

flags.DEFINE_float('current_length_normalization_factor', None, '')
flags.DEFINE_float('length_normalization_fator', None, '')

import sys, os, glob 
import random

import melt 

import numpy as np

from deepiu.util.text_predictor import TextPredictor

from deepiu.util import ids2text

def predict(predictor, imgs, file):
  if not FLAGS.full_text:
    texts_list, scores_list = predictor.predict_text( \
          np.array([melt.read_image(img_path) for img_path in imgs]), FLAGS.text_key, FLAGS.score_key)
  else:
    texts_list, scores_list = predictor.predict_full_text(np.array([melt.read_image(img_path) for img_path in imgs]))
  best_texts = [ids2text.translate(texts[0]) for texts in texts_list]
  best_scores = [scores[0] for scores in scores_list]
  all_texts = []
  all_scores = []
  for texts in texts_list:
    texts = [ids2text.translate(text) for text in texts]
    texts = ' '.join(texts)
    all_texts.append(texts)
  for scores in scores_list:
    scores = [str(score) for score in scores]
    scores = ' '.join(scores)
    all_scores.append(scores)

  for img, best_text, best_score, all_text, all_score in zip(imgs, best_texts, best_scores, all_texts, all_scores):
    print(os.path.basename(img).split('.')[0], best_text, best_score, all_text, all_score, sep='\t', file=file)

def main(_):
  model_dir = FLAGS.model_dir or sys.argv[1]
  assert model_dir
  model_path = melt.get_model_path(model_dir)
  print('model_path:', model_path, file=sys.stderr) 


  result_file = FLAGS.result_file or model_path + '.inference.txt'
  print('result file is:', result_file, file=sys.stderr)
  out = open(result_file, 'w')

  Predictor = TextPredictor
  image_model = None
  image_checkpoint_file = FLAGS.image_checkpoint_file or '/home/gezi/data/image_model_check_point/inception_resnet_v2_2016_08_30.ckpt'
  image_model_name = FLAGS.image_model_name or melt.image.get_imagenet_from_checkpoint(image_checkpoint_file).name
  if not melt.has_image_model(model_dir, image_model_name):
    image_model = melt.image.ImageModel(image_checkpoint_file)
  print('image_model:', image_model, file=sys.stderr)
  predictor = Predictor(model_dir, image_model=image_model, vocab_path=FLAGS.vocab,
                        current_length_normalization_factor=FLAGS.current_length_normalization_factor,
                        length_normalization_fator=FLAGS.length_normalization_fator) 

  imgs = []

  files = glob.glob(FLAGS.test_image_dir + '/*')
  random.shuffle(files)
  print('files[0]', files[0])

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
    predict(predictor, imgs, out)
    imgs = []


if __name__ == '__main__':
  tf.app.run()
