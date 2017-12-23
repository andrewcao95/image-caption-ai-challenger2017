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
flags.DEFINE_string('input_file', None, '')

flags.DEFINE_string('image_dir_', '/home/gezi/data2/data/ai_challenger/image_caption/pic', '') 
flags.DEFINE_integer('buffer_size', 512, '')
flags.DEFINE_integer('batch_size_', 512, '')

flags.DEFINE_string('result_file', None, '')

flags.DEFINE_boolean('full_text', False, '')

flags.DEFINE_string('text_key', None, '')
flags.DEFINE_string('score_key', None, '')

flags.DEFINE_string('feature_name', None, '')

import sys
import os
import glob

import melt

import numpy as np

from deepiu.util.sim_predictor import SimPredictor

from deepiu.util import ids2text
from deepiu.util import text2ids
from deepiu.util.text2ids import texts2ids

import operator

text2ids.init()

num_imgs_done = 0

batch_size = FLAGS.batch_size_

predicted_imgs = set()
done_imgs = set()
out = None
def _predict(predictor, imgs, texts_list, m):
  global num_imgs_done
  raw_imgs = [melt.read_image(os.path.join(
      FLAGS.image_dir_, img + '.jpg')) for img in imgs]
  text_ids_list = [texts2ids(texts) for texts in texts_list]
  raw_imgs = np.array(raw_imgs)
  text_ids_list = np.array(text_ids_list)
  print(sum([len(x) for x in text_ids_list]), 'num_imgs_done', num_imgs_done, file=sys.stderr, end='\r')
  scores_list = predictor.onebyone_predict(raw_imgs, text_ids_list)
  for img, texts, scores in zip(imgs, texts_list, scores_list):
    for text, score in zip(texts, scores):
      m[img][text] = score
    predicted_imgs.add(img)

  num_imgs_done += batch_size

  for img in predicted_imgs:
    if img in done_imgs:
      continue
    done_imgs.add(img)
    result = m[img]
    sorted_result = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
    texts = []
    scores = []
    for text, score in sorted_result:
      texts.append(text)
      scores.append(str(score))
    texts = ' '.join(texts)
    scores = ' '.join(scores)
    print(img, sorted_result[0][0], sorted_result[0][1], texts, scores, sep='\t', file=out)
    out.flush()


def predict(predictor, imgs, texts, m):
  batch_imgs = []
  batch_texts = []
  for img, text in zip(imgs, texts):
    batch_imgs.append(img)
    batch_texts.append(text)
    if len(batch_imgs) == batch_size:
      _predict(predictor, batch_imgs, batch_texts, m)
      batch_imgs = []
      batch_texts = []

  if batch_imgs:
    _predict(predictor, batch_imgs, batch_texts, m)

def main(_):
  model_dir = FLAGS.model_dir 
  assert model_dir
  model_path = melt.get_model_path(model_dir)
  print('model_path:', model_path, file=sys.stderr) 

  result_file = FLAGS.result_file or model_path + '.inference.txt'
  print('result file is:', result_file, file=sys.stderr)
  if os.path.exists(result_file):
    for line in open(result_file):
      img = line.strip().split()[0]
      done_imgs.add(img)
    print('result file exsits inference contiue, already done %d imgs' % len(done_imgs))
  else:
    print('inference from start')
  global out
  out = open(result_file, 'w')
  
  Predictor = SimPredictor
  image_model = None 
  image_checkpoint_file = FLAGS.image_checkpoint_file or '/home/gezi/data/image_model_check_point/inception_resnet_v2.ckpt'
  image_model_name = melt.get_imagenet_from_checkpoint(image_checkpoint_file).name
  if not melt.has_image_model(model_dir, image_model_name):
    image_model = melt.image.ImageModel(image_checkpoint_file, 
                                        FLAGS.image_model_name,
                                        feature_name=FLAGS.feature_name)
  print('image_model:', image_model, file=sys.stderr)
  predictor = Predictor(model_dir, image_model=image_model)
  
  input_file = FLAGS.input_file
  candidates = {}
  m = {}
  for line in open(input_file):
    l = line.strip().split('\t')
    img, texts = l[0], l[-2]
    if img in done_imgs:
      continue
    texts = texts.split(' ')
    candidates[img] = texts
    m[img] = {}
    for text in texts:
      m[img][text] = 0.
  
  imgs_tocalc, texts_tocalc = zip(*candidates.items())
  predict(predictor, imgs_tocalc, texts_tocalc, m)


if __name__ == '__main__':
  tf.app.run()
