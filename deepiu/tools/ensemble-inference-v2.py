#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   ensemble-inference-v2.py
#        \author   chenghuige  
#          \date   2017-10-21 14:56:40.017795
#   \Description   This is time cosuming 1hour and 12 mintues, and not performance 
#                  better so just use ensemble-inference.py will be fine
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('image_dir_', '/home/gezi/data2/data/ai_challenger/image_caption/pic', '') 
flags.DEFINE_string('vocab', '/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic/vocab.txt', '')
flags.DEFINE_float('current_length_normalization_factor', None, '')
flags.DEFINE_float('length_normalization_fator', None, '')

import sys, os

import glob
import operator

import melt, gezi
from deepiu.util import text2ids
from deepiu.util.text2ids import texts2ids
from deepiu.util.text_predictor import TextPredictor

import numpy as np

input = sys.argv[1]
type = sys.argv[2]

text2ids.init()

if ',' in input:
  files = input.split(',')
else:
  files = glob.glob(input + '/model*.%s.txt' % type)
  if not 'ensemble' in type:
    files = [x for x in files if not 'ensemble' in x]

dir = os.path.dirname(files[0])

ensemble_input_file = 'ensemble.%s.txt' % type

print('files:', files, 'len(files)', len(files), file=sys.stderr)
print('ensemble input file:', ensemble_input_file, file=sys.stderr)

batch_size = int(sys.argv[3])

num_imgs_done = 0

def _predict(predictor, imgs, texts_list, m):
	global num_imgs_done
	raw_imgs = [melt.read_image(os.path.join(FLAGS.image_dir_, img + '.jpg')) for img in imgs]
	text_ids_list = [texts2ids(texts) for texts in texts_list]
	raw_imgs = np.array(raw_imgs)
	text_ids_list = np.array(text_ids_list)
	print([len(x) for x in text_ids_list], sum([len(x) for x in text_ids_list]), \
			  'num_imgs_done', num_imgs_done, file=sys.stderr)
	scores_list = predictor.bulk_predict(raw_imgs, text_ids_list)

	if num_imgs_done == 0:
		print(scores_list.shape, scores_list, file=sys.stderr)

	for img, texts, scores in zip(imgs, texts_list, scores_list):
		for text, score in zip(texts, scores):
			m[img][text] = score

	num_imgs_done += batch_size


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

candidates = {}
for line in open(ensemble_input_file):
	l = line.strip().split('\t')
	img, texts = l[0], l[-2]
	texts = texts.split(' ')
	candidates[img] = texts

for file in files:
	model_dir = file.replace('.%s.txt'%type, '')
	ofile = os.path.join(dir, '%s.ensemble.%s.txt' % (model_dir, type))
	print('model_dir:', model_dir, 'ofile:', ofile)
	if gezi.non_empty(ofile):
		continue
	
	out = open(ofile, 'w')

	Predictor = TextPredictor  
	image_model = None 
	image_checkpoint_file = FLAGS.image_checkpoint_file or '/home/gezi/data/image_model_check_point/inception_resnet_v2_2016_08_30.ckpt'
	image_model_name = melt.image.get_imagenet_from_checkpoint(image_checkpoint_file).name
	print('image_model_name:', image_model_name)
	if not melt.has_image_model(model_dir, image_model_name):
	  image_model = melt.image.ImageModel(image_checkpoint_file, image_model_name)
	print('image_model:', image_model, file=sys.stderr)
	predictor = Predictor(model_dir, image_model=image_model, vocab_path=FLAGS.vocab,
												current_length_normalization_factor=FLAGS.current_length_normalization_factor,
	                      length_normalization_fator=FLAGS.length_normalization_fator) 

	#predictor = None

	m = {}
	for line in open(file):
		l = line.strip().split('\t')
		img, texts , scores = l[0], l[-2], l[-1]
		if img not in m:
			m[img] = {}
		texts = texts.split(' ')
		scores = map(float, scores.split(' '))
		for text, score in zip(texts, scores):
			m[img][text] = score 

	imgs_tocalc = []
	texts_tocalc = []

	for img, texts in candidates.items():
		texts_ = [x for x in texts if x not in m[img]]
		if texts_:
			imgs_tocalc.append(img)
			texts_tocalc.append(texts_)

	predict(predictor, imgs_tocalc, texts_tocalc, m)
	
	for img, result in m.items():
	  sorted_result = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
	  texts = []
	  scores = []
	  for text, score in sorted_result:
	    texts.append(text)
	    scores.append(str(score))
	  texts = ' '.join(texts)
	  scores = ' '.join(scores)
	  print(img, sorted_result[0][0], sorted_result[0][1], texts, scores, sep='\t', file=out)