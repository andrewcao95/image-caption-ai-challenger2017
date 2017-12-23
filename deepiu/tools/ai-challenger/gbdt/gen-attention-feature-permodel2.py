#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-attention-feature-permodel.py
#        \author   chenghuige  
#          \date   2017-12-02 20:08:52.122757
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import glob 
import numpy as np
import cPickle as pickle
import gezi
import melt 
import traceback

from deepiu.util.sim_predictor import SimPredictor
from deepiu.util import text2ids

import skimage.transform
from scipy import ndimage
from PIL import Image
import tempfile

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', None, '')
flags.DEFINE_string('type', 'valid', 'valid or test')
flags.DEFINE_string('img_dir', '/home/gezi/data2/data/ai_challenger/image_caption/pic/', '')
flags.DEFINE_integer('img_id', 0, '')

image_model_checkpoint_path = '/home/gezi/data/image_model_check_point/inception_resnet_v2.ckpt'
model_dir = '/home/gezi/new/temp/image-caption/ai-challenger/model.v4/mil.idf.rnn2/epoch'
vocab_path = '/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic/vocab.txt'
valid_dir = '/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic/valid'

image_model_name = melt.get_imagenet_from_checkpoint(image_model_checkpoint_path).name
image_model = None
if not melt.varname_in_checkpoint(image_model_name, model_dir):
  image_model = melt.image.ImageModel(image_model_checkpoint_path, 
                                      feature_name='attention')

print('image_model:', image_model)

text2ids.init(vocab_path)
vocab = text2ids.vocab

predictor = SimPredictor(model_dir, image_model=image_model)

ofile = FLAGS.input_file.replace('.txt', '.feature.txt')
debug_ofile = ofile.replace('.txt', '.debug.txt')
print('infile:', FLAGS.input_file, 'ofile:',ofile, file=sys.stderr)

#only consider the first occurence of letf hand and right hand
names = ['imtxt_sim_valid', 'imtxt_sim_score']
names = ['attention_' + x for x in names]

timer = gezi.Timer('load segged caption file')
with open('./segged_caption.pkl') as f:
  segged_captions = pickle.load(f)
timer.print()

sess = tf.InteractiveSession()
def calc_imtxt_sim(image, text, obj):
  # TODO duplicate decode encode
  image = melt.read_image(image_path)
  image=tf.image.decode_jpeg(image)
  image = tf.image.crop_to_bounding_box(image, obj[0].start, obj[1].start, obj[0].stop - obj[0].start, obj[1].stop - obj[1].start)
  image = tf.image.encode_jpeg(image)
  image = sess.run(image)
  text = text.replace(' ', '')
  return np.squeeze(predictor.predict([image], [text2ids.text2ids(text)]))

temp_pic =  '/tmp/%d.jpg' % FLAGS.img_id 
debug_out = open(debug_ofile, 'w')
with open(ofile, 'w') as out:
  print('\t'.join(['#img', 'caption'] + names), file=out)
  pre_img = None
  img_data = None
  for i, line in enumerate(open(FLAGS.input_file)):
    if i % 1000 == 0:
      print(i, end='\r', file=sys.stderr)
    img, caption, num_steps, attention_size, alignment = line.strip().split('\t')

    fe = [0.] * len(names)
    num_steps = int(num_steps)
    attention_size = int(attention_size)
    dim = int(np.sqrt(attention_size))
    alignment = np.reshape(np.array(map(float, alignment.split())), [num_steps, attention_size])
    words = segged_captions[caption]
    alignment = alignment[:len(words)]
    
    image_path = os.path.join(FLAGS.img_dir, img + '.jpg')
    #TODO later add 
    if img != pre_img:
      img_data = ndimage.imread(image_path)
      pre_img = img
    
    entropies = np.array([gezi.probs_entropy(x) for x in alignment])
    min_pos = entropies.argmin()
    entropy = entropies[min_pos]
    word = words[min_pos]

    imtxt_sim_valid = 0.
    imtxt_sim = 0.
    img_height = img_data.shape[0]
    img_width = img_data.shape[1]
    alpha_img = skimage.transform.resize((alignment[min_pos] > 0.005).reshape(dim, dim), [img_width, img_height])
    objs = ndimage.find_objects(alpha_img.astype(int), 1)
    x0, x1, y0, y1 = 0., 0., 0., 0.
    if objs and objs[0]:
      obj = objs[0]
      x0, x1, y0, y1 = obj[0].start, obj[0].stop, obj[1].start, obj[1].stop
      if x1 - x0 > 10 and y1 - y0 > 10 and x1 < img_height and y1 < img_width:
        try:
          imtxt_sim_valid = 1.
          imtxt_sim = calc_imtxt_sim(image_path, word, obj)
        except Exception:
          imtxt_sim_valid = 0. 
          imtxt_sim = 0. 
          print(traceback.format_exc())
          print(img, caption, min_pos, entropy, word, x0, x1, y0, y1, imtxt_sim_valid, imtxt_sim)

    print(img, caption, min_pos, entropy, word, x0, x1, y0, y1, imtxt_sim_valid, imtxt_sim, file=debug_out)

    fe[0] = imtxt_sim_valid
    fe[1] = imtxt_sim

    l = map(str, fe)
    print(img, ' '.join(words), '\t'.join(l), sep='\t', file=out)


  
