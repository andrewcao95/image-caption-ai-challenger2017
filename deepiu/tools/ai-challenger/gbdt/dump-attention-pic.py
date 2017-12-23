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

from deepiu.util.sim_predictor import SimPredictor

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

timer = gezi.Timer('load segged caption file')
with open('./segged_caption.pkl') as f:
  segged_captions = pickle.load(f)
timer.print()

out_img_dir = './pic-attention/'
with open(ofile, 'w') as out:
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

    image_path = os.path.join(FLAGS.img_dir, img + '.jpg')
    #TODO later add 
    if img != pre_img:
      img_data = ndimage.imread(image_path)
      pre_img = img

    attention_size = alignment.shape[-1]
    dim = int(np.sqrt(attention_size))
    img_height = img_data.shape[0]
    img_width = img_data.shape[1]
    
    entropies = np.array([1e20] * len(words))
    for i, word in enumerate(words):
      if not (word == '左手' or word == '右手' or word == '左肩' or word == '右肩'):
        entropies[i] = gezi.probs_entropy(alignment[i])
      if entropies[i] < 3.5:
        alpha_img = skimage.transform.resize((alignment > 0.005).reshape(dim, dim), [img_width, img_height])
        objs = ndimage.find_objects(alpha_img.astype(int), 1)
        if not objs:
          continue
        obj = objs[0]
        sub_img = image_data[obj[0].start: obj[0].stop, obj[1].start: obj[1].stop]
        sub_pic_name = '%s.%s.%s.%d.jpg' % (img, caption, word, i)
        sub_pic_path = os.path.join(out_img_dir, sub_pic_name)
        Image.fromarray(sub_img).save(sub_pic_path)


  