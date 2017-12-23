#!/usr/bin/env python
#encoding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', 'ensemble.train.detect.scene.lm.attention.logprobs.imagenet.txt', '')
flags.DEFINE_string('type', 'valid', 'valid or test')

import sys
import cPickle as pickle
import numpy as np

import gezi 

from deepiu.util import idf
from deepiu.util import vocabulary

vocabulary.init('/home/gezi/mount/temp/image-caption/ai-challenger/tfrecord/seq-basic/vocab.txt')
vocab = vocabulary.vocab
idf_weights = idf.get_idf()

print('type', FLAGS.type)

if FLAGS.type == 'test':
  FLAGS.input_file = './ensemble.inference.feature.detect.scene.lm.attention.logprobs.imagenet.txt'

hotword_file = './hotword.%s.pkl' % FLAGS.type

hotwords = open('/data2/data/ai_challenger/image_caption/hotword/hot-word.txt').readlines()
hotwords.insert(0, 'None')

ofile = FLAGS.input_file.replace('.txt', '.hotword.txt')
print('infile:', FLAGS.input_file, 'ofile:',ofile, file=sys.stderr)
names = []
objects = ['bao', 'shouji', 'maozi', 'toukui', 'pingzi', 'beizi', 'mojing', 'yanjing', 'huatong', 'bi', 'xiangzi', 'hezi', 'xiangji']
print('num_objects:', len(objects))
objects_cn = ['包', '手机', '帽子', '头盔', '瓶子', '杯子', '墨镜', '眼镜', '话筒', '笔', '箱子', '盒子', '相机']
print('num_objects_cn', len(objects_cn))
names += ['has_' + x for x in objects]
print('names has_', len(names))
names += ['logits_none']
names += ['logits_' + x for x in objects] 
print('names len', len(names))
#names += ['logits_idf_' + x for x in objects]

timer = gezi.Timer('load scene_file %s' % hotword_file, True)
with open(hotword_file) as f:
  hotwords_results = pickle.load(f)
timer.print()

timer = gezi.Timer('load segged caption file')
with open('./segged_caption.pkl') as f:
  segged_captions = pickle.load(f)
timer.print()

with open(ofile, 'w') as out:
  is_header = True 
  for line in open(FLAGS.input_file):
    l = line.strip().split('\t')
    
    # for new features only test purpose
    #l = l[:3]
    if is_header:
      names = l + names 
      print('\t'.join(names), file=out)
      with open('./feature_name.txt', 'w') as out_fname:
        for name in names[3:]:
          print(name, file=out_fname)
      is_header = False
      continue

    img, caption = l[0], l[1]
    
    words = segged_captions[caption]

    has_list = [0.] * len(objects)
    if words.count('包'):
      has_list[0] = 1. 
    if words.count('手机'):
      has_list[1] = 1. 
    if '帽' in caption:
      has_list[2] = 1. 
    if words.count('头盔'):
      has_list[3] = 1. 
    if '瓶' in caption:
      has_list[4] = 1. 
    if '杯' in caption:
      has_list[5] = 1. 
    if words.count('墨镜'):
      has_list[6] = 1. 
    if words.count('眼镜'):
      has_list[7] = 1. 
    if words.count('话筒'):
      has_list[8] = 1.
    if '笔' in caption:
      has_list[9] = 1.
    if '箱子' in caption:
      has_list[10] = 1.
    if '盒子' in caption:
      has_list[11] = 1.
    if '相机' in caption:
      has_list[12] = 1.
    
    l += map(str, has_list)

    logits = hotwords_results[img].logit
    l += map(str, logits)
 
    assert len(l) == len(names), 'len(l):%d len(names):%d' % (len(l), len(names))
    print('\t'.join(l), file=out)
      

