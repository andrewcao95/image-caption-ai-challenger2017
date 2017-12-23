#!/usr/bin/env python
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2017-10-06 19:10:05.965134
#   \Description   for getting upper bound of cider, bleu4 from current recall possible best sort
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('valid_resource_dir', '/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic/valid/', '')

flags.DEFINE_string('prediction_file', None, '')

flags.DEFINE_string('result_dir', None, '')
flags.DEFINE_string('caption_metrics_file', None, '')

flags.DEFINE_bool('write_step', False, 'if False will write epoch')

flags.DEFINE_integer('num_rounds', None, '')
flags.DEFINE_integer('num_imgs', None, '')


import sys, os
import pickle

from gezi.metrics import Bleu, Meteor, Rouge, Cider, PTBTokenizer

import jieba

import numpy as np

import melt
logging = melt.logging

import gezi

try:
  import cPickle as pickle
except ImportError:
  import pickle

import traceback

import dill

import operator

refs = None
refs_len = None
tokenizer = None
results = {}

document_frequency = None 
ref_len = None

def prepare_refs():
  global refs, refs_len
  if refs is None:
    ref_name = 'valid_refs.pkl'

    ref_path = os.path.join(FLAGS.valid_resource_dir, ref_name)
    if os.path.exists(ref_path):
      refs = pickle.load(open(ref_path, 'rb'))
      print('len refs:', len(refs), file=sys.stderr)
    else:
      print('not found ref path', file=sys.stderr)

    refs_len = len(refs)
    
    global document_frequency, ref_len
    test_dir = FLAGS.valid_resource_dir
    print('cider using global idf info from valid data')
    document_frequency_path = os.path.join(test_dir, 'valid_refs_document_frequency.dill')
    # how to debug add below and then tyep n
    #import pdb
    #pdb.set_trace() 
    assert os.path.exists(document_frequency_path)
    ref_len_path = os.path.join(test_dir, 'valid_ref_len.txt')
    assert os.path.exists(ref_len_path)
    document_frequency = dill.load(open(document_frequency_path))
    ref_len = float(open(ref_len_path).readline().strip())
    print('document_frequency {} ref_len {}'.format(len(document_frequency), ref_len))

  return refs

def prepare_tokenization():
  #actually only used for en corpus like MSCOCO, for cn corpus can ignore this
  #for compact also do it for cn corpus
  #print('tokenization...', file=sys.stderr)
  global tokenizer
  if tokenizer is None:
    tokenizer = PTBTokenizer()
  #return tokenizer.tokenize(refs)
  return tokenizer

def translation_reorder_keys(results, refs):
  selected_refs = {}
  selected_results = {}
  #by doing this can force same .keys()
  for key in results:
    rkey = key.split('_')[0]
    selected_refs[key] = refs[rkey]
    selected_results[key] = results[key]

    assert len(selected_results[key]) == 1, selected_results[key]
  assert selected_results.keys() == selected_refs.keys(), '%d %d' % (len(selected_results.keys()), len(selected_refs.keys()))  
  return selected_results, selected_refs

caption_info = {}
caption_sort_info = {}


segged_caption = {}

if os.path.exists('segged_caption.pkl'):
  with open('segged_caption.pkl', 'rb') as handle:
    segged_caption = pickle.load(handle)

num_segged_caption = len(segged_caption)
print('num_segged_caption', num_segged_caption, file=sys.stderr)


def calc_best_metric(metric_info):
  score = 0 
  num_imgs = 0
  for img, info in metric_info.items():
    sorted_result = sorted(info.items(), key=operator.itemgetter(1), reverse=True)
    score += sorted_result[0][1]
    num_imgs += 1
  return score / num_imgs

def main(_):
  prediction_file = FLAGS.prediction_file or sys.argv[1]

  assert prediction_file

  log_dir = os.path.dirname(prediction_file)
  log_dir = log_dir or './'
  print('prediction_file', prediction_file, 'log_dir', log_dir, file=sys.stderr)
  logging.set_logging_path(log_dir)

  refs = prepare_refs()
  tokenizer = prepare_tokenization()
  ##TODO some problem running tokenizer..
  #refs = tokenizer.tokenize(refs)

  caption_metrics_file = FLAGS.caption_metrics_file or prediction_file.replace('evaluate-inference', 'best_metrics')


  scorers = [
            (Bleu(4), ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]),
            (Cider(document_frequency=document_frequency, ref_len=ref_len), "cider"),
        ]

  bleu4_info = {}
  cider_info = {}

  for i, line in enumerate(open(prediction_file)):
    l = line.strip().split('\t')
    img, caption, all_caption, all_score = l[0], l[1], l[-2], l[-1]

    img = img.replace('.jpg', '')
    img += '.jpg'

    if i % 100 == 0:
      print(i, img, end='\r')
    if FLAGS.num_imgs and i == FLAGS.num_imgs:
      break

    bleu4_info[img] = {}
    cider_info[img] = {}

    caption_sort_info[img] = []

    captions = all_caption.split(' ')

    for caption in captions:
      if caption in segged_caption:
        words = segged_caption[caption]
      else:
        words = [x.encode('utf-8') for x in jieba.cut(caption)]
        segged_caption[caption] = words

      caption_str = ' '.join(words)

      bleu4_info[img][caption_str] = -1.
      cider_info[img][caption_str] = -1.

      caption_sort_info[img].append(caption_str)
    
      
  print('finish parsing file and create map')

  if caption_metrics_file:
    out = open(caption_metrics_file, 'w')
  else:
    out = None

  round = 0
  while True:
    imgs = []
    captions = []
    for img, info in bleu4_info.items():
      for caption in caption_sort_info[img]:
        val = info[caption]
        if val == -1.:
          imgs.append(img)
          captions.append([caption])
          break      
    if not imgs:
      break

    results = dict(zip(imgs, captions))
    #results = tokenizer.tokenize(results)
    selected_results, selected_refs = translation_reorder_keys(results, refs)
    
    scores_list = []
    for scorer, method in scorers:
      print('computing %s score...' % (scorer.method()), file=sys.stderr)
      score, scores = scorer.compute_score(selected_refs, selected_results)

      if type(method) == list:
        logging.info('bleu4:{}'.format(score))
        scores_list.append(scores[-1])
      else:
        logging.info('cider:{}'.format(score))
        scores_list.append(scores)

    if out:
      for i in range(len(selected_results)):
        img = selected_results.keys()[i] 
        caption = selected_results[img][0]
        selected_ref = '|'.join(selected_refs[img])
        bleu_4 = scores_list[0][i]
        bleu4_info[img][caption] = bleu_4
        cider = scores_list[1][i]
        cider_info[img][caption] = cider

        print(img.split('.')[0], caption.replace(' ', ''), bleu_4, cider, selected_ref.replace(' ', ''), sep='\t', file=out)
      out.flush()

    # notice bleu4 and cider is best for itsself actually if bleu4 best then cider can not be best must be lower TODO
    # for bleu4 need to add 0.05 compare to stard calc(0.532018993015 means 0.5822), cider is the same(all 1.8521) TODO
    logging.info('round:{} num_imgs:{} bleu4:{} cider:{}'.format(round, len(imgs), calc_best_metric(bleu4_info), calc_best_metric(cider_info)))
    round += 1
    if FLAGS.num_rounds and round == FLAGS.num_rounds:
      break

# TODO FIXME why here not save no new seg ? why ? if no new seg why ten gen-ensemble-feature has many new seg ?
if len(segged_caption) > num_segged_caption:
  print('num_segged_caption:', num_segged_caption)
  with open('segged_caption.pkl', 'wb') as handle:
      pickle.dump(segged_caption, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  tf.app.run()
