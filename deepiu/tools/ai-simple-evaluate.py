#!/usr/bin/env python
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2017-10-06 19:10:05.965134
#   \Description  
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

flags.DEFINE_integer('num_imgs', None, '')


import sys, os
import dill

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

def main(_):
  prediction_file = FLAGS.prediction_file or sys.argv[1]

  assert prediction_file

  log_dir = os.path.dirname(prediction_file)
  log_dir = log_dir or './'
  print('prediction_file', prediction_file, 'log_dir', log_dir, file=sys.stderr)
  logging.set_logging_path(log_dir)

  sess = tf.Session()
  summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

  refs = prepare_refs()
  tokenizer = prepare_tokenization()
  ##TODO some problem running tokenizer..
  #refs = tokenizer.tokenize(refs)

  caption_metrics_file = FLAGS.caption_metrics_file or prediction_file.replace('evaluate-inference', 'simple_metrics')
  imgs = []
  captions = []
  infos = {}
  for line in open(prediction_file):
    l = line.strip().split('\t')
    img, caption, all_caption, all_score = l[0], l[1], l[-2], l[-1]
    img = img.replace('.jpg', '')
    img += '.jpg'
    imgs.append(img)

    infos[img] = '%s %s' % (all_caption.replace(' ', '|'), all_score.replace(' ', '|'))

    caption = caption.replace(' ', '').replace('\t', '')
    caption_words = [x.encode('utf-8') for x in jieba.cut(caption)]
    caption_str = ' '.join(caption_words)
    captions.append([caption_str])

  results = dict(zip(imgs, captions))
  
  #results = tokenizer.tokenize(results)

  selected_results, selected_refs = translation_reorder_keys(results, refs)

  scorers = [
            (Bleu(4), ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]),
            (Cider(document_frequency=document_frequency, ref_len=ref_len), "cider"),
        ]

  score_list = []
  metric_list = []
  scores_list = []

  print('img&predict&label:{}:{}{}{}'.format(selected_results.items()[0][0], '|'.join(selected_results.items()[0][1]), '---', '|'.join(selected_refs.items()[0][1])), file=sys.stderr)
  
  for scorer, method in scorers:
    print('computing %s score...' % (scorer.method()), file=sys.stderr)
    score, scores = scorer.compute_score(selected_refs, selected_results)
    if type(method) == list:
      score_list.append(score[-1])
      metric_list.append(method[-1])
      scores_list.append(scores[-1])
      print(method[-1], score[-1], file=sys.stderr)
    else:
      score_list.append(score)
      metric_list.append(method)
      scores_list.append(scores)
      print(method, score, file=sys.stderr)

  assert(len(score_list) == 2)

  avg_score = np.mean(np.array(score_list))
  score_list.insert(0, avg_score)
  metric_list.insert(0, 'avg')

  if caption_metrics_file:
    out = open(caption_metrics_file, 'w')
    print('image_id', 'caption', 'ref', '\t'.join(metric_list), 'infos', sep='\t', file=out)
    for i in range(len(selected_results)):
      key = selected_results.keys()[i] 
      result = selected_results[key][0]
      refs = '|'.join(selected_refs[key])
      bleu_4 = scores_list[0][i]
      cider = scores_list[1][i]
      avg = (bleu_4 + cider) / 2.
      print(key.split('.')[0], result, refs, avg, bleu_4, cider, infos[key], sep='\t', file=out)

  metric_list = ['trans_' + x for x in metric_list]
  metric_score_str = '\t'.join('%s:[%.4f]' % (name, result) for name, result in zip(metric_list, score_list))
  logging.info('%s\t%s'%(metric_score_str, os.path.basename(prediction_file)))

  print(key.split('.')[0], 'None', 'None', '\t'.join(map(str, score_list)), 'None', sep='\t', file=out)

  summary = tf.Summary()
  if score_list and 'ckpt' in prediction_file:
    try:
      epoch = float(os.path.basename(prediction_file).split('-')[1])
      #for float epoch like 0.01 0.02 turn it to 1, 2, notice it make epoch 1 to 100 
      epoch = int(epoch * 100)
      step = int(float(os.path.basename(prediction_file).split('-')[2].split('.')[0]))
      prefix = 'step' if FLAGS.write_step else 'epoch'
      melt.add_summarys(summary, score_list, metric_list, prefix=prefix)
      step = epoch if not FLAGS.write_step else step
      summary_writer.add_summary(summary, step)
      summary_writer.flush()
    except Exception:
      print(traceback.format_exc(), file=sys.stderr)


if __name__ == '__main__':
  tf.app.run()
