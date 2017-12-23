#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   write-summary.py
#        \author   chenghuige  
#          \date   2017-10-25 08:12:06.491602
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', None, '')

import sys, os
import melt 

logging = melt.logging

import traceback

metrics_map = {}
scores_map = {}
epochs = []

def main(_):
  model_dir = FLAGS.model_dir or sys.argv[1]
  log_dir = FLAGS.log_dir or model_dir
  log_dir = log_dir or './'
  print('model log dir:', log_dir, file=sys.stderr)
  logging.set_logging_path(log_dir)

  command = 'rm -rf %s/events*' % model_dir
  print(command, file=sys.stderr)
  os.system(command)

  sess = tf.Session()
  summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
  #trans_avg:[0.8383]trans_bleu_1:[0.8235]trans_bleu_2:[0.7224]trans_bleu_3:[0.6312]trans_bleu_4:[0.5508]trans_cider:[1.7259]trans_meteor:[0.4020]trans_rouge_l:[0.6743]model.ckpt-27.60-396414.evaluate-inference.txt 

  for line in open(model_dir + '/log.html'):
    l = line.split('\t')
    
    metrics = []
    scores = []
    try:
      for i, item in enumerate(l[:-1]):
        if i == 0:
          item = item.split()[-1]
        metric, score = item.split(':')
        metrics.append(metric)
        scores.append(float(score[1:-1]))

      checkpoint = '.'.join(l[-1].split('.')[:-2])
      epoch = int(float(checkpoint.split('-')[1]) * 100)

      if epoch not in metrics_map:
        epochs.append(epoch)
      metrics_map[epoch] = metrics
      scores_map[epoch] = scores 
    except Exception:
      print(traceback.format_exc())
      continue

  epochs.sort()
  out = open(model_dir + '/summary.txt', 'w')
  for epoch in epochs:
    metrics = metrics_map[epoch]
    scores = scores_map[epoch]
    
    info = '\t'.join(['%s:%f' % (metric, score) for metric, score in zip(metrics, scores)])
    print(epoch / 100., info, file=out, sep='\t')

    summary = tf.Summary()
    if scores:
      melt.add_summarys(summary, scores, metrics, prefix='epoch')
      summary_writer.add_summary(summary, int(epoch))
      summary_writer.flush()

  if log_dir != model_dir:
    command = 'rsync -l -r -t %s/events* %s' % (log_dir, model_dir)
    print(command, file=sys.stderr)
    os.system(command)
    assert log_dir
    command = 'rm -rf %s/*' % log_dir
    print(command, file=sys.stderr)
    os.system(command)
 
if __name__ == '__main__':
  tf.app.run()

