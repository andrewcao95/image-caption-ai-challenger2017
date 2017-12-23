#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   classification-evaluate.py
#        \author   chenghuige  
#          \date   2017-11-10 17:05:17.938866
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import gezi
import melt
import traceback

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('prediction_file', None, '')
flags.DEFINE_string('metrics_file', None, '')
flags.DEFINE_bool('write_step', False, 'if False will write epoch')
flags.DEFINE_string('ref_file', '/data2/data/ai_challenger/scene/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.txt', '')
flags.DEFINE_string('img_dir', None, '')
flags.DEFINE_string('type', None, '')
flags.DEFINE_integer('imgs_per_row', 5, '')

logging = melt.logging

def main(_):
  #img_dir = 'D:\\new2\\data\\ai_challenger\\scene\\pic\\'
  img_dir = '/home/gezi/data/ai_challenger/scene/pic/'
  if FLAGS.img_dir:
    img_dir = FLAGS.img_dir 
  elif FLAGS.type:
    if FLAGS.type == 'inference':
      #img_dir = 'D:\\new2\\data\\ai_challenger\\scene\\ai_challenger_scene_test_a_20170922\\scene_test_a_images_20170922\\'
      img_dir = '/home/gezi/data/ai_challenger/scene/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922/'


  img_html = '<p> <td><a href={0} target=_blank><img src={0} height=250 width=250></a></p> {1} <br /> {2} <br /> {3}  <br /> {4} <br /> {5} <br /> {6} <br /> {7} <td>'


  ref_file = FLAGS.ref_file
  refs = {}
  label_map = {}
  for line in open(ref_file):
    img, label, text = line.strip().split('\t')
    refs[img] = label
    label_map[label] = text

  prediction_file = FLAGS.prediction_file or sys.argv[1]
  assert prediction_file

  log_dir = os.path.dirname(prediction_file)
  log_dir = log_dir or './'
  print('prediction_file', prediction_file, 'log_dir', log_dir, file=sys.stderr)
  logging.set_logging_path(log_dir)

  metrics_file = FLAGS.metrics_file or prediction_file.replace('evaluate-inference', 'metrics')
  out = open(metrics_file, 'w')
  print('image_id', 'label', 'label_probability', 'predict', 'probability', 'recall@1', 'recall@3', sep='\t', file=out)

  top1_err_html_file = prediction_file.replace('evaluate-inference.txt', 'top1_err.html')
  top3_err_html_file = prediction_file.replace('evaluate-inference.txt', 'top3_err.html')

  out_top1_html = open(top1_err_html_file, 'w')
  out_top3_html = open(top3_err_html_file, 'w')

  total = 0
  correct_at_1 = 0
  correct_at_3 = 0
  for line in open(prediction_file):
    img, top_classes, top_logits, top_predictions, logits, predictions = line.strip().split('\t')
    top_classes = top_classes.split()
    if refs[img] == top_classes[0]:
      correct_at_1 += 1
    else:
      if (total - correct_at_1) % FLAGS.imgs_per_row == 0:
        print('<table><tr>', file=out_top1_html)
      print(img_html.format('%s\\%s' % (img_dir, img),
                            label_map[refs[img]],  
                            predictions.split()[int(refs[img])], 
                            ' '.join([label_map[x] for x in top_classes]), 
                            top_logits,
                            top_predictions, 
                            int(refs[img] == top_classes[0]), 
                            int(refs[img] in top_classes[:3]), 
                            img), file=out_top1_html)
      if (total - correct_at_1 + 1) % FLAGS.imgs_per_row == 0:
        print('</tr></table>', file=out_top1_html)
    
    if refs[img] in top_classes[:3]:
      correct_at_3 += 1
    else:
      if (total - correct_at_3) % FLAGS.imgs_per_row == 0:
        print('<table><tr>', file=out_top3_html)
      print(img_html.format('%s\\%s' % (img_dir, img),
                            label_map[refs[img]],  
                            predictions.split()[int(refs[img])], 
                            ' '.join([label_map[x] for x in top_classes]), 
                            top_logits,
                            top_predictions, 
                            int(refs[img] == top_classes[0]), 
                            int(refs[img] in top_classes[:3]), 
                            img), file=out_top3_html)
      if (total - correct_at_3 + 1) % FLAGS.imgs_per_row == 0:
        print('</tr></table>', file=out_top3_html)
    total += 1
    

    print(img, label_map[refs[img]], predictions.split()[int(refs[img])], 
          ' '.join([label_map[x] for x in top_classes[:3]]), top_predictions, 
          int(refs[img] == top_classes[0]), int(refs[img] in top_classes), sep='\t', file=out)

  assert total == len(refs)

  recall_at_1 = correct_at_1 / total
  recall_at_3 = correct_at_3 / total

  sess = tf.Session()
  summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

  metric_score_str = 'recall@1:[%.4f]\trecall@3:[%.4f]' % (recall_at_1, recall_at_3)
  logging.info('%s\t%s' % (metric_score_str, os.path.basename(prediction_file)))

  print(total, correct_at_1, correct_at_3, file=sys.stderr)

  print('None', 'None', 'None', 'None', 'None', recall_at_1, recall_at_3, sep='\t', file=out)

  summary = tf.Summary()
  if 'ckpt' in prediction_file:
    try:
      epoch = float(os.path.basename(prediction_file).split('-')[1])
      epoch = int(epoch * 100)
      step = int(float(prediction_file.split('-')[2].split('.')[0]))
      prefix = 'step' if FLAGS.write_step else 'epoch'
      melt.add_summarys(summary, [recall_at_1, recall_at_3], ['recall@1', 'recall@3'], prefix=prefix)
      step = epoch if not FLAGS.write_step else step
      summary_writer.add_summary(summary, step)
      summary_writer.flush()
    except Exception:
      print(traceback.format_exc(), file=sys.stderr)


if __name__ == '__main__':
  tf.app.run()
