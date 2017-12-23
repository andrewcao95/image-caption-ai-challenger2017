#!/usr/bin/env python
# ==============================================================================
#          \file   Classifier.py
#        \author   chenghuige  
#          \date   2017-11-09
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import numpy as np 
from collections import namedtuple

import melt   

import numpy as np

import tensorflow as tf

class Classifier(object):
  def __init__(self, model_dir, top_k=3, sess=None, graph=None):
    predictor = melt.Predictor(model_dir, sess=sess, graph=graph)
    self.sess = predictor.sess
    self.graph = predictor.graph
    #print(self.graph.get_all_collection_keys())

    self.logits = self.graph.get_collection('logits')[-1] 
    # TODO experiment what will be better for ensemble predictions or logits
    self.predictions = self.graph.get_collection('predictions')[-1]
    self.top_logits, self.top_classes = tf.nn.top_k(self.logits, top_k)
    #print('---', self.logits, self.top_classes)

    self.feed = self.graph.get_collection('feed')[-1]

  def inference(self, images):
    images = melt.try_convert_images(images)
    # ---- Tensor("TopKV2:1", shape=(1, ?, 3), dtype=int32) Tensor("TopKV2:0", shape=(1, ?, 3), dtype=float32)
    #print('----', self.top_classes, self.top_logits)
    top_classes, top_logits, logits, predictions = self.sess.run(
                                    [self.top_classes, self.top_logits, self.logits, self.predictions],
                                    feed_dict={self.feed: images})
    top_predictions = []
    for prediction, top_class in zip(predictions, top_classes):
      top_predictions.append([prediction[x] for x in top_class])
    top_predictions = np.array(top_predictions)

    Result = namedtuple('Result', ['top_classes', 'top_logits', 'top_predictions', 'logits', 'predictions'])

    return Result(top_classes, top_logits, top_predictions, logits, predictions)


  def predict(self, images):
    return self.inference(images)