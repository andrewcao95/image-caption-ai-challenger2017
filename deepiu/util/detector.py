#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   detector.py
#        \author   chenghuige  
#          \date   2017-11-22 06:36:37.338237
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import numpy as np 
from collections import namedtuple

import melt   
import gezi

import numpy as np

import tensorflow as tf
 
from PIL import Image
import io

class Detector(object):
  def __init__(self, frozen_graph, frozen_graph_name='', sess=None, graph=None):
    predictor = melt.Predictor(frozen_graph=frozen_graph, 
                               frozen_graph_name=frozen_graph_name, 
                               sess=sess, graph=graph)
    self.sess = predictor.sess
    self.graph = predictor.graph

    self.feed = self.graph.get_tensor_by_name('image_tensor:0')
    self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
    self.scores = self.graph.get_tensor_by_name('detection_scores:0')
    self.classes = self.graph.get_tensor_by_name('detection_classes:0')
    self.num = self.graph.get_tensor_by_name('num_detections:0')
    with self.graph.as_default():
      self.image_feed = tf.placeholder(tf.string, [None, ], name='image')
      decode_one = melt.image.decode_image
      self.decode = tf.map_fn(lambda img: decode_one(img), self.image_feed, dtype=tf.uint8)

  def inference(self, images):
    images = melt.try_convert_images(images)
    # NOTICE below all work but tf version has some diff of scores with PIL vesion TODO
    #images = self.sess.run(self.decode, feed_dict={self.image_feed: images})
    
    images = np.array([gezi.load_image_into_numpy_array(Image.open(io.BytesIO(image))) for image in images])
    self.images = images

    results = self.sess.run([self.boxes, self.scores, self.classes, self.num],
                            feed_dict={self.feed: images})
    
    Result = namedtuple('Result', ['boxes', 'scores', 'classes', 'num'])


    return [Result(*t) for t in zip(*results)]


  def predict(self, images):
    return self.inference(images)