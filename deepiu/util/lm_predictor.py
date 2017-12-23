#!/usr/bin/env python
# ==============================================================================
#          \file   TextPredictor.py
#        \author   chenghuige  
#          \date   2017-09-14 17:35:07.765273
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import numpy as np 
import melt   
from deepiu.util import ids2text

class LanguageModelPredictor(object):
  def __init__(self, model_dir, 
               current_length_normalization_factor=None,
               length_normalization_fator=None,
               index=-1, 
               sess=None,
               graph=None):
    predictor = melt.Predictor(model_dir, sess=sess, graph=graph)
    self.sess = predictor.sess
    self.graph = predictor.graph

    self.score = self.graph.get_collection('score')[-1]
    self.feed = self.graph.get_collection('rfeed')[-1] 

  def predict(self, texts):
    if not isinstance(texts, (list, tuple, np.ndarray)):
      texts = [texts]
    score = self.sess.run(self.score, {self.feed: texts})
    return score
    
