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

class TextAttentionPredictor(object):
  def __init__(self, model_dir, 
               image_checkpoint_path=None, 
               image_model_name=None, 
               feature_name=None, 
               image_model=None,
               index=-1, 
               sess=None,
               graph=None):
    self.image_model = image_model
    if feature_name is None:
      model_dir_ = model_dir[0] if isinstance(model_dir, (list, tuple)) else model_dir
      if melt.varname_in_checkpoint('attention', model_dir_):
        # show atten and tell mode will use gen features
        print('TexPredictor attention model', file=sys.stderr)
        feature_name = 'attention'
    
    self.feature_name = feature_name
    
    print('TextPredictor feature_name:', feature_name, file=sys.stderr)
    if image_model is None and image_checkpoint_path:
      self.image_model = melt.image.ImageModel(image_checkpoint_path, 
                                               sess=sess,
                                               graph=graph)

    print('TextPredictor.image_model:', self.image_model, file=sys.stderr)

    predictor = melt.Predictor(model_dir, sess=sess, graph=graph)
    self.sess = predictor.sess
    self.graph = graph = predictor.graph
    self.ops = [graph.get_collection(x)[index] for x in ['beam_text', 'beam_text_score', 'beam_logprobs_history', 'beam_alignment_history']]

  def _predict(self, image):
    if self.image_model is not None:
      if self.feature_name is None:
        image = self.image_model.gen_feature(image)
      else:
        image = self.image_model.gen_features(image)
    return self.sess.run(self.ops, feed_dict={self.graph.get_collection('feed')[-1]: image})
  
  def predict_text(self, image):
    return self._predict(image)
