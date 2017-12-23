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

class TextPredictor(object):
  def __init__(self, model_dir, 
               vocab_path=None, 
               image_checkpoint_path=None, 
               image_model_name=None, 
               feature_name=None, 
               image_model=None,
               current_length_normalization_factor=None,
               length_normalization_fator=None,
               index=0, 
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

    if not isinstance(model_dir, (list, tuple)):
      self.predictor = melt.TextPredictor(model_dir, 
                                          index=index, 
                                          current_length_normalization_factor=current_length_normalization_factor,
                                          length_normalization_fator=length_normalization_fator,
                                          vocab=ids2text.vocab,
                                          sess=sess)
    else:
      self.predictor = melt.EnsembleTextPredictor(model_dir, index=index, sess=sess)

    print('TextPredictor.image_model:', self.image_model, file=sys.stderr)

    if vocab_path:
      ids2text.init(vocab_path)

  def _predict(self, image, text_key=None, score_key=None):
    if self.image_model is not None:
      if self.feature_name is None:
        image = self.image_model.gen_feature(image)
      else:
        image = self.image_model.gen_features(image)
    return self.predictor.predict_text(image, text_key, score_key)
  
  def predict_text(self, image, text_key=None, score_key=None):
    return self._predict(image, text_key, score_key)

  def predict_full_text(self, image):
    if self.image_model is not None:
      if self.feature_name is None:
        image = self.image_model.gen_feature(image)
      else:
        image = self.image_model.gen_features(image)
    return self.predictor.predict_full_text(image)    

  def predict(self, ltext, rtext):
    if self.image_model is not None:
      if self.feature_name is None:
        ltext = self.image_model.gen_feature(ltext)
      else:
        ltext = self.image_model.gen_features(ltext)
    return self.predictor.predict(ltext, rtext)

  def elementwise_predict(self, ltexts, rtexts):
    if self.image_model is not None:
      if self.feature_name is None:
        ltexts = self.image_model.gen_feature(ltexts)
      else:
        ltexts = self.image_model.gen_features(ltexts)
    return self.predictor.elementwise_predict(ltexts, rtexts)

  def bulk_predict(self, ltexts, rtexts):
    if self.image_model is not None:
      if self.feature_name is None:
        ltexts = self.image_model.gen_feature(ltexts)
      else:
        ltexts = self.image_model.gen_features(ltexts)
    return self.predictor.bulk_predict(ltexts, rtexts)

  def word_ids(self, image):
    return self._predict(image)

  def translate(self, image):
    texts, scores = self._predict(image)
    return [ids2text.translate(text[0]) for text in texts]

  def predict_best(self, image):
    return self.translate(image)