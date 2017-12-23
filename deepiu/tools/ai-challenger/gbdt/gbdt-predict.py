#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gbdt-predict.py
#        \author   chenghuige  
#          \date   2017-11-24 22:55:12.082016
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import gezi.nowarning 
from libmelt import PredictorFactory, Vector
predictor = PredictorFactory.LoadPredictor('./model/')

out = open('./ensemble.inference.gbdt_result.txt', 'w')  

for i, line in enumerate(open('./ensemble.inference.feature.final.txt')):
  if line.startswith('#'):
    continue
  if i % 10000 == 0:
    print('finish gbdt predict of %d pic-caption pair' % i, file=sys.stderr, end='\r')
  img, caption, feature_str = line.strip().split('\t', 2)
  print(img, caption, predictor.Predict(Vector(feature_str)), sep='\t', file=out)
  
