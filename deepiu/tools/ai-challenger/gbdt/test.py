#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   test.py
#        \author   chenghuige  
#          \date   2017-12-03 05:01:22.243326
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

for line in open('./showattentell.bahdanau.finetune_model.ckpt-28.70-941647.evaluate-inference.attention.feature.txt'):
  l = line.strip().split('\t')
  if len(l) != 39:
    print(line)
