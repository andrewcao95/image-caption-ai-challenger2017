#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   post-deal-tfidf-dict.py
#        \author   chenghuige  
#          \date   2017-10-31 08:31:06.516811
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import gezi 


for line in sys.stdin:
  word, tfidf = line.rstrip().split('\t')
  if not gezi.is_single_cn(word):
    print(word, tfidf, sep='\t')
