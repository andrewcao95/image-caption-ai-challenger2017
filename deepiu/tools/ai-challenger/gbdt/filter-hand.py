#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   filter-hand.py
#        \author   chenghuige  
#          \date   2017-11-30 00:09:44.931331
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

for line in sys.stdin:
  l = line.strip().split('\t')
  img , caption = l[0], l[1]
  if '左手' in caption or '右手' in caption:
    print(line.strip())
