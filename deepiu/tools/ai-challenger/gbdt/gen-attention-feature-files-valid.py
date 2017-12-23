#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dump-attention-files.py
#        \author   chenghuige  
#          \date   2017-12-02 18:06:26.372175
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import glob 

files = glob.glob('*model*.evaluate-inference.attention.txt')

for file in files:
  file = './' + file
  ofile = file.replace('.txt', '.feature.txt')
  print(file, ofile)
  if os.path.exists(ofile):
    print('exists %s' % ofile)
    continue
  os.system('touch %s' % ofile)
  command = 'python ./gen-attention-feature-permodel.py --input_file %s' % file
  print(command)
  os.system(command)
