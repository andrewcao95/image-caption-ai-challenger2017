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

files = glob.glob('*model*.evaluate-inference.txt')

models = [x.strip('.evaluate-inference.txt') for x in files if not 'mil' in x]

for model in models:
  model = './' + model
  ofile = './%s.evaluate-inference.logprobs.txt' % model 
  print(model, ofile)
  if os.path.exists(ofile):
    print('exists %s' % model)
    continue
  os.system('touch %s' % ofile)
  command = 'python ./dump-logprobs.py --model %s' % model 
  print(command)
  os.system(command)

