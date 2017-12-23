#!/usr/bin/env python
# ==============================================================================
#          \file   evaluate-inference-evaluate.py
#        \author   chenghuige  
#          \date   2017-10-06 21:09:24.398921
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import melt

model_dir = sys.argv[1] 

result_file = os.path.join(model_dir, 'ensemble.inference.txt')

if not os.path.exists(result_file):
  command = 'python /home/gezi/mine/hasky/deepiu/image_caption/inference/ensemble-inference.py %s %s'%(model_dir, 'inference')
  print(command, sys.stderr)
  os.system(command)
else:
  print('%s exists'%result_file)
