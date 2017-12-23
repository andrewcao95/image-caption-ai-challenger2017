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
#logging = melt.logging

model_dir = sys.argv[1] 
#logging.set_logging_path(model_dir)

result_file = os.path.join(model_dir, 'ensemble.evaluate-inference.txt')

if not os.path.exists(result_file):
  command = 'python ./inference/ai-challenger/ensemble-inference.py %s %s'%(model_dir, 'evaluate-inference')
  print(command, sys.stderr)
  os.system(command)
else:
  print('%s exists'%result_file)

result_file2 = result_file.replace('.evaluate-inference.txt', '.caption-metrics.txt')

if not os.path.exists(result_file2):
  command = 'python ./evaluate/ai-challenger/evaluate.py %s'%result_file
  print(command, sys.stderr)
  os.system(command)
else:
  print('%s exists'%result_file2)
