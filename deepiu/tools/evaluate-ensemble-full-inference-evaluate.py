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

assert melt.get_num_gpus() == 0

model_dir = sys.argv[1] 
#logging.set_logging_path(model_dir)

if len(sys.argv) > 2:
  weights = sys.argv[2]
else:
  weights = ''

result_file = os.path.join(model_dir, 'ensemble.full-evaluate-inference.txt')

#if not os.path.exists(result_file):
command = 'python ~/mine/hasky/deepiu/image_caption/inference/ai-challenger/ensemble-inference.py %s %s %s' % (model_dir, 'full-evaluate-inference', weights)
print(command, file=sys.stderr)
os.system(command)
#else:
#  print('%s exists'%result_file)

result_file2 = result_file.replace('.full-evaluate-inference.txt', '.full-caption-metrics.txt')

if not os.path.exists(result_file2):
  command = 'python ~/mine/hasky/deepiu/image_caption/evaluate/ai-challenger/evaluate.py %s' % result_file
  print(command, file=sys.stderr)
  os.system(command)
else:
  print('%s exists' % result_file2)
