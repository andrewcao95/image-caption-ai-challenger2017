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
  weights = 'None'

if len(sys.argv) > 3:
  method = sys.argv[3]
else:
  method = 'logits'

result_file = os.path.join(model_dir, 'ensemble.evaluate-inference.txt')

#if not os.path.exists(result_file):
command = 'python /home/gezi/mine/hasky/deepiu/tools/classification-ensemble-inference.py %s %s %s %s' % (model_dir, 'evaluate-inference', weights, method)
print(command, file=sys.stderr)
os.system(command)
#else:
#  print('%s exists'%result_file)

command = 'python /home/gezi/mine/hasky/deepiu/tools/classification-txt2json.py %s' % result_file
print(command, file=sys.stderr)
os.system(command)
command = '''python /home/gezi/mine/hasky/deepiu/tools/scene_eval.py \\
             --submit %s \\
             --ref /home/gezi/data2/data/ai_challenger/scene/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json \\
          ''' % result_file.replace('.txt', '.json')
print(command, file=sys.stderr)
os.system(command)

result_file2 = result_file.replace('.evaluate-inference.txt', '.caption-metrics.txt')
if not os.path.exists(result_file2):
  command = 'python /home/gezi/mine/hasky/deepiu/tools/classification-evaluate.py %s' % result_file
  print(command, file=sys.stderr)
  os.system(command)
else:
  print('%s exists' % result_file2)
