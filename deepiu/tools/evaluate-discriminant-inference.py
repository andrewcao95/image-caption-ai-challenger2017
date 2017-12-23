#!/usr/bin/env python
# ==============================================================================
#          \file   evaluate-discriminant-inference.py
#        \author   chenghuige  
#          \date   2017-10-06 20:29:49.144771
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import melt


model_dir = sys.argv[1]
input_file = sys.argv[2]

model_dir, model_path = melt.get_model_dir_and_path(model_dir)

feature_name = 'attention'
if len(sys.argv) > 3:
  feature_name = sys.argv[3]

batch_size = 512
if len(sys.argv) > 4:
  batch_size = int(sys.argv[4])

result_file = model_path + '.evaluate-inference.txt'

command = """python /home/gezi/mine/hasky/deepiu/tools/caption-discriminant-inference.py \\
  --vocab=/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic/'vocab.txt' \\
  --model_dir={} \\
  --input_file={} \\
  --result_file={} \\
  --batch_size_={} \\
  --feature_name={} \\
  """.format(model_path, input_file, result_file, batch_size, feature_name)

print(command)

os.system(command)
