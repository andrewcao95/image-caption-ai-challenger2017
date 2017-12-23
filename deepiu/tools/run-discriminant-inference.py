#!/usr/bin/env python
# ==============================================================================
#          \file   evaluate-inference.py
#        \author   chenghuige  
#          \date   2017-10-06 20:29:49.144771
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import melt

assert melt.get_num_gpus() == 1

model_dir = sys.argv[1]

model_dir, model_path = melt.get_model_dir_and_path(model_dir)

result_file = model_path + '.inference.txt'

input_file = sys.argv[2]

batch_size = 512
if len(sys.argv) > 3:
  batch_size = int(sys.argv[3])

command = """python /home/gezi/mine/hasky/deepiu/tools/caption-discriminant-inference.py \\
  --vocab=/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic-finetune/'vocab.txt' \\
  --model_dir={} \\
  --input_file={} \\
  --test_image_dir=/home/gezi/new2/data/ai_challenger/ai_challenger_caption_test1_20170923/caption_test1_images_20170923/ \\
  --batch_size_={} \\
  --result_file={} \\
  """.format(model_path, input_file, batch_size, result_file)

print(command)

os.system(command)
