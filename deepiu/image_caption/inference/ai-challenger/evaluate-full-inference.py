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


model_dir = sys.argv[1]

model_dir, model_path = melt.get_model_dir_and_path(model_dir)

result_file = model_path + '.evaluate-full-inference.txt'

command = """python /home/gezi/mine/hasky/deepiu/tools/caption-inference.py --vocab=/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic-finetune/'vocab.txt' \\
  --model_dir={} \\
  --full_text=1 \\
  --test_image_dir=/home/gezi/data2/data/ai_challenger/ai_challenger_caption_validation_20170910/caption_validation_images_20170910 \\
  --buffer_size=300 --result_file={} \\
  """.format(model_path, result_file)

print(command)

os.system(command)
