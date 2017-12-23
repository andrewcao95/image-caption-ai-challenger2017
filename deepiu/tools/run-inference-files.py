#!/usr/bin/env python
# ==============================================================================
#          \file   run-evaluate-inference-evaluate-dir.py
#        \author   chenghuige  
#          \date   2017-10-07 11:57:11.430650
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import glob 

infile = sys.argv[1]
#for file in glob.glob(dir + '/model*ckpt*.index'):
for file in open(infile):
  file = file.strip().split()[-1]
  file = file.replace('.index', '')
  model_dir, model_path = melt.get_model_dir_and_path(file)
  result_file = model_path + '.inference.txt'
  command = """python /home/gezi/mine/hasky/deepiu/tools/caption-inference.py --vocab=/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic-finetune/'vocab.txt' \\
    --model_dir={} \\
    --test_image_dir=/home/gezi/new2/data/ai_challenger/ai_challenger_caption_test1_20170923/caption_test1_images_20170923/ \\
    --buffer_size=300 --result_file={} \\
  """.format(model_path, result_file)
  print(command, file=sys.stderr)
  os.system(command)
