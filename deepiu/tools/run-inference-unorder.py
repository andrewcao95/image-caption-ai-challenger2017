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

image_checkpoint_dir = '/home/gezi/data/image_model_check_point/'
image_checkpoint_file_name = 'inception_resnet_v2_2016_08_30.ckpt'

batch_size=300 

model_dir = os.getcwd()
if len(sys.argv) > 2:
  image_checkpoint_file_name = sys.argv[2] + '.ckpt'
elif 'resnet152' in model_dir:
  image_checkpoint_file_name = 'resnet_v2_152.ckpt'
elif 'inceptionV4' in model_dir:
  image_checkpoint_file_name = 'inception_v4.ckpt'
elif 'nasnet' in model_dir:
  image_checkpoint_file_name = 'nasnet_large/model.ckpt'
  batch_size = 50
elif 'nasmobile' in model_dir:
  image_checkpoint_file_name = 'nasnet_mobile/model.ckpt'
image_checkpoint_file = os.path.join(image_checkpoint_dir, image_checkpoint_file_name)

if len(sys.argv) > 3:
	batch_size = int(sys.argv[3])

command = """python /home/gezi/mine/hasky/deepiu/tools/caption-inference-unorder.py \\
  --vocab=/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic-finetune/'vocab.txt' \\
  --model_dir={} \\
  --test_image_dir=/home/gezi/new2/data/ai_challenger/ai_challenger_caption_test1_20170923/caption_test1_images_20170923/ \\
  --buffer_size={} \\
  --result_file={} \\
  --image_checkpoint_file={} \\
  """.format(model_path, batch_size, result_file, image_checkpoint_file)

print(command)

os.system(command)
