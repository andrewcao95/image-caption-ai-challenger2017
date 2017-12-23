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

result_file = model_path + '.evaluate-inference.txt'

base = '../../../mount/'
image_checkpoint_dir = '%s/data/image_model_check_point/' % base
image_checkpoint_file_name = 'inception_resnet_v2.ckpt'

# local run please use 300, 100 since p40 24g, gtx1080ti 11g
# unfortuanately p40 run this inference still has bug, as some model will inference with 
# some pics totally wrong while other pics ok, and some models will be all ok 
# TODO DEBUG  I can not reproduce this on gtx1080ti or k40 machine so not run it right now
# it seems p40 train and evaluate is ok,.. that's lucky..
is_p40 = 1
if is_p40:
  multiplier = 2
else:
  multiplier = 1
batch_size = 300 * multiplier
small_batch_size = 50
if len(sys.argv) > 2:
  image_checkpoint_file_name = sys.argv[2] + '.ckpt'
elif 'resnet152' in model_dir:
  image_checkpoint_file_name = 'resnet_v2_152.ckpt'
elif 'inceptionV4' in model_dir:
  image_checkpoint_file_name = 'inception_v4.ckpt'
elif 'nasnet' in model_dir:
  image_checkpoint_file_name = 'nasnet_large/model.ckpt'
  batch_size = small_batch_size
elif 'nasmobile' in model_dir:
  image_checkpoint_file_name = 'nasnet_mobile/model.ckpt'

image_checkpoint_file = os.path.join(image_checkpoint_dir, image_checkpoint_file_name)
vocab_path = '%s/temp/image-caption/ai-challenger/tfrecord/seq-basic-finetune/vocab.txt' % base 
test_image_dir = '%s/data/ai_challenger/image_caption/caption_validation_images_20170910/' % base

command = """python ./scripts/ai/caption-inference.py \\
  --vocab={} \\
  --model_dir={} \\
  --test_image_dir={} \\
  --buffer_size={} \\
  --result_file={} \\
  --image_checkpoint_file={} \\
  """.format(vocab_path, model_path, test_image_dir, batch_size, result_file, image_checkpoint_file)

print(command, file=sys.stderr)
os.system(command)
