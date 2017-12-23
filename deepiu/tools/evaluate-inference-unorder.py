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

#for remote HACK

result_file = result_file.replace('hdfsmount', 'mount')

image_checkpoint_dir = '/home/gezi/data/image_model_check_point/'
image_checkpoint_file_name = 'inception_resnet_v2.ckpt'

def has_model(model_dir, model_name):
  return model_name in model_dir or model_name in os.getcwd()

batch_size = 300
# can be 300, for gpu 0 safe just use 200 
if os.environ['CUDA_VISIBLE_DEVICES'] == '0':
  batch_size=200
if len(sys.argv) > 3:
  image_checkpoint_file_name = sys.argv[3] + '.ckpt'
elif has_model(model_dir, 'resnet152'):
  image_checkpoint_file_name = 'resnet_v2_152.ckpt'
elif has_model(model_dir, 'inceptionV4'):
  image_checkpoint_file_name = 'inception_v4.ckpt'
elif has_model(model_dir, 'nasnet'):
  image_checkpoint_file_name = 'nasnet_large/model.ckpt'
  batch_size = 50  # hack for nasnet 50 can produce correct result 
elif has_model(model_dir, 'nasmobile'):
  image_checkpoint_file_name = 'nasnet_mobile/model.ckpt'

image_checkpoint_file = os.path.join(image_checkpoint_dir, image_checkpoint_file_name)

command = """python /home/gezi/mine/hasky/deepiu/tools/caption-inference-unorder.py \\
  --vocab=/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic-finetune/'vocab.txt' \\
  --model_dir={} \\
  --test_image_dir=/home/gezi/data2/data/ai_challenger/ai_challenger_caption_validation_20170910/caption_validation_images_20170910 \\
  --buffer_size={} \\
  --result_file={} \\
  --image_checkpoint_file={} \\
  """.format(model_path, batch_size, result_file, image_checkpoint_file)

print(command)

os.system(command)
