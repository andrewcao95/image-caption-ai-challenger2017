#!/usr/bin/env python
# ==============================================================================
#          \file   caption2html.py
#        \author   chenghuige  
#          \date   2017-09-29 21:36:34.791169
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('result_dir', None, '')
flags.DEFINE_string('img_dir', None, '')
flags.DEFINE_integer('max_lines', 0, '')
flags.DEFINE_integer('imgs_per_page', 200, '')
flags.DEFINE_integer('imgs_per_row', 5, '')
flags.DEFINE_bool('header', False, '')
flags.DEFINE_string('type', None, '')

import sys, os

count = 0
page = 0

result_dir = FLAGS.result_dir or sys.argv[1]
os.system('mkdir -p %s'%result_dir)
ofile = os.path.join(result_dir, '%03d.html'%page)
out = open(ofile, 'w')

img_dir = 'D:\\new2\\data\\ai_challenger\\image_caption\\pic\\'
if FLAGS.img_dir:
  img_dir = FLAGS.img_dir 
elif FLAGS.type:
  if FLAGS.type == 'inference':
    img_dir = 'D:\\new2\\data\\ai_challenger\\ai_challenger_caption_test1_20170923\\caption_test1_images_20170923\\'


img_html = '<p> <td><a href={0} target=_blank><img src={0} height=250 width=250></a></p> {1} <br /> {2} <br /> {3}<td>'
img_html2 = '<p> <td><a href={0} target=_blank><img src={0} height=250 width=250></a></p> {1} <br /> {2} <td>'

for line in sys.stdin:
  if FLAGS.header:
    FLAGS.header = False
    continue

  l = line.strip().split('\t', 2)
  
  img_id = l[0]
  img = '%s\\%s.jpg'%(img_dir, img_id)
  caption = l[1]

  if len(l) > 2:
    info = l[2]
  else:
    info = ''

  if count % FLAGS.imgs_per_row == 0:
    print('<table><tr>', file=out)
  if info:
    print(img_html.format(img, caption, info, img_id), file=out)
  else:
    print(img_html2.format(img, caption, img_id), file=out)
  if (count + 1) % FLAGS.imgs_per_row == 0:
    print('</tr></table>', file=out)

  count += 1

  if count % FLAGS.imgs_per_page == 0:
    page += 1
    out.close()
    ofile = os.path.join(result_dir, '%03d.html'%page)
    out = open(ofile, 'w')

  if count == FLAGS.max_lines:
    break

out.close()
