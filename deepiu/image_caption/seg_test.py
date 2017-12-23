#!/usr/bin/env python
# -*- coding: gbk -*-
# ==============================================================================
#          \file   test_seg.py
#        \author   chenghuige  
#          \date   2016-09-05 11:48:05.006754
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import gezi
import libsegment

seg = gezi.Segmentor()

print('\t'.join(seg.Segment('��Ůһ��Ҫ֧��')))
print('\x01'.join(seg.Segment('Oh q the same thing to me')))
print('\x01'.join(seg.Segment('Oh q the same thing to me', 'phrase_single')))
print('\x01'.join(seg.Segment('Oh q the same thing to me', 'phrase')))
print('\t'.join(seg.Segment('����')))
print('\t'.join(seg.segment('����')))
print('\t'.join(seg.segment_phrase('����')))
print('\t'.join(gezi.seg.Segment('����', libsegment.SEG_NEWWORD)))
print('\t'.join(gezi.seg.Segment('����')))

print('|'.join(gezi.segment_char('a baby is looking at �ҵ�С���oh �Ҳ�no noû��ϵ �ǲ���   tian, that not ')))


from libword_counter import Vocabulary

v = Vocabulary('/home/gezi/temp/textsum/tfrecord/seq-basic.10w/train/vocab.txt', 2)
print(v.id('��Ů'))
print(v.key(v.id('��Ů')))
