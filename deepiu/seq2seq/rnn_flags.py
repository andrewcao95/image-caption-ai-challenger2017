#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rnn_flags.py
#        \author   chenghuige  
#          \date   2016-12-24 17:02:28.330058
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
#import gflags as flags
FLAGS = flags.FLAGS
 
flags.DEFINE_string('cell', 'basic_lstm', 'might set to lstm_block which is faster, or lstm_block_fused, cudnn_lstm even faster')
flags.DEFINE_integer('num_layers', 1, 'or > 1')
flags.DEFINE_integer('encoder_num_layers', None, 'or > 1')
flags.DEFINE_float('encoder_keep_prob', None, '')
flags.DEFINE_integer('decoder_num_layers', None, 'or > 1')
flags.DEFINE_float('decoder_keep_prob', None, '')

flags.DEFINE_boolean('feed_initial_sate', False, """set true just like ptb_word_lm to feed 
                                                  last batch final state to be inital state 
                                                  but experiments not show better result(similar)""")
flags.DEFINE_integer('rnn_hidden_size', 512, 'rnn cell state hidden size, follow im2txt set default as 512')
