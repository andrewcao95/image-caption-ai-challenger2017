#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   input_flags.py
#        \author   chenghuige  
#          \date   2016-12-25 00:17:18.268341
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
#import gflags as flags
FLAGS = flags.FLAGS
  
#--------- read data
flags.DEFINE_integer('fixed_eval_batch_size', 30, """must >= num_fixed_evaluate_examples
                                                     if == real dataset len then fix sequence show
                                                     if not == can be show different fixed each time
                                                     usefull if you want only show see 2 and 
                                                     be different each time
                                                     if you want see 2 by 2 seq
                                                     then num_fixed_evaluate_example = 2
                                                          fixed_eval_batch_size = 2
                                                  """)

flags.DEFINE_integer('num_fixed_evaluate_examples', 30, '')
flags.DEFINE_integer('num_evaluate_examples', 1, '')

flags.DEFINE_integer('num_threads', 12, """threads for reading input tfrecords,
                                           setting to 1 may be faster but less randomness
                                        """)

flags.DEFINE_boolean('shuffle_files', True, '')
flags.DEFINE_boolean('batch_join', True, '')
flags.DEFINE_boolean('shuffle_batch', True, '')

flags.DEFINE_boolean('shuffle_then_decode', True, 
                     """ actually this is decided by is_sequence_example.. 
                     if is_sequence_example then False, if just example not sequence then True since is sparse
                     TODO remove this
                     """)
flags.DEFINE_boolean('is_sequence_example', False, '')
flags.DEFINE_string('buckets', '', 'empty meaning not use, other wise looks like 5,10,15,30')

flags.DEFINE_boolean('dynamic_batch_length', True, 
                     """very important False means all batch same size! 
                        otherwise use dynamic batch size
                        Now only not sequence_example data will support dyanmic_batch_length=False
                        Also for cnn you might need to set to False to make all equal length batch used
                        """)

flags.DEFINE_boolean('use_weights', False, '''from tfrecord per example word, usaually tf*idf weight, 
                                              same word has difference score in different example/instances''')
flags.DEFINE_boolean('use_idf_weights', False, 'idf only weight vocab based fixed for each vocab word')
  

flags.DEFINE_integer('num_negs', 1, '0 means no neg')

flags.DEFINE_boolean('feed_dict', False, 'depreciated, too complex, just prepare your data at first for simple')

#---------- input reader
flags.DEFINE_integer('min_after_dequeue', 0, """by deafualt will be 500, 
                                                set to large number for production training 
                                                for better randomness""")
flags.DEFINE_integer('num_prefetch_batches', 0, '')


#---------- input dirs
#@TODO will not use input pattern but use dir since hdfs now can not support glob well
flags.DEFINE_string('train_input', None, 'must provide')
flags.DEFINE_string('valid_input', None, 'if empty will train only')
flags.DEFINE_string('fixed_valid_input', None, 'if empty wil  not eval fixed images')
flags.DEFINE_string('num_records_file', None, '')
flags.DEFINE_integer('min_records', 12, '')
flags.DEFINE_integer('num_records', 0, 'if not 0, will check equal')
flags.DEFINE_string('base_dir', '../../../mount', '')

#--for scene
flags.DEFINE_string('scene_train_input', None, '')
flags.DEFINE_string('scene_valid_input', None, '')
flags.DEFINE_boolean('use_scene_embedding', True, '')

#----------eval
flags.DEFINE_boolean('legacy_rnn_decoder', False, '')
flags.DEFINE_boolean('experiment_rnn_decoder', False, '')

flags.DEFINE_boolean('show_eval', True, '')

flags.DEFINE_boolean('eval_shuffle_files', True, '')
flags.DEFINE_boolean('eval_fix_random', True, '')
flags.DEFINE_integer('eval_seed', 1024, '')

flags.DEFINE_integer('seed', 1024, '')

flags.DEFINE_boolean('fix_sequence', False, '')

#----------strategy 

flags.DEFINE_string('seg_method', 'default', '')
flags.DEFINE_boolean('feed_single', False, '')

flags.DEFINE_boolean('gen_predict', True, '')


flags.DEFINE_string('decode_name', 'text', '')
flags.DEFINE_string('decode_str_name', 'text_str', '')

flags.DEFINE_boolean('reinforcement_learning', False, '')
flags.DEFINE_float('reinforcement_ratio', 1., '')

#--------for image caption  TODO move to image_caption/input_flags.py ?
#--if use image dir already info in image_features
flags.DEFINE_string('image_dir', None, 'input images dir')

flags.DEFINE_boolean('pre_calc_image_feature', False, 'will set to true if not has image model auto in train.py')
flags.DEFINE_boolean('has_image_model', False, '')

flags.DEFINE_string('image_checkpoint_file', '/home/gezi/data/image_model_check_point/inception_resnet_v2_2016_08_30.ckpt', '''None means image model from scratch''')
flags.DEFINE_boolean('finetune_image_model', True, '''by default will be finetune otherwise
                                                   why not pre calc image feature much faster
                                                   but we also support''')
flags.DEFINE_string('finetune_end_point', None, 'if not None, only finetune from some ende point layers before will freeze')
                                                   
flags.DEFINE_boolean('distort_image', True, 'training option')
flags.DEFINE_boolean('random_crop_image', True, 'training option')

flags.DEFINE_string('image_model_name', None, 'InceptionResnetV2 might be by default but can get from checkpoint direclty')
flags.DEFINE_string('image_endpoint_feature_name', None, 'mostly None for showandtell, not None for show attend and tell features if not in endpoint dict')
flags.DEFINE_integer('image_attention_size', None, 'InceptionResnetV2 will be 64')
flags.DEFINE_integer('image_feature_len', None, '')
flags.DEFINE_integer('image_width', None, 'default width of inception 299, resnet 224 but google pretrain resnet v2 models also 299')
flags.DEFINE_integer('image_height', None, 'default height of inception 299, resnet 224 but google pretrain resnet v2 models also 299')
flags.DEFINE_boolean('image_features2feature', False, '''for show and tell if input is pre calc atteniton features input
                                                                    here set True will process attention features 
                                                                    to generate and use final feature jus similar like input without 
                                                                    attention vectors''')

# for image classification
flags.DEFINE_integer('num_image_classes', None, '')
flags.DEFINE_integer('num_pretrain_image_classes', None, 'HACK for using pretrain image models where class num is not 1001 as imagenet 1k label models')
flags.DEFINE_integer('image_top_k', 3, '')

flags.DEFINE_string('scene_model', None, 'if not None will use scene_model otherwise it is scene_model path')
flags.DEFINE_string('scene_cats', '/home/gezi/mine/hasky/deepiu/scene/place365/cat_names_cn.txt', '')
flags.DEFINE_integer('scene_feature_len', 15, '')

                                                  
#---in melt.apps.image_processing.py
#flags.DEFINE_string('image_model_name', 'InceptionV3', '')
flags.DEFINE_string('one_image', '/home/gezi/data/flickr/flickr30k-images/1000092795.jpg', '')

flags.DEFINE_string('image_feature_name', 'image_feature', 'for decoding tfrecord')


#---------negative smapling
flags.DEFINE_boolean('neg_left', False, 'ltext or image')
flags.DEFINE_boolean('neg_right', True, 'rtext or text')


#---------discriminant trainer

flags.DEFINE_string('activation', 'relu', 
                    """relu/tanh/sigmoid  seems sigmoid will not work here not convergent
                    and relu slightly better than tanh and convrgence speed faster""")
flags.DEFINE_boolean('bias', False, 'wether to use bias. Not using bias can speedup a bit')

flags.DEFINE_boolean('elementwise_predict', False, '')


flags.DEFINE_float('keep_prob', 1., 'or 0.9 0.8 0.5')
flags.DEFINE_float('dropout', 0., 'or 0.9 0.8 0.5')

flags.DEFINE_string('trainer_scope', None, '')


#----- encoder 
flags.DEFINE_string('image_encoder', 'ShowAndTell', '')
flags.DEFINE_string('text_encoder', None, '')
