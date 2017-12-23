#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   mil.py
#        \author   chenghuige  
#          \date   2017-11-19
#   \Description   multiple instance learning
#                  use_idf_weights seems better then use_weights(tf*idf) but
#                  seems bow method not works very well
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags 
#import gflags as flags
FLAGS = flags.FLAGS

##below in discriminant_trainer.py 
# flags.DEFINE_string('image_mlp_dims', '512,512', '')
# flags.DEFINE_string('text_mlp_dims', '512,512', '')


# flags.DEFINE_boolean('fix_image_embedding', False, 'image part all fixed, so just fintune text part')
# flags.DEFINE_boolean('fix_text_embedding', False, 'text part all fixed, so just fintune image part')

flags.DEFINE_string('mil_combiner', 'mean', 'use mean or use max or none for bow/rnn/cnn which just outpt one feature')
flags.DEFINE_bool('mil_baseline', False, 'baseline is just bow')
flags.DEFINE_string('mil_text_mlp_dims', None, '')
flags.DEFINE_string('mil_strategy', 'max', 'max or avg')


import os
import sys
import functools

import melt
logging = melt.logging
import melt.slim2

import deepiu
from deepiu.util import vocabulary
from deepiu.seq2seq import embedding, encoder_factory
from deepiu.util.rank_loss import  pairwise_loss, normalize, pairwise_correct_ratio
from deepiu.image_caption.conf import IMAGE_FEATURE_LEN, NUM_RESERVED_IDS
from deepiu.util import idf

import numpy as np
import dill

class MilTrainer(object):
  def __init__(self, is_training=True, is_predict=False):
    super(MilTrainer, self).__init__()
    self.is_training = is_training
    self.is_predict = is_predict

    logging.info('emb_dim:{}'.format(FLAGS.emb_dim))
    logging.info('margin:{}'.format(FLAGS.margin))

    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 
    self.vocab_size = vocab_size
    self.emb = embedding.get_or_restore_embedding_cpu()

    melt.visualize_embedding(self.emb, FLAGS.vocab)
    if is_training and FLAGS.monitor_level > 0:
      melt.monitor_embedding(self.emb, vocabulary.vocab, vocab_size)

    self.image_process_fn = lambda x: x
    if not FLAGS.pre_calc_image_feature:
      assert melt.apps.image_processing.image_processing_fn is not None, 'forget melt.apps.image_processing.init()'
      self.image_process_fn = functools.partial(melt.apps.image_processing.image_processing_fn,
                                                height=FLAGS.image_height, 
                                                width=FLAGS.image_width,
                                                trainable=FLAGS.finetune_image_model,
                                                is_training=is_training,
                                                random_crop=FLAGS.random_crop_image,
                                                finetune_end_point=FLAGS.finetune_end_point,
                                                distort=FLAGS.distort_image,
                                                feature_name=FLAGS.image_endpoint_feature_name)  

    ImageEncoder = deepiu.seq2seq.image_encoder.Encoders[FLAGS.image_encoder]
    self.image_encoder = ImageEncoder(is_training, is_predict, FLAGS.emb_dim)
    self.using_attention = FLAGS.image_encoder != 'ShowAndTell'
    assert self.using_attention

    with tf.variable_scope('text_encoder'):
      if FLAGS.text_encoder:
        self.text_encoder = encoder_factory.get_encoder(FLAGS.text_encoder, is_training, is_predict)
      else:
        self.text_encoder = None

    self.weights_initializer = tf.random_uniform_initializer(-FLAGS.initializer_scale, FLAGS.initializer_scale)
    self.activation = melt.activations[FLAGS.activation]
    self.text_mlp_dims = [int(x) for x in FLAGS.text_mlp_dims.split(',')] if FLAGS.text_mlp_dims is not '0' else None
    self.biases_initializer = melt.slim2.init_ops.zeros_initializer if FLAGS.bias else None

    logging.info('mil text_encoder:{}'.format(self.text_encoder))

    if FLAGS.use_idf_weights:
      self.idf_weights = tf.constant(idf.get_idf())
    else:
      self.idf_weights = tf.constant([0.] * NUM_RESERVED_IDS + [1.0 for id in range(NUM_RESERVED_IDS, vocab_size)])

    self.scope = FLAGS.trainer_scope or 'image_text_sim'

  def process(self, image_feature):
    if self.image_process_fn is not None:
      image_feature = self.image_process_fn(image_feature) 
    if self.using_attention:
      image_feature_len = FLAGS.image_feature_len or IMAGE_FEATURE_LEN
      image_feature = tf.reshape(image_feature, [-1, FLAGS.image_attention_size, int(image_feature_len / FLAGS.image_attention_size)])
    tf.add_to_collection('image_feature', image_feature)
    return image_feature

  def gen_text_feature(self, text, emb):
    """
    this common interface, ie may be lstm will use other way to gnerate text feature
    """
    with tf.variable_scope('text_encoder'):
      if self.text_encoder is None:
        text_feature = tf.nn.embedding_lookup(emb, text)
      else:
        try:
          results = self.text_encoder.encode(text, self.emb, output_method='all')
        except Exception:
          results = self.text_encoder.encode(text, self.emb)
        try:
          context_feature, states = results
          #attention_states = tf.concat([states, tf.expand_dims(context_feature, 1)])
          #return attention_states
          text_feature = context_feature
        except Exception:
          feature = results
          text_feature = tf.expand_dims(feature, 1)
      return text_feature

  def forward_image_layers(self, image_feature):
    attention_states, initial_state, image_emb = self.image_encoder.encode(self.process(image_feature))
    # # TODO maybe ignore this ? just using one image encoder doing this?
    # attention_states = tf.concat([attention_states, tf.expand_dims(image_emb, 1)], 1)
    if FLAGS.mil_baseline:
      attention_states = tf.slice(attention_states, [0, 0, 0], [-1, 1, -1])
    return attention_states

  def forward_text_layers(self, text_feature):
    dims = self.text_mlp_dims
    if not dims:
      return text_feature

    return melt.slim2.mlp(text_feature, 
                         dims, 
                         self.activation, 
                         weights_initializer=self.weights_initializer,
                         biases_initializer=self.biases_initializer, 
                         scope='text_mlp')

  def forward_text_feature(self, text_feature):
    text_feature = self.forward_text_layers(text_feature)
    #for pointwise comment below
    #must be -1 not 1 for num_negs might > 1 if lookup onece..
    text_feature = normalize(text_feature)
    return text_feature	

  def forward_text(self, text):
    """
    Args:
    text: batch text [batch_size, max_text_len]
    """
    text_feature = self.gen_text_feature(text, self.emb)
    text_feature = self.forward_text_feature(text_feature)
    return text_feature

  def forward_image_feature(self, image_feature):
    """
    Args:
      image: batch image [batch_size, image_feature_len]
    """
    image_feature = self.forward_image_layers(image_feature)
    
    #for point wise comment below
    image_feature = normalize(image_feature)

    return image_feature

  def compute_image_text_sim(self, normed_image_feature, normed_text_feature, 
                            text=None, weights=None, combiner=None):
    #[batch_size, hidden_size]
    if FLAGS.fix_image_embedding:
      normed_image_feature = tf.stop_gradient(normed_image_feature)

    if FLAGS.fix_text_embedding:
      #not only stop internal text ebmeddding but also mlp part so fix final text embedding
      normed_text_feature = tf.stop_gradient(normed_text_feature)
    
    # if not self.is_predict:
    #   normed_text_feature = normed_text_feature[:, :4, :]
    #   if weights is not None:
    #     weights = weights[:, :4]
    #   if text is not None:
    #     text = text[:, :4]
    # jupter/tensorflow/multiple_instance.ipynb
    # normed_text_feature [batch_size, text_len, emb_dim]  normed_image_feature [batch_size, num_image_features, emb_dim] 
    # -> [batch_size, text_len, num_image_features]
    aligments = melt.dot(normed_text_feature, normed_image_feature)

    if FLAGS.mil_strategy == 'max':
      # [batch_size, text_len]
      scores = tf.reduce_max(aligments, -1)
    else:
      # [batch_size, text_len, emb_dim]
      context_feature = tf.matmul(aligments, normed_image_feature)
      scores = melt.element_wise_cosine_nonorm(normed_text_feature, context_feature, keep_dims=False)

    # TODO only training ? eval and predict also use ?
    # if self.is_training:
    #assert text is not None or weights is not None
    if FLAGS.mil_combiner != 'none':
      if weights is None and text is not None:
        # use idf weights
        if self.idf_weights is not None:
          weights = tf.nn.embedding_lookup(self.idf_weights, text)
        else:
          weights = tf.to_float(tf.sign(text))
        # tf idf weights or idf weights
        tf.add_to_collection('targets', text)
        tf.add_to_collection('weights', weights)
    
      # if self.is_training and not FLAGS.train_loss_per_example:
      #   # per word loss instead of per example loss
      #   scores = tf.reshape(scores, [-1, 1])
      #   weights = tf.reshape(weights, [-1, 1])
      #weights = tf.nn.l2_normalize(weights, 1)

    if weights is not None:
      combiner = combiner or FLAGS.mil_combiner
      if combiner == 'mean':
        scores = tf.reduce_sum(tf.multiply(scores, weights), 1)
        total_size = tf.reduce_sum(weights, 1)    
        total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
        scores = scores / total_size
      elif combiner == 'max':
        scores = tf.reduce_max(tf.multiply(scores, weights), 1)
      elif combiner == 'sum':
        scores = tf.reduce_sum(tf.multiply(scores, weights), 1)
      else:
        raise ValueError('unsupport combiner %s' % combiner)

    # if self.is_training:
    #   scores = tf.reduce_mean(scores)
    #   scores = tf.reshape(scores, [-1, 1])

    if not self.is_predict:
      scores = tf.expand_dims(scores, 1)
    
    return scores

  def build_train_graph(self, image_feature, text, neg_image_feature, neg_text, 
                        weights=None, neg_weights=None):
    lookup_negs_once = True if self.text_encoder is None else False
    return self.build_graph(image_feature, text, neg_image_feature, 
                            neg_text, lookup_negs_once=lookup_negs_once, 
                            weights=weights, neg_weights=neg_weights)

  def build_graph(self, image_feature, text, neg_image_feature, neg_text, 
                  lookup_negs_once=False, weights=None, neg_weights=None):
    """
    Args:
    image_feature: [batch_size, IMAGE_FEATURE_LEN]
    text: [batch_size, MAX_TEXT_LEN]
    neg_text: [batch_size, num_negs, MAXT_TEXT_LEN]
    neg_image_feature: None or [batch_size, num_negs, IMAGE_FEATURE_LEN]
    """
    assert (neg_text is not None) or (neg_image_feature is not None)
    with tf.variable_scope(self.scope) as scope:
      #-------------get image feature
      #[batch_size, attenion_size + 1, hidden_size] <= [batch_size, IMAGE_FEATURE_LEN] 
      image_feature = self.forward_image_feature(image_feature)
      #--------------get image text sim as pos score
      #[batch_size, MAX_WORDS, hidden_size]
      text_feature = self.forward_text(text)

      pos_score = self.compute_image_text_sim(image_feature, text_feature, 
                                              text, weights)
      
      scope.reuse_variables()
      
      #--------------get image neg texts sim as neg scores
      #[batch_size, num_negs, text_MAX_WORDS, emb_dim] -> [batch_size, num_negs, emb_dim]  
      neg_scores_list = []
      if neg_text is not None:
        if lookup_negs_once:
          neg_text_feature = self.forward_text(neg_text)
    
        num_negs = neg_text.get_shape()[1]
        for i in range(num_negs):
          if lookup_negs_once:
            neg_text_feature_i = neg_text_feature[:, i, :]
          else:
            neg_text_feature_i = self.forward_text(neg_text[:, i, :])
          neg_scores_i = self.compute_image_text_sim(image_feature, neg_text_feature_i, neg_text[:, i], neg_weights)
          neg_scores_list.append(neg_scores_i)
      if neg_image_feature is not None:
        num_negs = neg_image_feature.get_shape()[1]
        for i in range(num_negs):
          neg_image_feature_feature_i = self.forward_image_feature(neg_image_feature[:, i, :])
          neg_scores_i = self.compute_image_text_sim(neg_image_feature_feature_i, text_feature, text, weights)
          neg_scores_list.append(neg_scores_i)

      #[batch_size, num_negs]
      neg_scores = tf.concat(neg_scores_list, 1)

      #[batch_size, 1 + num_negs]
      scores = tf.concat([pos_score, neg_scores], 1)
      tf.add_to_collection('scores', scores)

      if self.is_training:
        tf.add_to_collection('train_scores', scores)
      elif not self.is_predict:
        tf.add_to_collection('eval_scores', scores)

      loss = pairwise_loss(pos_score, neg_scores)
      loss = tf.identity(loss, name='loss')

      correct_ratio = pairwise_correct_ratio(pos_score, neg_scores)
      self.correct_ratio = tf.identity(correct_ratio, name='correct_ratio') 

    return loss
