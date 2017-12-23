#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   show_and_tell.py
#        \author   chenghuige  
#          \date   2016-09-04 17:49:20.030172
#   \Description  
# ==============================================================================
"""
show and tell not just follow paper it can be viewed as a simple generative method frame work
also support show attend and tell using this framework
there are slightly difference with google open source im2txt see FLAGS.image_as_init_state
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
#import gflags as flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('image_as_init_state', False, 'by default will treat image as input not inital state(im2txt usage)')
flags.DEFINE_boolean('show_atten_tell', False, 'wether to use attention as in paper Show,Attend and Tell: Neeural Image Caption Generation with Visual Attention')
flags.DEFINE_float('alignment_loss_ratio', 0., '')
flags.DEFINE_string('showtell_encode_scope', '', '')
flags.DEFINE_string('showtell_decode_scope', '', '')
flags.DEFINE_float('discriminant_loss_ratio', 0., '')

flags.DEFINE_float('mil_loss_ratio', 0., 'multi instance loss, see <<From Captions to Visual Concepts and Back>>')
flags.DEFINE_boolean('showtell_noimage', False, 'if True then only language model')

import os 
import sys
import functools
import math
import dill

import melt
logging = melt.logging
from deepiu.image_caption import conf 
from deepiu.image_caption.conf import IMAGE_FEATURE_LEN, TEXT_MAX_WORDS, NUM_RESERVED_IDS
from deepiu.util import vocabulary
from deepiu.seq2seq import embedding
import deepiu

from deepiu.seq2seq import embedding, encoder_factory
from deepiu.util.rank_loss import dot, compute_sim, pairwise_loss, normalize

from deepiu.util import idf

slim = tf.contrib.slim
  
#must outside ShowAndTell class and pass to share to pass ShowAndTell ShowAndTellPredictor
#based on <<Self-critical Sequence Training for Image Captioning>>
class RLInfo(object):
  def __init__(self):
    self.rewards_feed = tf.placeholder_with_default([0.], [None], name='rewards')
    self.baseline_feed = tf.placeholder_with_default([0.], [None], name='baseline')
    self.sampled_caption = None
    self.greedy_caption = None 

class SceneInfo(object):
  def __init__(self, batch_size=None):
    #self.feature_feed = tf.placeholder_with_default([[0.] * FLAGS.scene_feature_len], [None, FLAGS.scene_feature_len], name='scene_feature')
    #self.feature_feed = tf.placeholder(tf.float32, [None, FLAGS.scene_feature_len], name='scene_feature')
    if batch_size is None:
      batch_size = 1
    FLAGS.scene_feature_len = 365
    #self.feature_feed = tf.placeholder_with_default([[0.] * FLAGS.scene_feature_len] * batch_size, [None, FLAGS.scene_feature_len], name='scene_feature')
    self.feature_feed = tf.placeholder_with_default([[0.] * FLAGS.scene_feature_len] * batch_size, [None, FLAGS.scene_feature_len], name='scene_feature')
    
class ShowAndTell(object):
  """
  ShowAndTell class is a trainer class
  but has is_training mark for ShowAndTell predictor will share some code here
  3 modes
  train,
  evaluate,
  predict
  """
  def __init__(self, is_training=True, is_predict=False):
    super(ShowAndTell, self).__init__()

    if FLAGS.image_as_init_state:
      #just use default method here is ok!
      assert FLAGS.add_text_start is True, 'need to add text start for im2tx mode'
    #else:
      #just for experiment to be same as im2txt but result is worse
      #assert FLAGS.add_text_start is False, 'normal mode must not add text start'

    self.is_training = is_training 
    self.is_predict = is_predict
    self.is_evaluate = (not is_training) and (not is_predict)

    if FLAGS.showtell_noimage:
      FLAGS.pre_calc_image_feature = True

    #if is_training:
    logging.info('num_sampled:{}'.format(FLAGS.num_sampled))
    logging.info('log_uniform_sample:{}'.format(FLAGS.log_uniform_sample))
    logging.info('keep_prob:{}'.format(FLAGS.keep_prob))
    logging.info('emb_dim:{}'.format(FLAGS.emb_dim))
    logging.info('add_text_start:{}'.format(FLAGS.add_text_start))
    logging.info('zero_as_text_start:{}'.format(FLAGS.zero_as_text_start))

    emb = self.emb = embedding.get_or_restore_embedding_cpu()
    melt.visualize_embedding(self.emb, vocabulary.vocab_path)
    if is_training and FLAGS.monitor_level > 0:
      melt.monitor_embedding(emb, vocabulary.vocab, vocabulary.vocab_size)

    self.idf_weights = None
    if FLAGS.use_idf_weights:
      #for tf idf same as cider
      self.idf_weights = tf.constant(tf.get_idf())  

    self.emb_dim = FLAGS.emb_dim
    
    self.using_attention = FLAGS.image_encoder != 'ShowAndTell'

    ImageEncoder = deepiu.seq2seq.image_encoder.Encoders[FLAGS.image_encoder]
    #juse for scritps backward compact, TODO remove show_atten_tell
    if FLAGS.show_atten_tell:
      logging.info('warning, show_atten_tell mode depreciated, just set --image_encoder=')
      ImageEncoder = deepiu.seq2seq.image_encoder.MemoryEncoder
    
    self.encoder = ImageEncoder(is_training, is_predict, self.emb_dim)

    self.decoder = deepiu.seq2seq.rnn_decoder.RnnDecoder(is_training, is_predict)
    self.decoder.set_embedding(emb)
    
    #for image finetune with raw image as input 
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
    else:
      self.image_process_fn = None

    self.image_feature_len = FLAGS.image_feature_len or IMAGE_FEATURE_LEN

    if FLAGS.discriminant_loss_ratio > 0:
      encoder_type = 'bow'
      self.encoder2 = encoder_factory.get_encoder(encoder_type, is_training, is_predict)

    self.scene_feature = None
    self.scene_logits_resue = False
    if FLAGS.scene_train_input:
      self.scene_image_process_fn = functools.partial(melt.apps.image_processing.image_processing_fn,
                                                      height=FLAGS.image_height, 
                                                      width=FLAGS.image_width,
                                                      trainable=FLAGS.finetune_image_model,
                                                      is_training=is_training,
                                                      random_crop=FLAGS.random_crop_image,
                                                      finetune_end_point=FLAGS.finetune_end_point,
                                                      distort=FLAGS.distort_image,
                                                      feature_name=None)   

    self.dupimage = False    

  def feed_ops(self):
    """
    return feed_ops, feed_run_ops
    same as ptm example code
    not used very much, since not imporve result
    """
    if FLAGS.reinforcement_learning:
      pass

    if FLAGS.feed_initial_sate:
      return [self.decoder.initial_state], [self.decoder.final_state]
    else:
      return [], []

  def process(self, image_feature):
    if self.image_process_fn is not None:
      if self.dupimage:
        image_feature = self.dupimage_process(image_feature)
      else:
        image_feature = self.image_process_fn(image_feature) 
    if self.using_attention:
      image_feature = tf.reshape(image_feature, [-1, FLAGS.image_attention_size, int(self.image_feature_len / FLAGS.image_attention_size)])
    tf.add_to_collection('image_feature', image_feature)
    return image_feature

  def dupimage_process(self, image_feature):
    processed_image_feature = self.image_process_fn(tf.slice(image_feature, [0], [1]))
    # TODO seems below not work
    preocessed_image_feature = tf.contrib.seq2seq.tile_batch(processed_image_feature, melt.get_batch_size(image_feature)) 
    return processed_image_feature

  def _post_deal_image_embedding(self, image_emb, image):
    # if not self.scene_feature:
    #   if FLAGS.scene_model:
    #     assert self.scene, 'need to set scene forget to set FLAGS.scene_model ? %s'%FLAGS.scene_model
    #     self.scene_feature = self.scene.feature_feed

    # if self.scene_feature:  
    #   image_emb = self.encoder.build_image_embeddings(tf.concat([image_emb, self.scene_feature], -1), name='scene_embedding')

    if FLAGS.scene_train_input and FLAGS.use_scene_embedding:
      logging.info('use scene embedding')
      pre_logits = self.scene_image_process_fn(image)
      scene_logits = slim.fully_connected(pre_logits, FLAGS.num_image_classes, activation_fn=None, scope='SceneLogits')
      image_emb = self.encoder.build_image_embeddings(tf.concat([image_emb, scene_logits], -1), name='scene_embedding')

    return image_emb

  # def _post_deal(self, attention_states, initial_state, image_emb):
  #   if FLAGS.scene_model:
  #     #image_feature_len = FLAGS.image_feature_len or IMAGE_FEATURE_LEN
  #     #attention_feature_len = int(image_feature_len / FLAGS.image_attention_size)
  #     # [batch_size, emb_dim] <- [batch_size, num_scenes] [num_scens, emb_dim]  
  #     scene_emb = tf.contrib.layers.fully_connected(
  #         inputs=self.scene.feature_feed,
  #         num_outputs=self.emb_dim,  #turn image to word embdding space, same feature length
  #         activation_fn=None,
  #         weights_initializer=tf.random_uniform_initializer(minval=-FLAGS.initializer_scale, maxval=FLAGS.initializer_scale),
  #         biases_initializer=None,
  #         scope='scene_fc')
  #     attention_states = tf.concat([attention_states, tf.expand_dims(scene_emb, 1)], 1)
  #   return attention_states, initial_state, image_emb

  def encode(self, image_feature):
    if FLAGS.scene_model:
      if not hasattr(self.encoder, 'scene_feature'):
        self.encoder.scene_feature = self.scene.feature_feed
    attention_states, initial_state, image_emb = self.encoder.encode(self.process(image_feature))
    if FLAGS.image_as_init_state:
      #for im2txt one more step at first, just for exp not used much 
      with tf.variable_scope(self.decoder.scope) as scope:
        zero_state = self.decoder.cell.zero_state(batch_size=melt.get_batch_size(input), dtype=tf.float32)
        _, initial_state = self.decoder.cell(image_emb, zero_state)
        image_emb = None 

    self.image_emb = image_emb
    image_emb = self._post_deal_image_embedding(image_emb, image_feature)
    #attention_states, initial_state, image_emb = self._post_deal(attention_states, initial_state, image_emb)
    return attention_states, initial_state, image_emb   

  def calc_alignment_loss(self, alignment_history, lengths):
    """
    this is important for show attend tell bleu_4 improvment 
    from show attend and tell paper 4.2.1 Doublly Stochatic Attention try to pay attention equally to all image parts
    but seems only imporve at early steps and will decrease at later step for 0.25 ratio
    TODO test smaller ratio as 0.1 or 0.05
    may be use entropy ? to make entorpy low TODO tf.bayes_flow...entropy

    might not need mask since dynamic_decode with impute_finished=True
    """
    #alignment_history [batch_size, num_steps, attention_size]
    #reducd sum to [batch_size, attention_size]
    sequence_mask = tf.expand_dims(tf.to_float(tf.sequence_mask(lengths)), -1)
    alignment_history = alignment_history * sequence_mask
    #print('alignment_history', alignment_history) 
    reduced_attention = tf.reduce_sum(alignment_history, 1)
    #FIXME gradient error tensorflow.python.framework.errors_impl.InvalidArgumentError: Incompatible shapes: [256] vs. [256,64]
    #[[Node: show_and_tell/OptimizeLoss/gradients/show_and_tell/main/sub_grad/BroadcastGradientArgs = BroadcastGradientArgs[T=DT_INT32, _device="/job:localhost/replica:0/task:0/gpu:0"](show_and_tell/OptimizeLoss/gradients/show_and_tell/main/sub_grad/Shape, show_and_tell/OptimizeLoss/gradients/show_and_tell/main/sub_grad/Shape_1)]]
    reduced_attention = tf.expand_dims(tf.to_float(lengths), 1) / float(FLAGS.image_attention_size) - reduced_attention 
    #reduced_attention = 1. - reduced_attention
    reduced_attention *= reduced_attention
    #[batch_size, 1]
    alignment_loss = tf.reduce_mean(reduced_attention)
    #print('alignment_loss', alignment_loss)

    return alignment_loss

  #NOTICE for generative method, neg support removed to make simple!
  def build_graph(self, image_feature, text, 
                  neg_image_feature=None, neg_text=None, 
                  exact_prob=False, exact_loss=False,
                  weights=None):
    
    scope = tf.get_variable_scope()
    if not FLAGS.showtell_noimage:
      with tf.variable_scope(FLAGS.showtell_encode_scope or scope):
        attention_states, initial_state, image_emb = self.encode(image_feature)
        if image_emb is not None:
          assert not FLAGS.add_text_start, 'if use image emb as input then must not pad start mark before sentence'
        else:
          assert FLAGS.add_text_start, 'if not use image emb as input then must pad start mark before sentence'
    else:
      print('Language only mode!', file=sys.stderr)
      image_emb = tf.zeros([melt.get_batch_size(text), self.emb_dim])
      initial_state = None
      attention_states = None

    with tf.variable_scope(FLAGS.showtell_decode_scope or scope):
      #will pad start in decoder.sequence_loss if FLAGS.image_as_init_state
      scores = self.decoder.sequence_loss(text,
                                          input=image_emb, 
                                          initial_state=initial_state, 
                                          attention_states=attention_states, 
                                          exact_prob=exact_prob,
                                          exact_loss=exact_loss,
                                          vocab_weights=self.idf_weights if self.is_training else None,
                                          weights=weights if self.is_training else None) 

      loss = scores 

      if FLAGS.reinforcement_learning and self.is_training:
        assert not FLAGS.image_as_init_state, 'not support im2txt style for reinforcement_learning now, not tested!'
        assert self.rl, 'need to set rl for reinforcement_learning'
        tf.get_variable_scope().reuse_variables()
        max_words = TEXT_MAX_WORDS 
        convert_unk = True
        #code borrow from https://github.com/arieling/SelfCriticalSequenceTraining-tensorflow
        #scores is -(negative log loss)
        sampled_caption, sampled_loss = self.decoder.generate_sequence_multinomial(image_emb, 
                                          max_words=max_words, 
                                          #max_words=16,
                                          initial_state=initial_state,
                                          attention_states=attention_states,
                                          convert_unk=convert_unk,
                                          #length_normalization_factor=0.,
                                          need_logprobs=True)  

        self.rl.sampled_caption = sampled_caption

        greedy_caption, _ = self.decoder.generate_sequence_greedy(image_emb, 
                                          max_words=max_words,
                                          #max_words=20, 
                                          initial_state=initial_state,
                                          attention_states=attention_states,
                                          convert_unk=convert_unk,
                                          need_logprobs=False)

        self.rl.greedy_caption = greedy_caption

        ratio = FLAGS.reinforcement_ratio
        
        #if doing this need loss and sampled_loss same shape batch_size or batch_size * text_length 
        loss = ratio * (self.rl.rewards_feed - self.rl.baseline_feed) * sampled_loss + (1- ratio) * loss

        #loss = -loss

      if not self.is_predict:
        loss = tf.reduce_mean(loss)

      #if not self.is_training and not self.is_predict: #evaluate mode
      if self.is_training:
        tf.add_to_collection('train_scores', scores)
      elif not self.is_predict:
        tf.add_to_collection('eval_scores', scores)

      if FLAGS.discriminant_loss_ratio > 0 and self.is_training:
        assert neg_text is not None
        tf.get_variable_scope().reuse_variables()
        max_words = TEXT_MAX_WORDS 
        convert_unk = True
        greedy_caption, _ = self.decoder.generate_sequence_greedy(image_emb, 
                                  max_words=max_words,
                                  #max_words=20, 
                                  initial_state=initial_state,
                                  attention_states=attention_states,
                                  convert_unk=convert_unk,
                                  need_logprobs=False)
        text_feature = self.encoder2.encode(text, self.emb)
        text_feature = normalize(text_feature)
        # neg_text = neg_text[:, 0, :]
        # neg_text_feature = self.encoder2.encode(neg_text, self.emb)
        # neg_text_feature = normalize(neg_text_feature)
        caption_feature = self.encoder2.encode(greedy_caption, self.emb)
        caption_feature = normalize(caption_feature)
        pos_score = compute_sim(caption_feature, text_feature)
        # neg_score = compute_sim(caption_feature, neg_text_feature)
        tf.add_to_collection('pos_score', pos_score)
        # tf.add_to_collection('neg_score', neg_score)
        # discriminant_loss = pairwise_loss(pos_score, neg_score)
        discriminant_loss = tf.reduce_mean((1. - pos_score) / 2.)
        #TODO this is mean loss so can use reduced loss then add discriminant_loss * ratio
        tf.add_to_collection('discriminant_loss', discriminant_loss)
        ratio = FLAGS.discriminant_loss_ratio
        tf.add_to_collection('gen_loss', loss)
        loss += ratio * discriminant_loss 

      if FLAGS.alignment_history and self.is_training:
        alignment_history = self.decoder.alignment_history
        tf.add_to_collection('alignment_history', alignment_history)

        if FLAGS.alignment_loss_ratio > 0: 
          lengths = self.decoder.final_sequence_lengths
          alignment_loss = self.calc_alignment_loss(alignment_history, lengths)
          tf.add_to_collection('alignment_loss', alignment_loss)
          #alignment_loss might be 4.1 ..
          ratio = FLAGS.alignment_loss_ratio
          #loss = (1 - ratio) * loss + ratio * alignment_loss
          loss += ratio * alignment_loss 

    self.main_loss = loss

    if self.is_predict:
      loss = tf.squeeze(loss)

    return loss

  def build_train_graph(self, image_feature, text, neg_image_feature=None, neg_text=None, weights=None):
    return self.build_graph(image_feature, text, neg_image_feature, neg_text, weights=weights)


  def build_scene_graph(self, images, labels):
    #scene_logits = self.scene_logits_layer()
    pre_logits = self.scene_image_process_fn(images)

    if FLAGS.use_scene_embedding:
      scope = tf.get_variable_scope()
      with tf.variable_scope(FLAGS.showtell_encode_scope or scope):
        scene_logits = slim.fully_connected(pre_logits, FLAGS.num_image_classes, activation_fn=None, scope='SceneLogits', reuse=True)
    else:
      scene_logits = slim.fully_connected(pre_logits, FLAGS.num_image_classes, activation_fn=None, scope='SceneLogits')

    #print('labels:', labels)
    labels = labels[:, 0]
    print('logits', scene_logits, 'labels', labels)
    self.scene_recall_at_k = melt.precision_at_k(scene_logits, labels, FLAGS.image_top_k)
    self.scene_recall_at_1 = melt.precision_at_k(scene_logits, labels, 1)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scene_logits, labels=labels)   
    loss = tf.reduce_mean(loss)

    self.scene_loss = loss
    return loss

