#!/usr/bin/env python
# ==============================================================================
#          \file   image_encoder.py
#        \author   chenghuige  
#          \date   2017-09-17 21:57:04.147177
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
#import gflags as flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('image_features_batch_norm', False, '')
#TODO change to float, keepprob default is 1.0 like this..
flags.DEFINE_boolean('image_features_dropout', False, '')

import sys, os, math
import functools

import nets #slim nets
import tensorflow.contrib.slim as slim
from deepiu.seq2seq import embedding
from deepiu.seq2seq import rnn_encoder
from deepiu.seq2seq import static_loop_encoder

import melt

from deepiu.image_caption.conf import IMAGE_FEATURE_LEN


# TODO  features2featre factor for all image models move to melt
def inception_resnet_v2(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        reuse=None,
                        scope='InceptionResnetV2Help'):
  """Creates the Inception Resnet V2 model.

  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: float, the fraction to keep before final layer.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    create_aux_logits: Whether to include the auxilliary logits.

  Returns:
    logits: the logits outputs of the model.
    end_points: the set of end_points from the inception model.
  """

  with tf.variable_scope(scope, 'InceptionResnetV2Help', [inputs],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
    
      net = inputs
      with tf.variable_scope('Logits'):
        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                              scope='AvgPool_1a_8x8')
        net = slim.flatten(net)
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='Dropout')
    return net

def inception_v4(inputs, is_training=True,
                 dropout_keep_prob=0.8,
                 reuse=None,
                 scope='InceptionV4Help'):
  """Creates the Inception V4 model.

  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: float, the fraction to keep before final layer.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    create_aux_logits: Whether to include the auxiliary logits.

  Returns:
    logits: the logits outputs of the model.
    end_points: the set of end_points from the inception model.
  """
  with tf.variable_scope(scope, 'InceptionV4Help', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net = inputs

      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                          stride=1, padding='SAME'):
        # Final pooling and prediction
        with tf.variable_scope('Logits'):
          # 8 x 8 x 1536
          net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                scope='AvgPool_1a')
          # 1 x 1 x 1536
          net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b')
          net = slim.flatten(net, scope='PreLogitsFlatten')
    return net

# TODO since resnet v2 and v1  is the same except input image 299 and 224 so change scope to ResnetHelp TODO
def resnet(inputs, is_training=True,
             dropout_keep_prob=0.8,
             reuse=None,
             scope='ResnetV2Help'):
  with tf.variable_scope(scope, 'ResnetV2Help', [inputs], reuse=reuse) as scope:
    net = inputs
    net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
  return net

def nasnet(inputs, is_training=True,
           dropout_keep_prob=0.8,
           reuse=None,
           scope='NasnetHelp'):
  from nets.nasnet import nasnet_utils
  # err.. TODO type error of scope name
  with tf.variable_scope(scope, 'ResnetV2Help', [inputs], reuse=reuse) as scope:
  # with tf.variable_scope(scope, 'NasnetHelp', [inputs], reuse=reuse) as scope:
    net = inputs
    # TODO this is actually only for nasnet_large not for mobile etc..
    net = nasnet_utils.global_avg_pool(net, data_format='NHWC')
    net = slim.dropout(net, dropout_keep_prob, scope='dropout')

  return net          

def features2feature(inputs, 
                     is_training=False, 
                     image_model_name=None,
                     image_feature_len=None, 
                     image_attention_size=None,
                     dropout_keep_prob=0.8, 
                     reuse=None):
  if not FLAGS.image_features_dropout:
    is_training = False  

  if FLAGS.image_features_batch_norm:
    inputs = batch_norm(inputs, is_training=is_training)

  image_feature_len = image_feature_len or FLAGS.image_feature_len or IMAGE_FEATURE_LEN
  image_attention_size = image_attention_size or FLAGS.image_attention_size
  fe_dim = int(image_feature_len / image_attention_size)
  attn_dim = int(math.sqrt(image_attention_size))
  inputs = tf.reshape(inputs, [-1, attn_dim, attn_dim, fe_dim])

  #TODO for other image type use other ones
  image_model_name = image_model_name or FLAGS.image_model_name
  if image_model_name == 'InceptionResnetV2':
    return inception_resnet_v2(inputs, is_training, 
                               dropout_keep_prob=dropout_keep_prob, reuse=reuse)
  elif image_model_name == 'InceptionV4':
    return inception_v4(inputs, is_training, 
                        dropout_keep_prob=dropout_keep_prob, reuse=reuse)
  elif image_model_name.startswith('resnet'):
    return resnet(inputs, is_training, dropout_keep_prob, reuse)
  elif image_model_name.startswith('nasnet'):
    return nasnet(inputs, is_training, dropout_keep_prob, reuse)
  else:
    raise TypeError('features2feature not support %s right now' % image_model_name)


class ImageEncoder(object):
  def __init__(self, is_training=False, is_predict=False, 
               emb_dim=None, initializer=None):
    """
    emb_dim means word emb dim
    """
    self.is_training = is_training
    self.is_predict = is_predict
    self.emb_dim = emb_dim
    self.initializer = initializer or tf.random_uniform_initializer(minval=-FLAGS.initializer_scale, maxval=FLAGS.initializer_scale)
  
class ShowAndTellEncoder(ImageEncoder):
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ImageEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)

  def build_image_embeddings(self, image_feature, name='image_embedding', reuse=None):
    with tf.variable_scope(name, reuse=reuse) as scope:
      image_embeddings = tf.contrib.layers.fully_connected(
          inputs=image_feature,
          num_outputs=self.emb_dim,  #turn image to word embdding space, same feature length
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)
    return image_embeddings

  def encode(self, image_feature):
    encoder_output = None
    state = None
    image_emb = self.build_image_embeddings(image_feature)
    return encoder_output, state, image_emb

class SimpleMemoryEncoder(ShowAndTellEncoder):
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)

  def encode(self, image_features):
    image_embs = self.build_image_embeddings(image_features)
    encoder_output = image_emb
    state = None
    image_emb = tf.reduce_mean(image_embs, 1)
    return encoder_output, state, image_emb


def batch_norm(x, is_training=False, name=''):  
  return tf.contrib.layers.batch_norm(inputs=x,
                                      decay=0.95,
                                      center=True,
                                      scale=True,
                                      is_training=is_training,
                                      updates_collections=None,
                                      fused=True,
                                      scope=(name + 'batch_norm'))

  
class MemoryEncoder(ShowAndTellEncoder):
  """
  """
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)

  def encode(self, image_features):
    image_emb = features2feature(image_features, is_training=is_training)
    
    image_features = tf.concat([image_features, tf.expand_dims(image_emb, 1)], 1)    
    image_embs = self.build_image_embeddings(image_features)
    image_emb = image_embs[:,-1]
    image_embs = image_embs[:,:-1]
    #to make it like rnn encoder outputs
    with tf.variable_scope("attention_embedding") as scope:
      encoder_output = tf.contrib.layers.fully_connected(
          inputs=image_embs,
          num_outputs=FLAGS.rnn_hidden_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)
    state = None
    #image_emb = tf.reduce_mean(image_embs, 1)

    return encoder_output, state, image_emb

class MemoryWithPosConcatEncoder(ShowAndTellEncoder):
  """
  """
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)
    self.pos_emb = embedding.get_embedding_cpu(name='pos_emb', height=FLAGS.image_attention_size)

  def encode(self, image_features):
    image_emb = features2feature(image_features, is_training=self.is_training)
 
    image_features = tf.concat([image_features, tf.expand_dims(image_emb, 1)], 1)    
    image_embs = self.build_image_embeddings(image_features)
    image_emb = image_embs[:,-1]
    image_embs = image_embs[:,:-1]
    # 128,64,512 64,512
    image_embs = tf.concat([image_embs, tf.tile(tf.expand_dims(self.pos_emb, 0), [melt.get_batch_size(image_embs), 1, 1])], 1)
    #to make it like rnn encoder outputs
    with tf.variable_scope("attention_embedding") as scope:
      encoder_output = tf.contrib.layers.fully_connected(
          inputs=image_embs,
          num_outputs=FLAGS.rnn_hidden_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)
    state = None
    #image_emb = tf.reduce_mean(image_embs, 1)

    return encoder_output, state, image_emb

class MemoryWithPosSumEncoder(ShowAndTellEncoder):
  """
  """
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)
    self.pos_emb = embedding.get_embedding_cpu(name='pos_emb', height=FLAGS.image_attention_size)

  def encode(self, image_features):
    image_emb = features2feature(image_features, is_training=self.is_training)
    
    image_features = tf.concat([image_features, tf.expand_dims(image_emb, 1)], 1)    
    image_embs = self.build_image_embeddings(image_features)
    image_emb = image_embs[:,-1]
    image_embs = image_embs[:,:-1]
    image_embs += self.pos_emb
    #to make it like rnn encoder outputs
    with tf.variable_scope("attention_embedding") as scope:
      encoder_output = tf.contrib.layers.fully_connected(
          inputs=image_embs,
          num_outputs=FLAGS.rnn_hidden_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)
    state = None
    #image_emb = tf.reduce_mean(image_embs, 1)

    return encoder_output, state, image_emb

class BaselineMemoryEncoder(ShowAndTellEncoder):
  """
  my first show attend and tell implementation, this is baseline
  it works pointing to correct part, and convergent faster 
  but final result is not better then no attention
  """
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)

  def encode(self, image_features):
    image_embs = self.build_image_embeddings(image_features)
    #to make it like rnn encoder outputs
    with tf.variable_scope("attention_embedding") as scope:
      encoder_output = tf.contrib.layers.fully_connected(
          inputs=image_embs,
          num_outputs=FLAGS.rnn_hidden_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)
    state = None
    image_emb = tf.reduce_mean(image_embs, 1)
    return encoder_output, state, image_emb

class ShowAttendAndTellEncoder(ImageEncoder):
  """
  strictly follow paper and copy from  @yunjey 's implementation of show-attend-and-tell
  https://github.com/yunjey/show-attend-and-tell
  this is like im2txt must add_text_start pad <S> before original text
  """
  def __init__(self, is_training=False, is_predict=False):
    ImageEncoder.__init__(self, is_training=False, is_predict=False)

  def get_initial_lstm(self, features):
    with tf.variable_scope('initial_lstm') as scope:
      features_mean = tf.reduce_mean(features, 1)
      with tf.variable_scope("h_embedding") as scope:
        h = tf.contrib.layers.fully_connected(
              inputs=features_mean,
              num_outputs=FLAGS.rnn_hidden_size,
              activation_fn=tf.nn.tanh,
              weights_initializer=self.initializer,
              scope=scope)
      with tf.variable_scope("c_embedding") as scope:
        c = tf.contrib.layers.fully_connected(
              inputs=features_mean,
              num_outputs=FLAGS.rnn_hidden_size,
              activation_fn=tf.nn.tanh,
              weights_initializer=self.initializer,
              scope=scope)
    return c, h

  def encode(self, image_features):
    image_emb = None
    encoder_output = image_features
    state = self.get_initial_lstm(image_features)
    return encoder_output, state, image_emb

class RnnEncoder(ShowAndTellEncoder):
  """
  using rnn to encode image features
  """
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)
    self.encoder = rnn_encoder.RnnEncoder(is_training, is_predict)

  def encode(self, image_features):
    image_emb = features2feature(image_features, is_training=self.is_training)
 
    image_features = tf.concat([image_features, tf.expand_dims(image_emb, 1)], 1)    
    image_embs = self.build_image_embeddings(image_features)
    image_emb = image_embs[:, -1]
    image_embs = image_embs[:, :-1]
    # #to make it like rnn encoder outputs
    # if hasattr(self, 'scene_feature'):
    #   print('!!!!!using scene_feature as pre words in image feature rnn encoding', file=sys.stderr)
    #   secene_embs = tf.nn.embedding_lookup(self.word_emb, self.scene_feature)
    #   image_embs = tf.concat([secene_embs, image_embs], 1)

    #print('----image_embs', image_embs)
    encoder_output, state = self.encoder.encode(image_embs, 
                                                embedding_lookup=False, 
                                                output_method=melt.rnn.OutputMethod.all)
    #print('----encoder_output', encoder_output, state, image_emb)

    return encoder_output, state, image_emb

class BiRnnEncoder(ShowAndTellEncoder):
  """
  using rnn to encode image features
  """
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)
    self.encoder = rnn_encoder.RnnEncoder(is_training, is_predict)
    with tf.variable_scope('second_rnn'):
      self.encoder2 = rnn_encoder.RnnEncoder(is_training, is_predict)

  def encode(self, image_features):
    image_emb = features2feature(image_features, is_training=self.is_training)
 
    image_features = tf.concat([image_features, tf.expand_dims(image_emb, 1)], 1)    
    image_embs = self.build_image_embeddings(image_features)
    image_emb = image_embs[:, -1]
    image_embs = image_embs[:, :-1]

    attention_size = FLAGS.image_attention_size
    attn_dim = int(math.sqrt(attention_size))
    image_embs2 = tf.reshape(image_embs, [-1, attn_dim, attn_dim, self.emb_dim])
    image_embs2  = tf.transpose(image_embs2, [0, 2, 1, 3])
    image_embs2  = tf.reshape(image_embs2, [-1, attention_size, self.emb_dim])
    image_embs2 = tf.reverse(image_embs2, axis=[1])

    #to make it like rnn encoder outputs

    #print('----image_embs', image_embs)
    encoder_output, state = self.encoder.encode(image_embs, 
                                                embedding_lookup=False, 
                                                output_method=melt.rnn.OutputMethod.all)

    with tf.variable_scope('second_rnn'):
      encoder_output2, _ = self.encoder2.encode(image_embs2, 
                                                embedding_lookup=False, 
                                                output_method=melt.rnn.OutputMethod.all)  

    encoder_output = tf.concat([encoder_output, encoder_output2], -1)

    #print('----encoder_output', encoder_output, state, image_emb)

    return encoder_output, state, image_emb

class Rnn2Encoder(ShowAndTellEncoder):
  """
  using rnn to encode image features
  """
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)
    self.encoder = rnn_encoder.RnnEncoder(is_training, is_predict)

  def encode(self, image_features):
    image_emb = features2feature(image_features, is_training=self.is_training)
 
    image_features = tf.concat([tf.expand_dims(image_emb, 1), image_features], 1)    
    image_embs = self.build_image_embeddings(image_features)
    #to make it like rnn encoder outputs

    #print('----image_embs', image_embs)
    encoder_output, state = self.encoder.encode(image_embs, 
                                                embedding_lookup=False, 
                                                output_method=melt.rnn.OutputMethod.all)
    #print('----encoder_output', encoder_output, state, image_emb)

    return encoder_output, state, None


class RnnPosEncoder(ShowAndTellEncoder):
  """
  using rnn to encode image features
  """
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)
    self.encoder = rnn_encoder.RnnEncoder(is_training, is_predict)
    self.pos_emb = embedding.get_embedding_cpu(name='pos_emb', height=FLAGS.image_attention_size)

  def encode(self, image_features):
    image_emb = features2feature(image_features, is_training=self.is_training)
 
    image_features = tf.concat([image_features, tf.expand_dims(image_emb, 1)], 1)   
    image_embs = self.build_image_embeddings(image_features)
    image_emb = image_embs[:,-1]
    image_embs = image_embs[:,:-1]
    image_embs = tf.concat([image_embs, 
                            tf.tile(tf.expand_dims(self.pos_emb, 0), 
                              [melt.get_batch_size(image_embs), 1, 1])], -1) 


    encoder_output, state = self.encoder.encode(image_embs, 
                                                embedding_lookup=False, 
                                                output_method=melt.rnn.OutputMethod.all)

    return encoder_output, state, image_emb

class RnnControllerEncoder(ShowAndTellEncoder):
  """
  using rnn/lstm controller for some steps to encode image features
  TODO
  """
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)

    #self.num_units = FLAGS.rnn_hidden_size
    self.num_units = self.emb_dim
    # self.cell = melt.create_rnn_cell(self.num_units, 
    #                                  #cell_type=FLAGS.cell, 
    #                                  cell_type='lstm_block',
    #                                  initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=1234))

    create_rnn_cell = functools.partial(
        melt.create_rnn_cell, 
        num_units=FLAGS.rnn_hidden_size,
        is_training=is_training, 
        keep_prob=FLAGS.keep_prob, 
        num_layers=FLAGS.num_layers, 
        cell_type=FLAGS.cell)
        #cell_type='lstm_block')
        #cell_type='lstm')

    # #follow models/textsum
    self.cell = create_rnn_cell(initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=1234))

  def encode(self, image_features):
    image_emb = features2feature(image_features, is_training=self.is_training)
 
    image_features = tf.concat([image_features, tf.expand_dims(image_emb, 1)], 1)   
    image_embs = self.build_image_embeddings(image_features)
    image_emb = image_embs[:, -1]
    image_embs = image_embs[:, :-1]

    batch_size = melt.get_batch_size(image_emb)
    num_steps = 4
    create_attention_mechanism = melt.seq2seq.LuongAttention
    memory = image_embs

    #need a scope !
    with tf.variable_scope('rnn_controller_attention'):
      attention_mechanism = create_attention_mechanism(
          num_units=self.num_units,
          memory=memory,
          memory_sequence_length=None)

      cell = melt.seq2seq.AttentionWrapper(
              self.cell,
              attention_mechanism,
              attention_layer_size=None,
              initial_cell_state=None, 
              alignment_history=False,
              no_context=False)

    initial_state = cell.zero_state(batch_size, tf.float32)
    encoder_inputs = [image_emb] * num_steps
    #encoder_inputs = [tf.zeros([batch_size, 1])] * num_steps
    encoder_inputs = tf.stack(encoder_inputs, 1)

    encoder_inputs = tf.reshape(encoder_inputs, [batch_size, num_steps, self.num_units])

    length = tf.ones([batch_size], dtype=tf.int32) * num_steps
    helper = melt.seq2seq.LoopHelper(encoder_inputs, length)

    my_decoder = melt.seq2seq.BasicTrainingDecoder(
             cell=cell,
             helper=helper,
             initial_state=initial_state)
    encoder_output, state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, scope='rnn_controller_loop')
    state = state.cell_state

    #TODO FIXME
    #ValueError: Trying to share variable show_and_tell/main/static_loop_encoder/basic_lstm_cell/kernel, but specified shape (1024, 2048) and found shape (513, 2048).
    #encoder_output, state = static_loop_encoder.encode(encoder_input, initial_state, cell, 
    #                                                   num_steps=num_steps)

    #state = state.cell_state
    
    #print('attention for decoder:', encoder_output)

    return encoder_output, state, image_emb


class RnnController2Encoder(ShowAndTellEncoder):
  """
  using rnn/lstm controller for some steps to encode image features
  TODO
  """
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)

    #self.num_units = FLAGS.rnn_hidden_size
    self.num_units = self.emb_dim

    create_rnn_cell = functools.partial(
        melt.create_rnn_cell, 
        num_units=FLAGS.rnn_hidden_size,
        is_training=is_training, 
        keep_prob=FLAGS.keep_prob, 
        num_layers=FLAGS.num_layers, 
        cell_type=FLAGS.cell)
        #cell_type='lstm_block')
        #cell_type='lstm')

    # #follow models/textsum
    self.cell = create_rnn_cell(initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=1234))
    self.encoder = rnn_encoder.RnnEncoder(is_training, is_predict)

  def encode(self, image_features):
    image_emb = features2feature(image_features, is_training=self.is_training)
 
    image_features = tf.concat([image_features, tf.expand_dims(image_emb, 1)], 1)   
    image_embs = self.build_image_embeddings(image_features)
    image_emb = image_embs[:, -1]
    image_embs = image_embs[:, :-1]

    encoder_output, state = self.encoder.encode(image_embs, 
                                              embedding_lookup=False, 
                                              output_method=melt.rnn.OutputMethod.all)

    batch_size = melt.get_batch_size(image_emb)
    num_steps = 8
    create_attention_mechanism = melt.seq2seq.LuongAttention
    memory = encoder_output

    #need a scope !
    with tf.variable_scope('rnn_controller_attention'):
      attention_mechanism = create_attention_mechanism(
          num_units=self.num_units,
          memory=memory,
          memory_sequence_length=None)

      cell = melt.seq2seq.AttentionWrapper(
              self.cell,
              attention_mechanism,
              attention_layer_size=None,
              initial_cell_state=state, 
              alignment_history=False,
              no_context=False)

    initial_state = cell.zero_state(batch_size, tf.float32)
    #encoder_inputs = [image_emb] * num_steps
    encoder_inputs = [tf.zeros([batch_size, self.num_units])] * num_steps
    encoder_inputs = tf.stack(encoder_inputs, 1)

    encoder_inputs = tf.reshape(encoder_inputs, [batch_size, num_steps, self.num_units])

    length = tf.ones([batch_size], dtype=tf.int32) * num_steps
    helper = melt.seq2seq.LoopHelper(encoder_inputs, length)

    my_decoder = melt.seq2seq.BasicTrainingDecoder(
             cell=cell,
             helper=helper,
             initial_state=initial_state)
    encoder_output, state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, scope='rnn_controller_loop')
    state = state.cell_state

    #TODO maybe concat rnn encoder_output and rnn controller encoder output ?
    return encoder_output, state, image_emb

class RnnController3Encoder(ShowAndTellEncoder):
  """
  using rnn/lstm controller for some steps to encode image features
  TODO
  """
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)

    #self.num_units = FLAGS.rnn_hidden_size
    self.num_units = self.emb_dim

    create_rnn_cell = functools.partial(
        melt.create_rnn_cell, 
        num_units=FLAGS.rnn_hidden_size,
        is_training=is_training, 
        keep_prob=FLAGS.keep_prob, 
        num_layers=FLAGS.num_layers, 
        cell_type=FLAGS.cell)
        #cell_type='lstm_block')
        #cell_type='lstm')

    # #follow models/textsum
    self.cell = create_rnn_cell(initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=1234))
    self.encoder = rnn_encoder.RnnEncoder(is_training, is_predict)

  def encode(self, image_features):
    image_emb = features2feature(image_features, is_training=self.is_training)
 
    image_features = tf.concat([image_features, tf.expand_dims(image_emb, 1)], 1)   
    image_embs = self.build_image_embeddings(image_features)
    image_emb = image_embs[:, -1]
    image_embs = image_embs[:, :-1]

    rnn_output, state = self.encoder.encode(image_embs, 
                                              embedding_lookup=False, 
                                              output_method=melt.rnn.OutputMethod.all)

    batch_size = melt.get_batch_size(image_emb)
    num_steps = 8
    create_attention_mechanism = melt.seq2seq.LuongAttention
    memory = rnn_output

    #need a scope !
    with tf.variable_scope('rnn_controller_attention'):
      attention_mechanism = create_attention_mechanism(
          num_units=self.num_units,
          memory=memory,
          memory_sequence_length=None)

      cell = melt.seq2seq.AttentionWrapper(
              self.cell,
              attention_mechanism,
              attention_layer_size=None,
              initial_cell_state=state, 
              alignment_history=False,
              no_context=False)

    initial_state = cell.zero_state(batch_size, tf.float32)
    #encoder_inputs = [image_emb] * num_steps
    encoder_inputs = [tf.zeros([batch_size, self.num_units])] * num_steps
    encoder_inputs = tf.stack(encoder_inputs, 1)

    encoder_inputs = tf.reshape(encoder_inputs, [batch_size, num_steps, self.num_units])

    length = tf.ones([batch_size], dtype=tf.int32) * num_steps
    helper = melt.seq2seq.LoopHelper(encoder_inputs, length)

    my_decoder = melt.seq2seq.BasicTrainingDecoder(
             cell=cell,
             helper=helper,
             initial_state=initial_state)
    encoder_output, state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, scope='rnn_controller_loop')
    state = state.cell_state

    encoder_output = tf.concat([encoder_output, rnn_output], 1)

    return encoder_output, state, image_emb

class RnnController4Encoder(ShowAndTellEncoder):
  """
  using rnn/lstm controller for some steps to encode image features
  TODO
  """
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)

    #self.num_units = FLAGS.rnn_hidden_size
    self.num_units = self.emb_dim

    create_rnn_cell = functools.partial(
        melt.create_rnn_cell, 
        num_units=FLAGS.rnn_hidden_size,
        is_training=is_training, 
        keep_prob=FLAGS.keep_prob, 
        num_layers=FLAGS.num_layers, 
        cell_type=FLAGS.cell)
        #cell_type='lstm_block')
        #cell_type='lstm')

    # #follow models/textsum
    self.cell = create_rnn_cell(initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=1234))
    self.encoder = rnn_encoder.RnnEncoder(is_training, is_predict)

  def encode(self, image_features):
    image_emb = features2feature(image_features, is_training=self.is_training)
 
    image_features = tf.concat([image_features, tf.expand_dims(image_emb, 1)], 1)   
    image_embs = self.build_image_embeddings(image_features)
    image_emb = image_embs[:, -1]
    image_embs = image_embs[:, :-1]

    encoder_output, state = self.encoder.encode(image_embs, 
                                              embedding_lookup=False, 
                                              output_method=melt.rnn.OutputMethod.all)

    batch_size = melt.get_batch_size(image_emb)
    num_steps = 8
    create_attention_mechanism = melt.seq2seq.LuongAttention
    memory = encoder_output

    #need a scope !
    with tf.variable_scope('rnn_controller_attention'):
      attention_mechanism = create_attention_mechanism(
          num_units=self.num_units,
          memory=memory,
          memory_sequence_length=None)

      cell = melt.seq2seq.AttentionWrapper(
              self.cell,
              attention_mechanism,
              attention_layer_size=None,
              initial_cell_state=state, 
              alignment_history=False,
              no_context=False)

    initial_state = cell.zero_state(batch_size, tf.float32)
    encoder_inputs = [image_emb] * num_steps
    #encoder_inputs = [tf.zeros([batch_size, self.num_units])] * num_steps
    encoder_inputs = tf.stack(encoder_inputs, 1)

    encoder_inputs = tf.reshape(encoder_inputs, [batch_size, num_steps, self.num_units])

    length = tf.ones([batch_size], dtype=tf.int32) * num_steps
    helper = melt.seq2seq.LoopHelper(encoder_inputs, length)

    my_decoder = melt.seq2seq.BasicTrainingDecoder(
             cell=cell,
             helper=helper,
             initial_state=initial_state)
    encoder_output, state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, scope='rnn_controller_loop')
    state = state.cell_state

    #TODO maybe concat rnn encoder_output and rnn controller encoder output ?
    return encoder_output, state, image_emb

class RnnController5Encoder(ShowAndTellEncoder):
  """
  using rnn/lstm controller for some steps to encode image features
  TODO
  """
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)

    #self.num_units = FLAGS.rnn_hidden_size
    self.num_units = self.emb_dim

    create_rnn_cell = functools.partial(
        melt.create_rnn_cell, 
        num_units=FLAGS.rnn_hidden_size,
        is_training=is_training, 
        keep_prob=FLAGS.keep_prob, 
        num_layers=FLAGS.num_layers, 
        cell_type=FLAGS.cell)
        #cell_type='lstm_block')
        #cell_type='lstm')

    # #follow models/textsum
    self.cell = create_rnn_cell(initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=1234))
    self.encoder = rnn_encoder.RnnEncoder(is_training, is_predict)

  def encode(self, image_features):
    image_emb = features2feature(image_features, is_training=self.is_training)
 
    image_features = tf.concat([image_features, tf.expand_dims(image_emb, 1)], 1)   
    image_embs = self.build_image_embeddings(image_features)
    image_emb = image_embs[:, -1]
    image_embs = image_embs[:, :-1]

    rnn_output, state = self.encoder.encode(image_embs, 
                                              embedding_lookup=False, 
                                              output_method=melt.rnn.OutputMethod.all)

    batch_size = melt.get_batch_size(image_emb)
    num_steps = 8
    create_attention_mechanism = melt.seq2seq.LuongAttention
    memory = rnn_output

    #need a scope !
    with tf.variable_scope('rnn_controller_attention'):
      attention_mechanism = create_attention_mechanism(
          num_units=self.num_units,
          memory=memory,
          memory_sequence_length=None)

      cell = melt.seq2seq.AttentionWrapper(
              self.cell,
              attention_mechanism,
              attention_layer_size=None,
              initial_cell_state=state, 
              alignment_history=False,
              no_context=False)

    initial_state = cell.zero_state(batch_size, tf.float32)
    encoder_inputs = [image_emb] * num_steps
    #encoder_inputs = [tf.zeros([batch_size, self.num_units])] * num_steps
    encoder_inputs = tf.stack(encoder_inputs, 1)

    encoder_inputs = tf.reshape(encoder_inputs, [batch_size, num_steps, self.num_units])

    length = tf.ones([batch_size], dtype=tf.int32) * num_steps
    helper = melt.seq2seq.LoopHelper(encoder_inputs, length)

    my_decoder = melt.seq2seq.BasicTrainingDecoder(
             cell=cell,
             helper=helper,
             initial_state=initial_state)
    encoder_output, state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, scope='rnn_controller_loop')
    state = state.cell_state

    encoder_output = tf.concat([encoder_output, rnn_output], 1)

    return encoder_output, state, image_emb

Encoders = {
  'ShowAndTell': ShowAndTellEncoder,
  'SimpleMemory': SimpleMemoryEncoder,
  'BaselineMemory': BaselineMemoryEncoder,
  'Memory': MemoryEncoder,
  'MemoryWithPosSum': MemoryWithPosSumEncoder,
  'MemoryWithPosConcat': MemoryWithPosConcatEncoder,
  'ShowAttendAndTell': ShowAttendAndTellEncoder,
  'Rnn': RnnEncoder,
  'BiRnn': BiRnnEncoder,
  'Rnn2': Rnn2Encoder,
  'RnnPos': RnnPosEncoder,
  'RnnController': RnnControllerEncoder,
  'RnnController2': RnnController2Encoder,
  'RnnController3': RnnController3Encoder,
  'RnnController4': RnnController4Encoder,
  'RnnController5': RnnController5Encoder,
}
