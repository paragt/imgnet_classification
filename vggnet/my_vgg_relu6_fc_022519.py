# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
import tensorflow as tf


fcorr_method_ids = {'AND' : 1, 'XOR' : 2, 'LINEAR' : 3}



def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with arg_scope(
      [layers.conv2d, layers_lib.fully_connected],
      activation_fn=nn_ops.relu6,
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      biases_initializer=init_ops.zeros_initializer()):
    with arg_scope([layers.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def vgg_a(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='vgg_a'):
  """Oxford Net VGG 11-Layers version A Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with variable_scope.variable_scope(scope, 'vgg_a', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with arg_scope(
        [layers.conv2d, layers_lib.max_pool2d],
        outputs_collections=end_points_collection):
      net = layers_lib.repeat(
          inputs, 1, layers.conv2d, 64, [3, 3], scope='conv1')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
      net = layers_lib.repeat(net, 1, layers.conv2d, 128, [3, 3], scope='conv2')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
      net = layers_lib.repeat(net, 2, layers.conv2d, 256, [3, 3], scope='conv3')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
      net = layers_lib.repeat(net, 2, layers.conv2d, 512, [3, 3], scope='conv4')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
      net = layers_lib.repeat(net, 2, layers.conv2d, 512, [3, 3], scope='conv5')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      net = layers.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
      net = layers_lib.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout6')
      net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
      net = layers_lib.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout7')
      net = layers.conv2d(
          net,
          num_classes, [1, 1],
          activation_fn=None,
          normalizer_fn=None,
          scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points


vgg_a.default_image_size = 224


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with variable_scope.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with arg_scope(
        [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
        outputs_collections=end_points_collection):
      net = layers_lib.repeat(
          inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
      net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
      net = layers_lib.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
      net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
      net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      net = layers.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
      net = layers_lib.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout6')
      net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
      net = layers_lib.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout7')
      net = layers.conv2d(
          net,
          num_classes, [1, 1],
          activation_fn=None,
          normalizer_fn=None,
          scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points


vgg_16.default_image_size = 224

def repeat_conv(inp, nrepeat, noutputs, filtsz, scope):

    # inp dims: [1, fx, fy, n_input_channels, n_output_channels]

    res=inp
    with tf.variable_scope(scope):
        for i in range(nrepeat):
            res = layers.conv2d(res, noutputs, filtsz, scope=scope+'_'+str(i+1))
    return res 


def ANDop(res):


def init_conv_corr(inp, nrepeat, noutputs, filtsz, scope, forder_in, corr_feat):

    # inp dims: [1, fx, fy, n_input_channels, n_output_channels]
    forder = forder_in
    with tf.variable_scope(scope):
        res = layers.conv2d(inp, noutputs, filtsz, scope=scope+'_1')

        ninp_ch = tf.shape(res)[3]
        res2 = tf.gather(res, forder, axis=3) #res[:,:,:,rndidx,:]

        # use AND or OR, not use 6-res2
        #fres = tf.math.multiply(res, 6-res2) + tf.math.multiply(6-res, res2) # xor
        fres = tf.math.multiply(res, res2) #+ tf.math.multiply(6-res, 6-res2) # xnor
        fres_clipped = tf.clip_by_value(fres, 0, 6)

        res0 = layers.conv2d(res, noutputs, filtsz, scope=scope+'_2')
        res1 = layers.conv2d(fres_clipped, noutputs, filtsz, scope=scope+'_2_corr')
             
        combined = tf.concat([res0, res1], axis=3)
        vname = 'vgg_16/'+scope+'/'+scope+'_2'
        forder = corr_feat[vname]

    return combined, forder

def repeat_conv_corr(inp, nrepeat, noutputs, filtsz, scope, forder_in, corr_feat):

    # inp dims: [1, fx, fy, n_input_channels, n_output_channels]

    res=inp
    forder = forder_in
    with tf.variable_scope(scope):
        for i in range(nrepeat):
            ninp_ch = tf.shape(res)[3]
            ninp_ch_half = tf.cast((ninp_ch/2), dtype=tf.int64)

            inp0 = tf.gather(res, tf.range(0,ninp_ch_half), axis=3) #res[:,:,:,:(ninp_ch/2),:]
            inp1 = tf.gather(res, tf.range(ninp_ch_half,ninp_ch), axis=3) #res[:,:,:,(ninp_ch/2):,:]

            ninp_ch0 = tf.shape(inp0)[3]
            inp0_rnd = tf.gather(inp0, forder , axis=3) #inp0[:,:,:,rndidx,:]

            # use AND or OR, not use 6-inp0  
            #fres = tf.math.multiply(inp0, 6-inp0_rnd) + tf.math.multiply(6-inp0, inp0_rnd) # xor
            fres = tf.math.multiply(inp0, inp0_rnd) #+ tf.math.multiply(6-inp0, 6-inp0_rnd) # xnor
            fres_clipped = tf.clip_by_value(fres, 0, 6)
            fres_combined = tf.concat([inp1, fres_clipped], axis=3)

            # required, layers.conv2d wont work otherwise 
            inp0.set_shape([None,res.shape[1],res.shape[2], res.shape[3].__floordiv__(2)] )
            fres_combined.set_shape([None,res.shape[1],res.shape[2], inp0.shape[3].__mul__(2)] )

            res0 = layers.conv2d(inp0, noutputs, filtsz, scope=scope+'_'+str(i+1))
            res1 = layers.conv2d(fres_combined, noutputs, filtsz, scope=scope+'_'+str(i+1)+'_corr')
             
            res = tf.concat([res0, res1], axis=3)

            vname = 'vgg_16/'+scope+'/'+scope+'_'+str(i+1)
            forder = corr_feat[vname]

    return res, forder

def vgg_16_tcorr(inputs,
           corr_features,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16_tcorr'):

  with variable_scope.variable_scope(scope, 'vgg_16_tcorr', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with arg_scope(
        [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
        outputs_collections=end_points_collection):
      forder = corr_features['vgg_16/conv1/conv1_1']
      net, forder = init_conv_corr(inputs, 2, 64, [3, 3], 'conv1', forder, corr_features)
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')

      net, forder = repeat_conv_corr(net, 2, 128, [3, 3], 'conv2', forder, corr_features)
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')

      net, forder = repeat_conv_corr(net, 3, 256, [3, 3], 'conv3', forder, corr_features)
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')

      net, forder = repeat_conv_corr(net, 3, 512, [3, 3], 'conv4', forder, corr_features)
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')

      net, _ = repeat_conv_corr(net, 3, 512, [3, 3], 'conv5', forder, corr_features)
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')

      net = layers.conv2d(net, 512, [1,1], scope = 'fuse5')

      # Use conv2d instead of fully_connected layers.
      net = layers.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
      net = layers_lib.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout6')
      net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
      net = layers_lib.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout7')
      net = layers.conv2d(
          net,
          num_classes, [1, 1],
          activation_fn=None,
          normalizer_fn=None,
          scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points


vgg_16_tcorr.default_image_size = 224



def init_conv_comb(inp, nrepeat, noutputs, filtsz, scope):

    # inp dims: [1, fx, fy, n_input_channels, n_output_channels]
    with tf.variable_scope(scope):
        res = layers.conv2d(inp, noutputs, filtsz, scope=scope+'_1')
       
        '''
        ninp_ch = tf.shape(res)[3]
        chidx = tf.range(0,ninp_ch)
        chidx_shift = tf.mod(chidx+1,ninp_ch)  
        res2 = tf.gather(res, chidx , axis=3) #res[:,:,:,rndidx,:]
        #res2 = tf.gather(res, chidx_shift , axis=3) #res[:,:,:,rndidx,:]

        fres = tf.math.multiply(res, 6-res2) + tf.math.multiply(6-res, res2) # xor
        #fres = tf.math.multiply(res, res2) + tf.math.multiply(6-res, 6-res2) # xnor
        fres_clipped = tf.clip_by_value(fres, 0, 6)
        #fres_combined = tf.concat([res, fres_clipped], axis=3)
        fres_combined = tf.concat([res, res2], axis=3)

        fres_combined.set_shape([None,res.shape[1],res.shape[2], res.shape[3].__mul__(2)] )
        #res = layers.conv2d(fres_combined, noutputs, filtsz, scope=scope+'_2_comb')
        ''' 

        res = layers.conv2d(res, noutputs, filtsz, scope=scope+'_2_comb')


    return res 

def repeat_conv_comb(inp, nrepeat, ninputs, noutputs, filtsz, scope):

    # inp dims: [1, fx, fy, n_input_channels, n_output_channels]

    res=inp
    ninput_channels=ninputs
    with tf.variable_scope(scope):
        for i in range(nrepeat):
            ninp_ch = tf.shape(res)[3]
            
            chidx = tf.range(0,ninp_ch)
            chidx_shift = tf.mod(chidx+1, ninp_ch) 
            inp_rnd = tf.gather(res, chidx_shift, axis=3) 
            
            fres = tf.math.multiply(res, inp_rnd) #+ tf.math.multiply(6-res, inp_rnd) # xor
            fres_clipped = tf.clip_by_value(fres, 0, 6)
            fres_combined = tf.concat([res, fres_clipped], axis=3)

            #extra_feat = layers.conv2d(res, ninput_channels, [1, 1], scope=scope+'_'+str(i+1)+'_rel')
            #fres_combined = tf.concat([res, extra_feat], axis=3)

            # required, layers.conv2d wont work otherwise 
            fres_combined.set_shape([None,res.shape[1],res.shape[2], res.shape[3].__mul__(2)] )

            res = layers.conv2d(fres_combined, noutputs, filtsz, scope=scope+'_'+str(i+1)+'_comb')
            ninput_channels = noutputs

    return res 




def vgg_16_tcomb(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16_tcomb'):

  with variable_scope.variable_scope(scope, 'vgg_16_tcomb', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with arg_scope(
        [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
        outputs_collections=end_points_collection):
      net = init_conv_comb(inputs, 2, 64, [3, 3], 'conv1')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')

      net = repeat_conv_comb(net, 2, 64, 128, [3, 3], 'conv2')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')

      net = repeat_conv_comb(net, 3, 128, 256, [3, 3], 'conv3')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')

      net = repeat_conv_comb(net, 3, 256, 512, [3, 3], 'conv4')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')

      net = repeat_conv_comb(net, 3, 512, 512, [3, 3], 'conv5')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = layers.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
      net = layers_lib.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout6')
      net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
      net = layers_lib.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout7')
      net = layers.conv2d(
          net,
          num_classes, [1, 1],
          activation_fn=None,
          normalizer_fn=None,
          scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points


vgg_16_tcomb.default_image_size = 224






def vgg_19(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19'):
  """Oxford Net VGG 19-Layers version E Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with variable_scope.variable_scope(scope, 'vgg_19', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with arg_scope(
        [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
        outputs_collections=end_points_collection):
      net = layers_lib.repeat(
          inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
      net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
      net = layers_lib.repeat(net, 4, layers.conv2d, 256, [3, 3], scope='conv3')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
      net = layers_lib.repeat(net, 4, layers.conv2d, 512, [3, 3], scope='conv4')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
      net = layers_lib.repeat(net, 4, layers.conv2d, 512, [3, 3], scope='conv5')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      net = layers.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
      net = layers_lib.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout6')
      net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
      net = layers_lib.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout7')
      net = layers.conv2d(
          net,
          num_classes, [1, 1],
          activation_fn=None,
          normalizer_fn=None,
          scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points


vgg_19.default_image_size = 224

# Alias
vgg_d = vgg_16
vgg_e = vgg_19
