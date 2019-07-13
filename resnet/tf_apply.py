
import argparse
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import pdb
from dataloader import mydataloader_train, mydataloader_val
import numpy as np

from tensorflow.python.tools import inspect_checkpoint as chkp
import resnet_model
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='coco-animals/train')
parser.add_argument('--val_dir', default='coco-animals/val')
parser.add_argument('--model_path', default='vgg_16.ckpt', type=str)
parser.add_argument('--labelmap',  type=str)
parser.add_argument('--batch_size', default=25, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=3, type=int)
parser.add_argument('--num_epochs2', default=3, type=int)
parser.add_argument('--learning_rate1', default=1e-3, type=float)
parser.add_argument('--learning_rate2', default=1e-5, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)

def check_accuracy(sess, correct_prediction, dataset_init_op, prediction, logits):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    while True:
        try:
            correct_pred = sess.run(correct_prediction)
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc

def _get_block_sizes(resnet_size):
    choices = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }

    try:
        return choices[resnet_size]
    except KeyError:
        err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
        raise ValueError(err)


class ImagenetModel(resnet_model.Model):
    """Model class with appropriate defaults for Imagenet data."""

    def __init__(self, resnet_size,num_classes, resnet_version=1, data_format='channels_last', dtype=resnet_model.DEFAULT_DTYPE):
        """These are the parameters that work for Imagenet data.

        Args:
            resnet_size: The number of convolutional layers needed in the model.
            data_format: Either 'channels_first' or 'channels_last', specifying which data format to use when setting up the model.
            num_classes: The number of output classes needed from the model. This enables users to extend the same model to their own datasets.
            resnet_version: Integer representing which version of the ResNet network to use. See README for details. Valid values: [1, 2]
            dtype: The TensorFlow dtype to use for calculations.
        """

        # For bigger models, we want to use "bottleneck" layers
        if resnet_size < 50:
            bottleneck = False
        else:
            bottleneck = True

        super(ImagenetModel, self).__init__(
            resnet_size=resnet_size,
            bottleneck=bottleneck,
            num_classes=num_classes,
            num_filters=64,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            block_sizes=_get_block_sizes(resnet_size),
            block_strides=[1, 2, 2, 2],
            resnet_version=resnet_version,
            data_format=data_format,
            dtype=dtype)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def main(args):

    im0 = Image.open('/trn_dir/imgnet/data/trn/whippet/n02091134_12142.JPEG')
    im0=im0.resize((224,224))
    image_np = load_image_into_numpy_array(im0)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    #modelDir = '/trn_dir/models/model1'
    #builder =  tf.saved_model.builder.SavedModelBuilder(modelDir)
    # --------------------------------------------------------------------------
    # In TensorFlow, you first want to define the computation graph with all the
    # necessary operations: loss, training op, accuracy...
    # Any tensor created in the `graph.as_default()` scope will be part of `graph`
    graph = tf.Graph()
    with graph.as_default():
        
        #load data and create an iterator.

        #dataldr_trn=mydataloader_train(args.train_dir)
        dataldr_val=mydataloader_val(args.val_dir, args.labelmap)

        iterator = tf.data.Iterator.from_structure(dataldr_val.batched_val_dataset.output_types,dataldr_val.batched_val_dataset.output_shapes)
        images, labels = iterator.get_next()

        #train_init_op = iterator.make_initializer(dataldr_trn.batched_train_dataset)
        val_init_op = iterator.make_initializer(dataldr_val.batched_val_dataset)

        # Indicates whether we are in training or in test mode
        #is_training = tf.placeholder(tf.bool)

        #vgg = tf.contrib.slim.nets.vgg
        #with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
            #logits, endpoints_dict = vgg.vgg_16(images, num_classes=dataldr_val.num_classes, is_training=is_training,dropout_keep_prob=args.dropout_keep_prob)

        #with slim.arg_scope(vgg_relu6.vgg_arg_scope(weight_decay=args.weight_decay)):
            #logits, endpoints_dict = vgg_relu6.vgg_16(images, num_classes=dataldr_val.num_classes, is_training=is_training,dropout_keep_prob=args.dropout_keep_prob)


        model = ImagenetModel(resnet_size=50,num_classes=dataldr_val.num_classes+1,data_format='channels_last', resnet_version=1)

        logits = model(images, training = False)
        logits = tf.cast(logits, tf.float32)


        #tf.add_to_collection("logits",logits)


        # Specify where the model checkpoint is (pretrained weights).
        model_path = args.model_path
        #assert(os.path.isfile(model_path))

 
        # Restore only the layers up to fc7 (included)
        # Calling function `init_fn(sess)` will load all the pretrained weights.
        variables_to_restore = tf.contrib.framework.get_variables_to_restore()
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

        # Initialization operation from scratch for the new "fc8" layers
        # `get_variables` will only return the variables whose name starts with the given pattern
        #fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
        #fc8_init = tf.variables_initializer(var_list=fc8_variables)

        # ---------------------------------------------------------------------
        # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
        # We can then call the total loss easily
        #tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        #loss = tf.losses.get_total_loss()

        # First we want to train only the reinitialized last layer fc8 for a few epochs.
        # We run minimize the loss only with respect to the fc8 variables (weight and bias).
        
        #step = tf.Variable(0, trainable=False)
        #boundaries = [100000, 150000]
        #values = [0.001, 0.0009, 0.0008]
        #piecewise_lr = tf.train.piecewise_constant(global_step, boundaries, values)
        #learning_rate = tf.train.exponential_decay(0.001, step, 80000, 0.95, staircase=True)
        
        #full_optimizer = tf.train.RMSPropOptimizer(args.learning_rate2, momentum=0.9,epsilon=0.01)
        #full_optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.01) #beta1=0.9, beta2=0.99
        #full_train_op = full_optimizer.minimize(loss, global_step=step)

        #fc8_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate1)
        #fc8_train_op = fc8_optimizer.minimize(loss, var_list=fc8_variables)

        # Then we want to finetune the entire model for a few epochs.
        # We run minimize the loss only with respect to all the variables.
        #full_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate2)
        #full_train_op = full_optimizer.minimize(loss)

        # Evaluation metrics
        prediction = tf.to_int32(tf.argmax(logits, 1))-1
        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        global_init=tf.global_variables_initializer() # initializes all variables in the graph
        #gs_init = tf.variables_initializer(var_list=[step])
        #opt_var_init=tf.variables_initializer(full_optimizer.variables())
        #var_list=[var for var in tf.global_variables() ] # debug pruposes


        #modelSaver = tf.train.Saver()

        #tf.get_default_graph().finalize()




    # --------------------------------------------------------------------------
    # Now that we have built the graph and finalized it, we define the session.
    # The session is the interface to *run* the computational graph.
    # We can call our training operations with `sess.run(train_op)` for instance
    with tf.Session(graph=graph) as sess:
        pdb.set_trace()
        #chkp.print_tensors_in_checkpoint_file(model_path, tensor_name='',all_tensors=False, all_tensor_names=True)
        sess.run(global_init) 
        init_fn(sess)  # load the pretrained weights
        sess.run(val_init_op)
        
        #sess.run(gs_init)
        #sess.run(fc8_init)  # initialize the new fc8 layer
        #sess.run(opt_var_init)        
        #print('global_step: %s' % tf.train.get_global_step(graph)) 

        #sess.run(var_list) # look into the variable list in debugger to have an idea

        pred = sess.run(prediction)

        val_acc = check_accuracy(sess, correct_prediction, val_init_op, prediction, logits)
        print('Val accuracy: %f\n' % val_acc)

      


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
