
import argparse
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import pdb
#from dataloader import mydataloader_train, mydataloader_val
import numpy as np
import preprocess
from tensorflow.python.tools import inspect_checkpoint as chkp
import resnet_model
from PIL import Image
import json

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='coco-animals/train')
parser.add_argument('--val_dir', default='coco-animals/val')
parser.add_argument('--label_map', default='/trn_dir/imgnet/data/labelmap.json')
parser.add_argument('--model_dir', default='vgg_16.ckpt', type=str)
parser.add_argument('--batch_size', default=25, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs2', default=3, type=int)
parser.add_argument('--epoch_size', default=0, type=int)
parser.add_argument('--num_gpus', default=1, type=int)
parser.add_argument('--learning_rate1', default=1e-3, type=float)
parser.add_argument('--learning_rate2', default=1e-5, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float) # google  uses 4e-5, was using 4e-5 before


def list_images(directory,labelmap_file):
    """
    Get all the images and labels in directory/label/*.jpg
    """

    #pdb.set_trace()
    fid= open(labelmap_file)
    lbldata = json.load(fid)
    fid.close()

    label_map = lbldata[0]

    alldirs = sorted(os.listdir(directory))

    filenames = []
    labels = []
    files_and_labels = []
    for label in alldirs:
        for f in os.listdir(os.path.join(directory, label)):
            labelid = label_map[label]
            filenames.append(os.path.join(directory, label, f))
            labels.append(labelid)

    return filenames, labels

def check_accuracy(sess,correct_prediction, image_names, labels, train_mode, dirname, labelmap_file, prediction, logits):

    filenames, filelbls = list_images(dirname, labelmap_file)
    total_examples = len(filenames)
    num_correct, num_samples = 0, 0
    #pdb.set_trace()

    bsz = args.batch_size
    nbatches = int(total_examples/bsz)
    for nb in range(nbatches):
        batch_names = filenames[nb*bsz:(nb+1)*bsz]
        batch_labels = filelbls[nb*bsz:(nb+1)*bsz]

        correct_pred = sess.run(correct_prediction, feed_dict={image_names: batch_names, labels: batch_labels, train_mode: False})
        num_correct += correct_pred.sum()
        num_samples += correct_pred.shape[0]
        #if num_samples % 2000 == 0: print('detected ',num_samples)  

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


def get_input_image(imagename, train_mode):
    pp = preprocess.preprocessor()
    image = pp._parse_function(imagename)
    cimage = tf.cond(train_mode, lambda: pp.training_preprocess(image),lambda: pp.val_preprocess(image))
    return cimage


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def main(args):

    #pdb.set_trace()
    train_filenames, train_labels = list_images(args.train_dir, args.label_map)
    total_examples = len(train_labels) # imgnet 1,281,167
    if args.epoch_size<1:
        exm_per_epoch = total_examples
    else:
        exm_per_epoch = args.epoch_size #int(total_examples/2)
    print('epoch size = ', exm_per_epoch)
    print('weight decay = ', args.weight_decay)
    num_classes = len(set(train_labels))
    modelDir = args.model_dir

    graph = tf.Graph()
    with graph.as_default():
        
        image_names = tf.placeholder(dtype=tf.string, name='image_names')
        labels = tf.placeholder(dtype=tf.int32, name='labels')
        train_mode = tf.placeholder(dtype=tf.bool,name = 'train_mode')

        preprocess_fn = lambda x: get_input_image(x, train_mode)
        images = tf.map_fn(preprocess_fn, image_names,dtype=tf.float32)


        model = ImagenetModel(resnet_size=50,num_classes=num_classes,data_format='channels_last', resnet_version=1)

        logits = model(images, training = train_mode)
        logits = tf.cast(logits, tf.float32)


        tf.add_to_collection("logits",logits)


        prediction = tf.to_int32(tf.argmax(logits, 1))
        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        ce_loss=tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss_filter_fn=None
        def exclude_batch_norm(name):
            return 'batch_normalization' not in name
        loss_filter_fn = exclude_batch_norm

        # Add weight decay to the loss.
        l2_loss = args.weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables() if exclude_batch_norm(v.name)])

        loss = ce_loss + l2_loss
        


        # ---------------------------------------------------------------------
        step=tf.train.get_or_create_global_step()
        nu_epoch = tf.cast(exm_per_epoch*1.0/(args.batch_size), tf.int32) # num_update_per_epoch

        init_lr = 0.005

        # piecewise constant  
        #boundaries = [10*nu_epoch, 20*nu_epoch, 30*nu_epoch, 40*nu_epoch, 50*nu_epoch, 55*nu_epoch, 60*nu_epoch, 65*nu_epoch, 70*nu_epoch, 75*nu_epoch]
        #values=[init_lr, init_lr*0.85, init_lr*0.65, init_lr*0.35, init_lr*0.15, init_lr*0.09, init_lr*(0.08), init_lr*(0.07), init_lr*(0.06), init_lr*(0.05), init_lr*(0.04)]
        ##values=[init_lr, init_lr*0.85, init_lr*0.75, init_lr*0.5, init_lr*0.25, init_lr*0.25*0.95, init_lr*(0.25*0.9), init_lr*(0.25*0.85), init_lr*(0.25*0.8), init_lr*(0.25*0.75), init_lr*(0.25*0.7)]
        #learning_rate_piecewise = tf.train.piecewise_constant(step, boundaries, values)

        # exponential decay
        decay_step_exp = nu_epoch*2
        learning_rate_exp = tf.train.exponential_decay(init_lr, step, decay_step_exp, 0.94, staircase=True)
        learning_rate = learning_rate_exp
        tf.summary.scalar('lr',learning_rate)

        '''
        if warmup:
            warmup_steps = int(batches_per_epoch * 5)
            warmup_lr = (initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
              warmup_steps, tf.float32))
            learning_rate= tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: learning_rate)
        '''

        #tf.summary.scalar('epsilon',epsilon)
        my_optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9,use_nesterov=False)
        #my_optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9,epsilon=1.0)
        #my_optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.1, beta1=0.9, beta2=0.99)

        #train_op = my_optimizer.minimize(loss, global_step=step)
        grad_vars = my_optimizer.compute_gradients(loss)
        minimize_op = my_optimizer.apply_gradients(grad_vars, global_step=step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)


        global_init=tf.global_variables_initializer() # initializes all variables in the graph
        #gs_init = tf.variables_initializer(var_list=[step])
        #opt_var_init=tf.variables_initializer(my_optimizer.variables())
        #var_list=[var for var in tf.global_variables() ] # debug pruposes
        merged_summary= tf.summary.merge_all()

        all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_model')


    train_writer = tf.summary.FileWriter(modelDir)
    with tf.Session(graph=graph) as sess:
        #chkp.print_tensors_in_checkpoint_file(model_path, tensor_name='',all_tensors=False, all_tensor_names=True)
        sess.run(global_init) 
        #pdb.set_trace()
        
        trn_iteration=0
        for epoch in range(args.num_epochs2):
            rand_idx = np.random.permutation(total_examples)
            print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs2))
            start_idx=0
            is_eoe =False
            #iteration=0
            while not is_eoe:
                end_idx = start_idx+ (args.batch_size)
                if end_idx > exm_per_epoch:
                    ndiff = end_idx- exm_per_epoch
                    end_idx = exm_per_epoch
                    is_eoe =True

                    idx_range1 = range(start_idx,end_idx)
                    idx_range2 = range(0,ndiff)
                    idx_range = list(idx_range1) +list(idx_range2)
                else:
                    idx_range = list(range(start_idx,end_idx))

                batch_filenames = [train_filenames[idx] for idx in rand_idx[idx_range]]
                batch_labels =  [train_labels[idx] for idx in rand_idx[idx_range]]
                #pdb.set_trace()

                sess.run(train_op, feed_dict={image_names: batch_filenames, labels: batch_labels, train_mode: True})



                start_idx = end_idx
                #iteration +=1
                trn_iteration +=1

                #if iteration >=20: break
                if trn_iteration%100 == 0:
                    msummary = sess.run(merged_summary, feed_dict={image_names: batch_filenames, labels: batch_labels, train_mode: True})
                    train_writer.add_summary(msummary, trn_iteration)
                if trn_iteration%50 == 0:
                    tower_losses,batch_accuracy,lr = sess.run([loss,accuracy, learning_rate], feed_dict={image_names: batch_filenames, labels: batch_labels, train_mode: True})
                    print('epoch, itr:(',epoch,', ',trn_iteration,') lr = ',lr,' loss = ',tower_losses, 'acc = ',batch_accuracy)
                    #pdb.set_trace()
            # # pred=sess.run(tf.get_collection("logits")[0],{is_training:True})
            if epoch>49 and epoch%2==0:

                #pdb.set_trace()  
                saver0 = tf.train.Saver()
                modelName = 'model-'+str(epoch).zfill(6)
                modelPath= os.path.join(modelDir,modelName)#'/trn_dir/models/model_tmp/model-00000'
                saver0.save(sess, modelPath)
                ## Generates MetaGraphDef.
                saver0.export_meta_graph(modelPath+'.meta')

            if epoch%2==0:
                #pdb.set_trace()
                #bnorm_mean=sess.run(tf.contrib.framework.get_variables('resnet_model/batch_normalization/moving_mean:0'))
                #bnorm_variance=sess.run(tf.contrib.framework.get_variables('resnet_model/batch_normalization/moving_variane:0'))
                val_filenames, val_filelbls = list_images(args.val_dir, args.label_map)
                val_total_examples = len(val_filenames)
                val_num_correct, val_num_samples = 0, 0

                val_bsz = args.batch_size
                val_nbatches = int(val_total_examples/val_bsz)
                for nb in range(val_nbatches):
                    val_batch_names = val_filenames[nb*val_bsz:(nb+1)*val_bsz]
                    val_batch_labels = val_filelbls[nb*val_bsz:(nb+1)*val_bsz]

                    val_correct_pred = sess.run(correct_prediction, feed_dict={image_names: val_batch_names, labels: val_batch_labels, train_mode: False})
                    val_num_correct += val_correct_pred.sum()
                    val_num_samples += val_correct_pred.shape[0]
        #if num_samples % 2000 == 0: print('detected ',num_samples)  

    # Return the fraction of datapoints that were correctly classified
                val_acc = float(val_num_correct) / val_num_samples
                #val_acc = check_accuracy(sess, correct_prediction, image_names, labels, train_mode, args.val_dir, args.label_map, prediction, logits)
                print('Val accuracy: %f\n' % val_acc)

      


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
