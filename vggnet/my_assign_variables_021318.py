
import argparse
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from tensorflow.contrib import layers
import pdb
from dataloader import mydataloader_train, mydataloader_val
import numpy as np
from PIL import Image

import my_vgg_relu6_fc


parser = argparse.ArgumentParser()
#parser.add_argument('--train_dir', default='coco-animals/train')
#parser.add_argument('--val_dir', default='coco-animals/val')
parser.add_argument('--model_path', default='vgg_16.ckpt', type=str)
parser.add_argument('--batch_size', default=25, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=3, type=int)
parser.add_argument('--num_epochs2', default=3, type=int)
parser.add_argument('--learning_rate1', default=1e-3, type=float)
parser.add_argument('--learning_rate2', default=1e-5, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)

def check_accuracy(sess, correct_prediction, is_training, dataset_init_op):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    while True:
        try:
            correct_pred = sess.run(correct_prediction, {is_training: False})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def my_get_variable_names():
    # get the names from vgg_16 definition 
    variable_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    nrepeats = [2, 2, 3, 3, 3]

    #pdb.set_trace()
    vgg16_names=[]
    vgg16_t_names=[]
    for i in range(len(variable_names)): 
        vname = variable_names[i]
        nrep = nrepeats[i]
        for rr in range(nrep):
            namestr= 'vgg_16/'+vname+'/'+vname+'_'+str(rr+1)
            vgg16_names.append(namestr)
            namestr_t= 'vgg_16_t/'+vname+'/'+vname+'_'+str(rr+1)
            vgg16_t_names.append(namestr_t)

    return vgg16_names, vgg16_t_names


def copy_values_from_vgg16(vgg16_names, vgg16_t_names):

    assign_ops = []
    #for main_var, target_var in zip(main_variables, target_variables):
    for i in range(len(vgg16_names)):
        source_name = vgg16_names[i]+'/weights:0'
        source_var = tf.contrib.framework.get_variables(source_name)
        target_name = vgg16_t_names[i] + '/weights:0'  
        target_var = tf.contrib.framework.get_variables(target_name)
            
        assign_ops.append(tf.assign(target_var[0], tf.identity(source_var[0])))

    copy_operation = tf.group(*assign_ops)

    return copy_operation

def main(args):

    pdb.set_trace()
    #modelDir = '/trn_dir/models/model1'
    model_path = '/trn_dir/models/model3_relu6/model-000039'
    num_classes=20
    im0 = Image.open('/trn_dir/imgnet/data/trn/whippet/n02091134_12142.JPEG')
    im0=im0.resize((224,224))   
    image_np = load_image_into_numpy_array(im0)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    vgg16_variable_names, vgg16_t_variable_names = my_get_variable_names() 
    #builder =  tf.saved_model.builder.SavedModelBuilder(modelDir)
    # --------------------------------------------------------------------------
    # In TensorFlow, you first want to define the computation graph with all the
    # necessary operations: loss, training op, accuracy...
    # Any tensor created in the `graph.as_default()` scope will be part of `graph`
    graph = tf.Graph()
    with graph.as_default():
        

        # Indicates whether we are in training or in test mode
        is_training = tf.placeholder(tf.bool)
        images = tf.placeholder(dtype=tf.float32, shape=[None,224,224,3])
        
        res = layers.conv2d(images, 64, [3,3], scope='tmp')

        ninp_ch = tf.shape(res)[3]
        chidx = tf.range(0,tf.cast(ninp_ch/2,tf.int32))
        rndidx = tf.random.shuffle(chidx)
        res2 = tf.gather(res, chidx, axis=3) #res[:,:,:,rndidx,:]
        
 
        res2.set_shape([None,res.shape[1],res.shape[2], res.shape[3].__floordiv__(2)] )
        res1 = layers.conv2d(res2, 128, [3,3], scope='tmp_2c')

        chidx1 = tf.range(ninp_ch)
        chidx1_shift = tf.mod(chidx1+1,ninp_ch)
        

        #vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(my_vgg_relu6_fc.vgg_arg_scope(weight_decay=args.weight_decay)):
            logits, endpoints_dict = my_vgg_relu6_fc.vgg_16(images, num_classes=num_classes, is_training=is_training,dropout_keep_prob=args.dropout_keep_prob)


        with slim.arg_scope(my_vgg_relu6_fc.vgg_arg_scope(weight_decay=args.weight_decay)):
            logits_t, endpoints_dict_t = my_vgg_relu6_fc.vgg_16_t(images, num_classes=num_classes, is_training=is_training,dropout_keep_prob=args.dropout_keep_prob)

        # Specify where the model checkpoint is (pretrained weights).
        #model_path = args.model_path
        #assert(os.path.isfile(model_path))

 
        # Restore only the layers up to fc7 (included)
        # Calling function `init_fn(sess)` will load all the pretrained weights.
        
        main_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_16')
        #target_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_16_t')
        #variables_to_restore = tf.contrib.framework.get_variables_to_restore()#exclude=['vgg_16/fc8'])

        vgg_trainable_names = tf.trainable_variables(scope='vgg_16') 
        vgg_t_trainable_names = tf.trainable_variables('vgg_16_t')

        variables_to_restore = tf.contrib.framework.get_variables_to_restore(vgg16_variable_names)
        var_load_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

        #new_variables = tf.contrib.framework.get_variables('vgg_16_ext')
        #new_var_init = tf.variables_initializer(var_list=fc8_variables)
 
        copy_op = copy_values_from_vgg16(vgg16_variable_names, vgg16_t_variable_names)

        conv_weights = tf.contrib.framework.get_variables('vgg_16/conv2/conv2_1/weights:0')
        zero_weights = tf.zeros_like(conv_weights, dtype=conv_weights[0].dtype)
        ext_weights = tf.concat([conv_weights, zero_weights], axis=3)


        c1_1w=tf.contrib.framework.get_variables('vgg_16/conv1/conv1_1/biases:0')
        tc1_1w=tf.contrib.framework.get_variables('vgg_16_t/conv1/conv1_1/biases:0')
        assign_op = tf.assign(tc1_1w[0],tf.identity(c1_1w[0]))

        idx = tf.range(0,15)
        randidx = tf.random.shuffle(idx)

        '''
        conv_weights = tf.contrib.framework.get_variables('vgg_16_ext/conv2/conv2_1/weights:0')
        conv_input_dim = tf.shape(conv_weights)[3] 
        new_weights = tf.contrib.framework.get_variables('vgg_16_ext/conv2/conv2_1/weights:0')
        ext_weights = tf.concat([conv_weights, new_weights[:,:,:,conv_input_dim:,:]], axis=3)
        '''

        '''
        '''

        # Initialization operation from scratch for the new "fc8" layers
        # `get_variables` will only return the variables whose name starts with the given pattern
        global_init=tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        pdb.set_trace()
        #sess.run(new_var_init)
        sess.run(global_init)
        var_load_fn(sess)  # load the pretrained weights
 
        # print name, shape of each trainable variable

        trainable_values = sess.run(vgg_trainable_names)
        for nm, v in zip(vgg_trainable_names, trainable_values):
            print('name = ',nm.name, ' shape = ',v.shape)    

        conv_val = sess.run(conv_weights, {images:image_np_expanded, is_training:True})
        ext_val = sess.run(ext_weights, {images:image_np_expanded, is_training:True})
        
        randidx = sess.run(randidx, {images:image_np_expanded, is_training:True})
        pp=1     


if __name__ == '__main__':
    args = parser.parse_args()

    main(args)
