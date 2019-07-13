
import argparse
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import pdb
import preprocess

import numpy as np
import json

import vgg_relu6


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
parser.add_argument('--weight_decay', default=5e-4, type=float)




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

def check_accuracy(sess,correct_prediction, image_names, labels, train_mode, dirname, labelmap_file):

    filenames, filelbls = list_images(dirname, labelmap_file)
    total_examples = len(filenames)
    num_correct, num_samples = 0, 0
    #pdb.set_trace()

    bsz = args.batch_size*args.num_gpus
    nbatches = int(total_examples/bsz)
    for nb in range(nbatches):
        batch_names = filenames[nb*bsz:(nb+1)*bsz]
        batch_labels = filelbls[nb*bsz:(nb+1)*bsz]

        correct_pred = sess.run(correct_prediction, feed_dict={image_names: batch_names, labels: batch_labels, train_mode: False})
        num_correct += correct_pred.sum()
        num_samples += correct_pred.shape[0]
        #if num_samples % 2000 == 0: print('detected ',num_samples)  

    if total_examples > (nb+1)*bsz:
        batch_names = filenames[(nb+1)*bsz:total_examples]
        batch_labels = filelbls[(nb+1)*bsz:total_examples]
        nvalid_examples = len(batch_names)

        while len(batch_names)<bsz:
            batch_names.append(filenames[0])
            batch_labels.append(filelbls[0])

        correct_pred = sess.run(correct_prediction, feed_dict={image_names: batch_names, labels: batch_labels, train_mode: False})

        correct_pred = correct_pred[:nvalid_examples]

        num_correct += correct_pred.sum()
        num_samples += correct_pred.shape[0]

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc



def get_input_image(imagename, train_mode):
    pp = preprocess.preprocessor()
    image = pp._parse_function(imagename)
    cimage = tf.cond(train_mode, lambda: pp.training_preprocess(image),lambda: pp.val_preprocess(image))
    return cimage

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads



def tower_loss(data_tensor, label_tensor, num_classes, train_mode):

    with slim.arg_scope(vgg_relu6.vgg_arg_scope(weight_decay=args.weight_decay)):
        logits, endpoints_dict = vgg_relu6.vgg_16(data_tensor, num_classes=num_classes, is_training=train_mode,dropout_keep_prob=args.dropout_keep_prob)


    loss=tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=label_tensor, logits=logits))

    return loss, logits

def main(args):

    #pdb.set_trace()
    train_filenames, train_labels = list_images(args.train_dir, args.label_map)
    total_examples = len(train_labels) # imgnet 1,281,167
    if args.epoch_size<1:
        exm_per_epoch = total_examples
    else:
        exm_per_epoch = args.epoch_size #int(total_examples/2)
    print('epoch size = ', exm_per_epoch)
    print('weight decay =', args.weight_decay)

    num_classes = len(set(train_labels))
    modelDir = args.model_dir

    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):

        #step = tf.Variable(0, trainable=False)
        step=tf.train.get_or_create_global_step()
        #boundaries = [100000, 150000]
        #values = [0.001, 0.0009, 0.0008]
        nu_epoch = tf.cast(exm_per_epoch*1.0/(args.batch_size*args.num_gpus), tf.int32) # num_update_per_epoch
        boundaries = [10*nu_epoch, 20*nu_epoch, 30*nu_epoch, 40*nu_epoch, 50*nu_epoch, 55*nu_epoch, 60*nu_epoch, 65*nu_epoch, 70*nu_epoch, 75*nu_epoch]
        init_lr = 0.005
        values=[init_lr, init_lr*0.85, init_lr*0.75, init_lr*0.5, init_lr*0.25, init_lr*0.125, init_lr*(0.125*0.9), init_lr*(0.125*0.81), init_lr*(0.125*0.729), init_lr*(0.125*0.6561), init_lr*(0.125*0.59049)]
        #init_lr = 0.002
        #values=[init_lr, init_lr*0.75, init_lr*0.5, init_lr*0.25]
        learning_rate = tf.train.piecewise_constant(step, boundaries, values)

        
        #learning_rate = tf.train.exponential_decay(0.0025, step, 80000, 0.95, staircase=True)
        tf.summary.scalar('lr',learning_rate)
         
        #epsilon = tf.math.maximum(1e-8, tf.math.pow(1.0,(0.01+tf.cast(tf.math.floordiv(step,10000),tf.float32)))) 

        #tf.summary.scalar('epsilon',epsilon)
        #full_optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9,use_nesterov=True)
        #full_optimizer = tf.train.RMSPropOptimizer(args.learning_rate2, momentum=0.9,epsilon=1.0)
        full_optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.1, beta1=0.9, beta2=0.99)

        image_names = tf.placeholder(dtype=tf.string, name='image_names')
        labels = tf.placeholder(dtype=tf.int32, name='labels')
        train_mode = tf.placeholder(dtype=tf.bool,name = 'train_mode')

        preprocess_fn = lambda x: get_input_image(x, train_mode)

        images = tf.map_fn(preprocess_fn, image_names,dtype=tf.float32)
        #images = tf.concat(image_array, axis=0)
        #nimages = tf.shape(images)
        split_data=tf.split(images, args.num_gpus)
        split_labels = tf.split(labels, args.num_gpus)


        tower_grads = []
        loss_array = []
        logits_array= []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.name_scope('tower_%d' % (i)), tf.device('/gpu:%d' % i):
                    #with tf.name_scope('tower_%d' % (i)) as scope:

                    #X = images[i*args.batch_size : (i+1)*args.batch_size]
                    #Y = labels[i*args.batch_size : (i+1)*args.batch_size]
                    loss, logits = tower_loss(split_data[i], split_labels[i], num_classes, train_mode)


                    # Calculate the gradients for the batch of data on this CIFAR tower.
                    grads = full_optimizer.compute_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)
                    loss_array.append(loss)
                    logits_array.append(logits)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        avg_grads = average_gradients(tower_grads)
        train_op = full_optimizer.apply_gradients(avg_grads, global_step=step)




        
        tf.summary.scalar('max_loss',tf.reduce_max(loss_array))


        logits_array = tf.concat(values=logits_array,axis=0)
        tf.add_to_collection("logits",logits_array)

        prediction = tf.to_int32(tf.argmax(logits_array, 1))
        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('batch_accuracy',accuracy)

        
        global_init=tf.global_variables_initializer() # initializes all variables in the graph


        #modelSaver = tf.train.Saver()

        merged_summary= tf.summary.merge_all()
        

    train_writer = tf.summary.FileWriter(modelDir)
    with tf.Session(graph=graph) as sess:
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
                end_idx = start_idx+ (args.batch_size*args.num_gpus)
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

                 
                #tower_losses, _, batch_accuracy, lr, msummary = sess.run([loss_array,train_op,accuracy,learning_rate, merged_summary], feed_dict={image_names: batch_filenames, labels: batch_labels, train_mode: True})
                # *debug*
                # fc8_biases_val = sess.run(tf.contrib.framework.get_variables('vgg_16/fc8/biases:0')) 
                # name_scape does not affect get_variables
 
                #if np.isnan(fc8_biases_val[0][0]):
                    #pdb.set_trace()
                #endpoints_dict= sess.run([endpoints_dict],{is_training:True})
                # * *
 

                start_idx = end_idx
                #iteration +=1

                #if iteration >=20: break
                if trn_iteration%50 == 0:
                    msummary = sess.run(merged_summary, feed_dict={image_names: batch_filenames, labels: batch_labels, train_mode: True})   
                    train_writer.add_summary(msummary, trn_iteration)
                if trn_iteration%50 == 0:
                    tower_losses,batch_accuracy,lr = sess.run([loss_array,accuracy, learning_rate], feed_dict={image_names: batch_filenames, labels: batch_labels, train_mode: True})
                    print('epoch, itr:(',epoch,', ',trn_iteration,') lr = ',lr,' loss = ',max(tower_losses), 'acc = ',batch_accuracy)

                trn_iteration +=1

            # # pred=sess.run(tf.get_collection("logits")[0],{is_training:True})
            if epoch%5==0:
                
                #pdb.set_trace()  
                saver0 = tf.train.Saver()
                modelName = 'model-'+str(epoch).zfill(6)
                modelPath= os.path.join(modelDir,modelName)#'/trn_dir/models/model_tmp/model-00000'
                saver0.save(sess, modelPath)
                ## Generates MetaGraphDef.
                saver0.export_meta_graph(modelPath+'.meta')
                

                print('computing trn, val accuracies ...') 
                val_acc = check_accuracy(sess, correct_prediction, image_names, labels, train_mode, args.val_dir, args.label_map)
                #train_acc = check_accuracy(sess, correct_prediction, image_names, labels, train_mode, args.train_dir, args.label_map)

                #print('Train accuracy: %f' % train_acc)
                print('Val accuracy: %f\n' % val_acc)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    #main()
