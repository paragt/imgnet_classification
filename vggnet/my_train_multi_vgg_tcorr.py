
import argparse
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import pdb
import preprocess

import numpy as np
import json

import my_vgg_relu6_fc

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='coco-animals/train')
parser.add_argument('--val_dir', default='coco-animals/val')
parser.add_argument('--label_map', default='/trn_dir/imgnet/data/labelmap.json')
parser.add_argument('--model_dir', default='vgg_16.ckpt', type=str)
parser.add_argument('--corr_op', default='AND', type=str) # AND, XOR, LINEAR
parser.add_argument('--batch_size', default=25, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs2', default=3, type=int)
parser.add_argument('--num_gpus', default=1, type=int)
parser.add_argument('--learning_rate1', default=1e-3, type=float)
parser.add_argument('--learning_rate2', default=1e-5, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)

#corr_features = {}


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



def tower_loss(data_tensor, label_tensor, num_classes, train_mode, corr_features, corr_op):

    #vgg = tf.contrib.slim.nets.vgg
    with slim.arg_scope(my_vgg_relu6_fc.vgg_arg_scope(weight_decay=args.weight_decay)):
        logits, endpoints_dict = my_vgg_relu6_fc.vgg_16_tcorr(data_tensor, corr_features, corr_op, num_classes=num_classes, is_training=train_mode,dropout_keep_prob=args.dropout_keep_prob)


    loss=tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=label_tensor, logits=logits))

    return loss, logits

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
            namestrw = namestr+'/weights:0'
            namestrb = namestr+'/biases:0'
            vgg16_names.append(namestrw)
            vgg16_names.append(namestrb)
            namestr_t= 'vgg_16_tcorr/'+vname+'/'+vname+'_'+str(rr+1)
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

        source_name = vgg16_names[i]+'/biases:0'
        source_var = tf.contrib.framework.get_variables(source_name)
        target_name = vgg16_t_names[i] + '/biases:0'
        target_var = tf.contrib.framework.get_variables(target_name)

        assign_ops.append(tf.assign(target_var[0], tf.identity(source_var[0])))

    copy_operation = tf.group(*assign_ops)

    return copy_operation




def main(args):

    #model_path = '/trn_dir/models/model_multi_cl20_relu6/model-000075' 
    model_path = '/trn_dir/models/model_multi_cl20_preprocess/model-000060' 

    train_filenames, train_labels = list_images(args.train_dir, args.label_map)
    total_examples = len(train_labels) # imgnet 1,281,167
    num_classes = len(set(train_labels))
    modelDir = args.model_dir

    vgg16_variable_names, vgg16_t_variable_names = my_get_variable_names()

    #global corr_features
    featcorrname = os.path.basename(model_path)+'_featcorr.json'
    fid = open(featcorrname)
    corr_features = json.load(fid)
    fid.close()

    phase1_nepoch = 5#10
    corr_op = args.corr_op # AND, XOR, LINEAR


    #pdb.set_trace()
    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):

        #step = tf.Variable(0, trainable=False)
        step=tf.train.get_or_create_global_step()
        #boundaries = [100000, 150000]
        #values = [0.001, 0.0009, 0.0008]
        nu_epoch = tf.cast(total_examples*1.0/(args.batch_size*args.num_gpus), tf.int32) # num_update_per_epoch
        init_lr = 0.005
        #boundaries = [10*nu_epoch, 15*nu_epoch, 20*nu_epoch, 25*nu_epoch, 30*nu_epoch, 35*nu_epoch, 40*nu_epoch, 45*nu_epoch, 50*nu_epoch, 55*nu_epoch, 60*nu_epoch, 65*nu_epoch, 70*nu_epoch]
        #values=[init_lr, init_lr*0.75, init_lr*0.5, init_lr*0.25, init_lr*0.125, init_lr*(0.125*0.9), init_lr*(0.125*0.81), init_lr*(0.125*0.729), init_lr*(0.125*0.6561), init_lr*(0.125*0.59049), init_lr*(0.125*0.531441), init_lr*(0.125*0.4783), init_lr*(0.125*0.4304), init_lr*(0.125*0.3874)]
        ##init_lr = 0.002
        ##values=[init_lr, init_lr*0.75, init_lr*0.5, init_lr*0.25]
        #learning_rate = tf.train.piecewise_constant(step, boundaries, values)

        decay_step_exp = 2*nu_epoch
        learning_rate = tf.train.exponential_decay(init_lr, step, decay_step_exp, 0.94, staircase=True)
        tf.summary.scalar('lr',learning_rate)
         
        #epsilon = tf.math.maximum(1e-8, tf.math.pow(1.0,(0.01+tf.cast(tf.math.floordiv(step,10000),tf.float32)))) 

        #tf.summary.scalar('epsilon',epsilon)
        #full_optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9,use_nesterov=True)
        #full_optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9,epsilon=0.1)
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

        #main_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_16')
        #all_trainable_names = tf.trainable_variables(scope='vgg_16')
        #existing_trainable_names = [x.name for x in all_trainable_names if x.name in vgg16_variable_names]
        #existing_variables = [tf.contrib.framework.get_variables(x) for x in existing_trainable_names]
        #new_trainable_names = [x.name for x in all_trainable_names if x.name not in vgg16_variable_names]
        #new_variables = [tf.contrib.framework.get_variables(x) for x in new_trainable_names]




        tower_grads = []
        tower_grads_new = []
        loss_array = []
        logits_array= []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.name_scope('tower_%d' % (i)), tf.device('/gpu:%d' % i):
                    #with tf.name_scope('tower_%d' % (i)) as scope:

                    #X = images[i*args.batch_size : (i+1)*args.batch_size]
                    #Y = labels[i*args.batch_size : (i+1)*args.batch_size]
                    loss, logits = tower_loss(split_data[i], split_labels[i], num_classes, train_mode, corr_features, corr_op)

                    all_trainable_names = tf.trainable_variables(scope='vgg_16')
                    new_trainable_names = [x.name for x in all_trainable_names if x.name not in vgg16_variable_names]
                    new_variables = [tf.contrib.framework.get_variables(x) for x in new_trainable_names]

                    # Calculate the gradients for the batch of data on this  tower.
                    grads = full_optimizer.compute_gradients(loss)
                    grads_new = full_optimizer.compute_gradients(loss, var_list=new_variables)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)
                    tower_grads_new.append(grads_new)
                    loss_array.append(loss)
                    logits_array.append(logits)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.

        avg_grads_new = average_gradients(tower_grads_new)
        train_op_new = full_optimizer.apply_gradients(avg_grads_new, global_step=step)
        avg_grads = average_gradients(tower_grads)
        train_op = full_optimizer.apply_gradients(avg_grads, global_step=step)

        #avg_grads_selected = average_gradients(tower_grads)
        #train_op_selected = full_optimizer.apply_gradients(avg_grads_selected, global_step=step)

        
        tf.summary.scalar('max_loss',tf.reduce_max(loss_array))


        logits_array = tf.concat(values=logits_array,axis=0)
        #tf.add_to_collection("logits",logits)

        prediction = tf.to_int32(tf.argmax(logits_array, 1))
        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('batch_accuracy',accuracy)

        
        global_init=tf.global_variables_initializer() # initializes all variables in the graph

        #with slim.arg_scope(my_vgg_relu6_fc.vgg_arg_scope(weight_decay=args.weight_decay)):
            #logits, endpoints_dict = my_vgg_relu6_fc.vgg_16(images, num_classes=num_classes, is_training=train_mode,dropout_keep_prob=args.dropout_keep_prob)

        variables_to_restore = tf.contrib.framework.get_variables_to_restore(vgg16_variable_names)
        var_load_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path,  variables_to_restore)

        #copy_op = copy_values_from_vgg16(vgg16_variable_names, vgg16_t_variable_names)


        #modelSaver = tf.train.Saver()

        merged_summary= tf.summary.merge_all()
        

    train_writer = tf.summary.FileWriter(modelDir)
    with tf.Session(graph=graph) as sess:
        sess.run(global_init)
        #pdb.set_trace() 
        var_load_fn(sess)
        #_ = sess.run(copy_op)
        trn_iteration=0

        ops = graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}

        for epoch in range(args.num_epochs2):
            rand_idx = np.random.permutation(total_examples)
            print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs2))
            start_idx=0
            is_eoe =False
            #iteration=0
            while not is_eoe:
                end_idx = start_idx+ (args.batch_size*args.num_gpus)
                if end_idx > total_examples:
                    ndiff = end_idx-total_examples
                    end_idx = total_examples
                    is_eoe =True

                    idx_range1 = range(start_idx,end_idx)
                    idx_range2 = range(0,ndiff)
                    idx_range = list(idx_range1) +list(idx_range2) 
                else:
                    idx_range = list(range(start_idx,end_idx))      

                batch_filenames = [train_filenames[idx] for idx in rand_idx[idx_range]]        
                batch_labels =  [train_labels[idx] for idx in rand_idx[idx_range]]  
                #pdb.set_trace()
    
                if epoch <= phase1_nepoch:
                    sess.run(train_op_new, feed_dict={image_names: batch_filenames, labels: batch_labels, train_mode: True})
                else:
                    sess.run(train_op, feed_dict={image_names: batch_filenames, labels: batch_labels, train_mode: True})
  
                #if epoch < phase1_nepoch:
                    #_ = sess.run(copy_op)
                 
                # *debug*
                # fc8_biases_val = sess.run(tf.contrib.framework.get_variables('vgg_16/fc8/biases:0')) 
                # name_scape does not affect get_variables
 
                #if np.isnan(fc8_biases_val[0][0]):
                    #pdb.set_trace()
                #endpoints_dict= sess.run([endpoints_dict],{is_training:True})
                # * *
 

                start_idx = end_idx
                #iteration +=1
                trn_iteration +=1

                #if iteration >=20: break
                if trn_iteration%50 == 0:
                    msummary = sess.run(merged_summary, feed_dict={image_names: batch_filenames, labels: batch_labels, train_mode: True})   
                    train_writer.add_summary(msummary, trn_iteration)
                if trn_iteration%100 == 0:
                    tower_losses, batch_accuracy, lr  = sess.run([loss_array,accuracy,learning_rate], feed_dict={image_names: batch_filenames, labels: batch_labels, train_mode: True})
                    print('epoch, itr:(',epoch,', ',trn_iteration,') lr = ',lr,' loss = ',max(tower_losses), 'acc = ',batch_accuracy)

            # # pred=sess.run(tf.get_collection("logits")[0],{is_training:True})
            if epoch>= phase1_nepoch and epoch%2==0:
                '''
                #pdb.set_trace()  
                saver0 = tf.train.Saver()
                modelName = 'model-'+str(epoch).zfill(6)
                modelPath= os.path.join(modelDir,modelName)#'/trn_dir/models/model_tmp/model-00000'
                saver0.save(sess, modelPath)
                ## Generates MetaGraphDef.
                saver0.export_meta_graph(modelPath+'.meta')
                '''

                print('computing trn, val accuracies ...') 
                val_acc = check_accuracy(sess, correct_prediction, image_names, labels, train_mode, args.val_dir, args.label_map)
                #train_acc = check_accuracy(sess, correct_prediction, image_names, labels, train_mode, args.train_dir, args.label_map)

                #print('Train accuracy: %f' % train_acc)
                print('Val accuracy: %f\n' % val_acc)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    #main()
