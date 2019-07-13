
import argparse
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import pdb
import preprocess

import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='coco-animals/train')
parser.add_argument('--val_dir', default='coco-animals/val')
parser.add_argument('--label_map', default='/trn_dir/imgnet/data/labelmap.json')
parser.add_argument('--model_dir', default='vgg_16.ckpt', type=str)
parser.add_argument('--batch_size', default=25, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=3, type=int)
parser.add_argument('--num_epochs2', default=3, type=int)
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



def get_input_image(imagename, train_mode):
    pp = preprocess.preprocessor()
    image = pp._parse_function(imagename)
    cimage = tf.cond(train_mode, lambda: pp.training_preprocess(image),lambda: pp.val_preprocess(image))
    return cimage


def main(args):

    #pdb.set_trace()
    train_filenames, train_labels = list_images(args.train_dir, args.label_map)
    total_examples = len(train_labels)
    num_classes = len(set(train_labels))
    modelDir = args.model_dir

    graph = tf.Graph()
    with graph.as_default():

        image_names = tf.placeholder(dtype=tf.string, name='image_names')
        labels = tf.placeholder(dtype=tf.int32, name='labels')
        train_mode = tf.placeholder(dtype=tf.bool,name = 'train_mode')

        preprocess_fn = lambda x: get_input_image(x, train_mode)

        images = tf.map_fn(preprocess_fn, image_names,dtype=tf.float32)
        #images = tf.concat(image_array, axis=0)
        #nimages = tf.shape(images)
        

        vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
            logits, endpoints_dict = vgg.vgg_16(images, num_classes=num_classes, is_training=train_mode,dropout_keep_prob=args.dropout_keep_prob)

        tf.add_to_collection("logits",logits)

        #loss= tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits))
        # replace the following with above
        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss = tf.losses.get_total_loss()

        tf.summary.scalar('loss',loss)
        #step = tf.Variable(0, trainable=False)
        step=tf.train.get_or_create_global_step()        
        #boundaries = [100000, 150000]
        #values = [0.001, 0.0009, 0.0008]
        #piecewise_lr = tf.train.piecewise_constant(global_step, boundaries, values)
        learning_rate = tf.train.exponential_decay(0.001, step, 80000, 0.95, staircase=True)
        tf.summary.scalar('lr',learning_rate)
         
        #epsilon = tf.math.maximum(1e-8, tf.math.pow(1.0,(0.01+tf.cast(tf.math.floordiv(step,10000),tf.float32)))) 

        #tf.summary.scalar('epsilon',epsilon)
        #full_optimizer = tf.train.RMSPropOptimizer(args.learning_rate2, momentum=0.9,epsilon=0.01)
        full_optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.1, beta1=0.9, beta2=0.99)
        full_train_op = full_optimizer.minimize(loss, global_step=step)

        prediction = tf.to_int32(tf.argmax(logits, 1))
        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('batch_accuracy',accuracy)

        
        global_init=tf.global_variables_initializer() # initializes all variables in the graph

        gs_init = tf.variables_initializer(var_list=[step])
        opt_var_init=tf.variables_initializer(full_optimizer.variables())
        var_list=[var for var in tf.global_variables() ] # debug pruposes

        #modelSaver = tf.train.Saver()

        #tf.get_default_graph().finalize()
        merged_summary= tf.summary.merge_all()
        

    train_writer = tf.summary.FileWriter(modelDir)
    with tf.Session(graph=graph) as sess:
        sess.run(global_init)
        #pdb.set_trace() 
        #tmpname, ni = sess.run([image_array, nimages], feed_dict={image_names:['/trn_dir/imgnet/data/trn/whippet/n02091134_11956.JPEG', '/trn_dir/imgnet/data/trn/whippet/n02091134_13143.JPEG'], labels:[1, 1]})
        trn_iteration=0
        for epoch in range(args.num_epochs2):
            rand_idx = np.random.permutation(total_examples)
            print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs2))
            start_idx=0
            is_eoe =False
            #iteration=0
            while not is_eoe:
                end_idx = start_idx+ args.batch_size
                if end_idx > total_examples:
                    end_idx = total_examples
                    is_eoe =True  

                batch_filenames = [train_filenames[idx] for idx in rand_idx[start_idx:end_idx]]        
                batch_labels =  [train_labels[idx] for idx in rand_idx[start_idx:end_idx]]  
                #pdb.set_trace()
    
                msummary, rloss, _ = sess.run([merged_summary, loss,full_train_op], feed_dict={image_names: batch_filenames, labels: batch_labels, train_mode: True})
                #print('epoch: ',epoch, ' iteration: ',iteration, ' loss = ',dummy[0])
                fc8_biases_val = sess.run(tf.contrib.framework.get_variables('vgg_16/fc8/biases:0'))
                #pdb.set_trace()
                #endpoints_dict= sess.run([endpoints_dict],{is_training:True})
                if np.isnan(fc8_biases_val[0][0]):
                    pdb.set_trace()

                start_idx = end_idx
                #iteration +=1
                trn_iteration +=1

                #if iteration >=20: break
                if trn_iteration%20 == 0:
                    train_writer.add_summary(msummary, trn_iteration)
                if trn_iteration%500 == 0:
                    print('epoch: ',epoch, ' iteration: ',trn_iteration, ' loss = ',rloss)

            # # pred=sess.run(tf.get_collection("logits")[0],{is_training:True})

            #pdb.set_trace()  
            saver0 = tf.train.Saver()
            modelName = 'model-'+str(epoch).zfill(6)
            modelPath= os.path.join(modelDir,modelName)#'/trn_dir/models/model_tmp/model-00000'
            saver0.save(sess, modelPath)
            ## Generates MetaGraphDef.
            saver0.export_meta_graph(modelPath+'.meta')

            val_acc = check_accuracy(sess, correct_prediction, image_names, labels, train_mode, args.val_dir, args.label_map)
            train_acc = check_accuracy(sess, correct_prediction, image_names, labels, train_mode, args.train_dir, args.label_map)

            ## Check accuracy on the train and val sets every epoch
            #train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
            #val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
            print('Train accuracy: %f' % train_acc)
            print('Val accuracy: %f\n' % val_acc)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    #main()
