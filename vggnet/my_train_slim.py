
import argparse
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import pdb
from dataloader import mydataloader_train, mydataloader_val

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

def image_preprocessing_fucntion(image):

    # for training

    image = tf.cast(image_decoded, tf.float32)

    # rescale/warp
    smallest_side = 256.0
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height = tf.to_float(height)
    width = tf.to_float(width)

    scale = tf.cond(tf.greater(height, width),lambda: smallest_side / width,lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)

    # center crop
    crop_image = tf.random_crop(resized_image, [224, 224, 3])                       # (3)
    flip_image = tf.image.random_flip_left_right(crop_image)                # (4)

    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    centered_image = flip_image - means                                     # (5)

    return centered_image, label



def main(args):

    #pdb.set_trace()
    train_filenames, train_labels = list_images(args.train_dir, args.label_map)
    #total_examples = len(train_labels)
    #num_classes = len(set(train_labels))
    modelDir = args.model_dir
    batchSz=25

    graph = tf.Graph()
    with graph.as_default():

        #create tfrecords for the data
        # create slim.data.dataset for the tfrecords with the encoder and decoder
        # create slim.dataset_data_provider for the dataset  

        train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))

        provider = slim.dataset_data_provider.DatasetDataProvider(train_dataset, common_queue_capacity=20 * batchSz, common_queue_min=10 * batchSz)
        [image, label] = provider.get(['image', 'label'])
        label -= FLAGS.labels_offset


        image = image_preprocessing_fn(image)

        images, labels = tf.train.batch(
          [image, label],
          batch_size=batchSz,
          capacity=5 * batchSz)

        #labels = slim.one_hot_encoding(labels, dataset.num_classes - FLAGS.labels_offset)



        #image_names = tf.placeholder(dtype=tf.string, name='image_names')
        #labels = tf.placeholder(dtype=tf.int32, name='labels')
        #train_mode = tf.placeholder(dtype=tf.bool,name = 'train_mode')

        #preprocess_fn = lambda x: get_input_image(x, train_mode)

        #images = tf.map_fn(preprocess_fn, image_names,dtype=tf.float32)
        
        #image_names = [train_filenames[0]]
        #labels = [train_labels[0]]
        #images = [get_input_image(x,train_mode=True) for x in image_names] 

        dataldr_trn=mydataloader_train(args.train_dir,args.label_map)
        #dataldr_val=mydataloader_val(args.val_dir,args.label_map)

        iterator = tf.data.Iterator.from_structure(dataldr_trn.batched_train_dataset.output_types,dataldr_trn.batched_train_dataset.output_shapes)
        images, labels = iterator.get_next()

        train_init_op = iterator.make_initializer(dataldr_trn.batched_train_dataset)

        vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
            logits, endpoints_dict = vgg.vgg_16(images, num_classes=dataldr_trn.num_classes, is_training=True,dropout_keep_prob=args.dropout_keep_prob)

        tf.add_to_collection("logits",logits)

        # add labels = slim.one_hot_encoding(labels, num_classes=dataldr_trn.num_classes,) 
        # for using slim.softmax_cross_entropy()

        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss = tf.losses.get_total_loss()
        tf.summary.scalar('total_loss', loss)

        #step = tf.Variable(0, trainable=False)
        step = slim.get_or_create_global_step()
        #boundaries = [100000, 150000]
        #values = [0.001, 0.0009, 0.0008]
        #piecewise_lr = tf.train.piecewise_constant(global_step, boundaries, values)
        learning_rate = tf.train.exponential_decay(0.001, step, 50000, 0.95, staircase=True)

        tf.summary.scalar('iter', step)
        tf.summary.scalar('learning_rate', learning_rate)
        #full_optimizer = tf.train.RMSPropOptimizer(args.learning_rate2, momentum=0.9,epsilon=0.01)
        full_optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.01) #beta1=0.9, beta2=0.99
        #full_train_op = full_optimizer.minimize(loss, global_step=step)

        # create_train_op that ensures that when we evaluate it to get the loss,
        # the update_ops are done and the gradient updates are computed.
        train_tensor = slim.learning.create_train_op(loss, full_optimizer)


        # Actually runs training.
        slim.learning.train(train_tensor, modelDir, local_init_op = train_init_op, global_step=step, number_of_steps=100000,save_summaries_secs=10, save_interval_secs=1800)

        '''

        prediction = tf.to_int32(tf.argmax(logits, 1))

        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        
        global_init=tf.global_variables_initializer() # initializes all variables in the graph

        gs_init = tf.variables_initializer(var_list=[step])
        opt_var_init=tf.variables_initializer(full_optimizer.variables())
        var_list=[var for var in tf.global_variables() ] # debug pruposes

        #modelSaver = tf.train.Saver()

        #tf.get_default_graph().finalize()




    with tf.Session(graph=graph) as sess:
        sess.run(global_init)
        #pdb.set_trace() 
        #tmpname, ni = sess.run([image_array, nimages], feed_dict={image_names:['/trn_dir/imgnet/data/trn/whippet/n02091134_11956.JPEG', '/trn_dir/imgnet/data/trn/whippet/n02091134_13143.JPEG'], labels:[1, 1]})
        for epoch in range(args.num_epochs2):
            rand_idx = np.random.permutation(total_examples)
            print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs2))
            start_idx=0
            is_eoe =False
            iteration=0
            while not is_eoe:
                end_idx = start_idx+ args.batch_size
                if end_idx > total_examples:
                    end_idx = total_examples
                    is_eoe =True  

                batch_filenames = [train_filenames[idx] for idx in rand_idx[start_idx:end_idx]]        
                batch_labels =  [train_labels[idx] for idx in rand_idx[start_idx:end_idx]]  
                #pdb.set_trace()
    
                dummy = sess.run([loss,full_train_op], feed_dict={image_names: batch_filenames, labels: batch_labels, train_mode: True})
                #print('epoch: ',epoch, ' iteration: ',iteration, ' loss = ',dummy[0])
                fc8_biases_val = sess.run(tf.contrib.framework.get_variables('vgg_16/fc8/biases:0'))
                #pdb.set_trace()
                #endpoints_dict= sess.run([endpoints_dict],{is_training:True})
                if np.isnan(fc8_biases_val[0][0]):
                    pdb.set_trace()

                start_idx = end_idx
                iteration +=1
                #if iteration >=20: break
                if iteration%500 == 0:
                    print('epoch: ',epoch, ' iteration: ',iteration, ' loss = ',dummy[0])

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

            '''

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    #main()
