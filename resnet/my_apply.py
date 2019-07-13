import argparse
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import pdb
import preprocess

import numpy as np
import json

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


if __name__ == '__main__':

    pdb.set_trace()
    filenames, filelbls = list_images('/trn_dir/imgnet/data/val/','/trn_dir/imgnet/data/labelmap.json')
    total_examples = len(filenames)
    num_correct, num_samples = 0, 0
    #pdb.set_trace()

    bsz = 25
    nbatches = int(total_examples/bsz)

    #new_saver = tf.train.Saver()
    graph=tf.Graph()
    with tf.Session(graph=graph) as sess:
        new_saver = tf.train.import_meta_graph('/trn_dir/models/model2/model-000039.meta')
        new_saver.restore(sess, '/trn_dir/models/model2/model-000039')
        # tf.get_collection() returns a list. In this example we only want the
        # first one.
        image_names = graph.get_tensor_by_name('image_names:0')
        labels = graph.get_tensor_by_name('labels:0')
        train_mode = graph.get_tensor_by_name('train_mode:0')
        logits = tf.get_collection('logits')[0]


        for nb in range(nbatches):
            batch_names = filenames[nb*bsz:(nb+1)*bsz]
            batch_labels = filelbls[nb*bsz:(nb+1)*bsz]



            pred=sess.run(logits,feed_dict={image_names: batch_names, labels:batch_labels, train_mode:False})

            prediction = (np.argmax(pred, 1)).astype(np.int32)
            correct_pred = np.int32(prediction==np.array(batch_labels))
            num_correct += np.sum(correct_pred)
            num_samples += correct_pred.shape[0]
        #if num_samples % 2000 == 0: print('detected ',num_samples)  

    # Return the fraction of datapoints that were correctly classified
        acc = float(num_correct) / num_samples
        print('accuracy = ',acc)
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
