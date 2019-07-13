
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
import json
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
            namestr_t= 'vgg_16_tcorr/'+vname+'/'+vname+'_'+str(rr+1)
            vgg16_t_names.append(namestr_t)

    return vgg16_names, vgg16_t_names



def compute_feature_correlation(weights, biases):
 
    #pdb.set_trace() 
    
    nfeat = weights.shape[-1]
    corr_featid=np.zeros(nfeat, dtype=np.int32)
    for ff in range(nfeat):
        ww = weights[..., ff]
        wwr = np.tile(np.expand_dims(ww,axis=3), [1,1,1,nfeat])  
        diff = weights - wwr # other difference/similariy measures can be used
        udiff = np.absolute(diff)
        udiffsum = np.sum(udiff,axis=(0,1,2))
        udiffsum[ff] = 1000

        bb = biases[ff]
        bbr = np.tile(bb, nfeat)
        bdiff = np.absolute(biases-bb)
        
        cdiffsum = bdiff + udiffsum 
        corr_featid[ff] = np.argmin(cdiffsum)
        #print(np.min(udiffsum))

    return corr_featid

def main(args):

    pdb.set_trace()
    #modelDir = '/trn_dir/models/model1'
    #model_path = '/trn_dir/models/model_multi_cl20_relu6/model-000075'  
    model_path = '/trn_dir/models/model_multi_cl20_preprocess/model-000060'  

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
        
        

        #vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(my_vgg_relu6_fc.vgg_arg_scope(weight_decay=args.weight_decay)):
            logits, endpoints_dict = my_vgg_relu6_fc.vgg_16(images, num_classes=num_classes, is_training=is_training,dropout_keep_prob=args.dropout_keep_prob)


        
        main_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_16')
        #target_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_16_t')
        #variables_to_restore = tf.contrib.framework.get_variables_to_restore()#exclude=['vgg_16/fc8'])

        vgg_trainable_names = tf.trainable_variables(scope='vgg_16') 

        variables_to_restore = tf.contrib.framework.get_variables_to_restore(vgg16_variable_names)
        var_load_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

 



        global_init=tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        pdb.set_trace()
        #sess.run(new_var_init)
        sess.run(global_init)
        var_load_fn(sess)  # load the pretrained weights
 
        # print name, shape of each trainable variable
        savename = os.path.basename(model_path)+'_featcorr.json'
        vgg_variable_values={}
        feat_corr={}
        trainable_values = sess.run(vgg_trainable_names)
        for nm, v in zip(vgg_trainable_names[:44], trainable_values[:44]):
            print('name = ',nm.name, ' shape = ',v.shape)
            if 'vgg_16_tcorr' not in nm.name:
                vgg_variable_values[nm.name] = np.array(v)    
 
        for vname in vgg16_variable_names:
            print(vname)
            wname = vname+'/weights:0'
            bname = vname+'/biases:0'
            feat_corr[vname] = compute_feature_correlation(vgg_variable_values[wname], vgg_variable_values[bname]).tolist()        


    
        with open(savename,'w') as fid:
            json.dump(feat_corr,fid)
            fid.close()
        #feat_corr[nm.name] = compute_feature_correlation(vgg_variable_values[nm.name])
       
        #feat_corr = compute_feature_correlation(vgg_variable_values['vgg_16/conv3/conv3_2/weights:0'])

        pp=1     


if __name__ == '__main__':
    args = parser.parse_args()

    main(args)
