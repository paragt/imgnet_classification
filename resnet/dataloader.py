import tensorflow as tf
import os,sys
import json


VGG_MEAN = [123.68, 116.78, 103.94]

'''
def list_images(directory):
    """
    Get all the images and labels in directory/label/*.jpg
    """
    labels = os.listdir(directory)
    # Sort the labels so that training and validation get them in the same order
    labels.sort()

    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            files_and_labels.append((os.path.join(directory, label, f), label))

    filenames, labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    unique_labels = list(set(labels))

    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i

    labels = [label_to_int[l] for l in labels]

    return filenames, labels

'''
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


class mydataloader_train:

    def __init__(self,train_dir,labelmap_file, batch_size=30,num_workers=4):

        batch_size=30

        train_filenames, train_labels = list_images(train_dir,labelmap_file)

        self.num_classes = len(set(train_labels))
        self.num_trn_examples= len(train_labels)
 
        # Training dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
        train_dataset = train_dataset.map(self._parse_function,num_parallel_calls=num_workers).prefetch(batch_size)
        train_dataset = train_dataset.map(self.training_preprocess,num_parallel_calls=num_workers).prefetch(batch_size)
        train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
        self.batched_train_dataset = train_dataset.batch(batch_size)



    # Standard preprocessing for VGG on ImageNet taken from here:
    # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py
    # Also see the VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf

    # Preprocessing (for both training and validation):
    # (1) Decode the image from jpg format
    # (2) Resize the image so its smaller side is 256 pixels long
    def _parse_function(self,filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
        image = tf.cast(image_decoded, tf.float32)

        smallest_side = 256.0
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        height = tf.to_float(height)
        width = tf.to_float(width)

        scale = tf.cond(tf.greater(height, width),lambda: smallest_side / width,lambda: smallest_side / height)
        new_height = tf.to_int32(height * scale)
        new_width = tf.to_int32(width * scale)

        resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
        return resized_image, label

    # Preprocessing (for training)
    # (3) Take a random 224x224 crop to the scaled image
    # (4) Horizontally flip the image with probability 1/2
    # (5) Substract the per color mean `VGG_MEAN`
    # Note: we don't normalize the data here, as VGG was trained without normalization
    def training_preprocess(self,image, label):
        crop_image = tf.random_crop(image, [224, 224, 3])                       # (3)
        flip_image = tf.image.random_flip_left_right(crop_image)                # (4)

        means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
        centered_image = flip_image - means                                     # (5)

        return centered_image, label





class mydataloader_val:

    def __init__(self,val_dir,labelmap_file,batch_size=30,num_workers=4):

        batch_size=30

        val_filenames, val_labels = list_images(val_dir,labelmap_file)

        self.num_classes = len(set(val_labels))
        self.num_trn_examples= len(val_labels)
 
        # Validation dataset
        val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
        val_dataset = val_dataset.map(self._parse_function,num_parallel_calls=num_workers).prefetch(batch_size)
        val_dataset = val_dataset.map(self.val_preprocess,num_parallel_calls=num_workers).prefetch(batch_size)
        self.batched_val_dataset = val_dataset.batch(batch_size)




    # Standard preprocessing for VGG on ImageNet taken from here:
    # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py
    # Also see the VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf

    # Preprocessing (for both training and validation):
    # (1) Decode the image from jpg format
    # (2) Resize the image so its smaller side is 256 pixels long
    def _parse_function(self,filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
        image = tf.cast(image_decoded, tf.float32)

        smallest_side = 256.0
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        height = tf.to_float(height)
        width = tf.to_float(width)

        scale = tf.cond(tf.greater(height, width),lambda: smallest_side / width,lambda: smallest_side / height)
        new_height = tf.to_int32(height * scale)
        new_width = tf.to_int32(width * scale)

        resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
        return resized_image, label


    # Preprocessing (for validation)
    # (3) Take a central 224x224 crop to the scaled image
    # (4) Substract the per color mean `VGG_MEAN`
    # Note: we don't normalize the data here, as VGG was trained without normalization
    def val_preprocess(self, image, label):
        crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)

        means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
        centered_image = crop_image - means                                     # (4)

        return centered_image, label

