import tensorflow as tf
import os,sys



VGG_MEAN = [123.68, 116.78, 103.94]

class preprocessor:

    def __int__(self):
        pass

    def _parse_function(self,filename):
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
        return resized_image


    def training_preprocess(self,image):
        crop_image = tf.random_crop(image, [224, 224, 3])                       # (3)
        flip_image = tf.image.random_flip_left_right(crop_image)                # (4)

        means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
        centered_image = flip_image - means                                     # (5)

        return centered_image


    def val_preprocess(self, image):
        crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)

        means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
        centered_image = crop_image - means                                     # (4)

        return centered_image

