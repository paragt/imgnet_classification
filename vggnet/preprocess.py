import tensorflow as tf
import os,sys



VGG_MEAN = [123.68, 116.78, 103.94]


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(tf.equal(tf.rank(image), 3),['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(tf.logical_and(tf.greater_equal(original_shape[0], crop_height),tf.greater_equal(original_shape[1], crop_width)),['Crop size greater than the image size.'])
     
    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)

    return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
    if not image_list:
        raise ValueError('Empty image_list.')

    # Compute the rank assertions.
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(tf.equal(image_rank, 3),['Wrong rank for tensor  %s [expected] [actual]',image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(image_list[0])
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(tf.logical_and(tf.greater_equal(image_height, crop_height),tf.greater_equal(image_width, crop_width)),['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]
 
    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
            shape = tf.shape(image)
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(tf.equal(height, image_height),['Wrong height for tensor %s [expected][actual]',image.name, height, image_height])
        width_assert = tf.Assert(tf.equal(width, image_width),['Wrong width for tensor %s [expected][actual]',
image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

    # Create a random bounding box.
    #
    # Use tf.random_uniform and not numpy.random.rand as doing the former would
    # generate random numbers at graph eval time, unlike the latter which
    # generates random numbers at graph definition time.
    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
    with tf.control_dependencies(asserts):
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    offset_height = tf.random_uniform([], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)

    return [_crop(image, offset_height, offset_width, crop_height, crop_width) for image in image_list]


def _central_crop(image_list, crop_height, crop_width):
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = (image_height - crop_height) / 2
        offset_width = (image_width - crop_width) / 2

        outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
    return outputs


def _smallest_size_at_least(height, width, smallest_side):

    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
    new_height = tf.to_int32(tf.rint(height * scale))
    new_width = tf.to_int32(tf.rint(width * scale))

    return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):

    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])

    return resized_image


class preprocessor:

    def __init__(self):
        self.output_height = 224
        self.output_width = 224
        self.resize_side_min = 256
        self.resize_side_max = 512

    def _parse_function(self,filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
        return image_decoded


    def training_preprocess(self, image):

        resize_side = tf.random_uniform([], minval= self.resize_side_min, maxval= self.resize_side_max+1, dtype=tf.int32)

        resized_image = _aspect_preserving_resize(image, resize_side)
        cropped_image = _random_crop([resized_image], self.output_height, self.output_width)[0]
        cropped_image.set_shape([self.output_height, self.output_width, 3])
        float_image   = tf.to_float(cropped_image)
        flipped_image = tf.image.random_flip_left_right(float_image)

        #crop_image = tf.random_crop(image, [224, 224, 3])                       # (3)
        #flip_image = tf.image.random_flip_left_right(crop_image)                # (4)

        means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
        centered_image = flipped_image - means                                     # (5)

        return centered_image


    def val_preprocess(self, image):

        resized_image = _aspect_preserving_resize(image, self.resize_side_min)
        cropped_image = _central_crop([resized_image], self.output_height, self.output_width)[0]
        cropped_image.set_shape([self.output_height, self.output_width, 3])
        float_image = tf.to_float(cropped_image)

        #crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)

        means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
        centered_image = float_image - means                                     # (4)

        return centered_image

