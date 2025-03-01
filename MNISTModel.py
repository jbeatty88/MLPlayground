import math

import tensorflow as tf

class MNISTModel(object):
    # Model Init
    def __init__(self, input_dim, output_size):
        self.input_dim = input_dim
        self.output_size = output_size

    # Apply Dropout to Final <Fully-Connected> Layer (avoid co-adaptation)
    def apply_dropout(self, dense, is_training):
        # Dropout rate <rd> : 0.4
        dropout = tf.python.layers.dropout(dense, rate=0.4, training=is_training)
        return dropout

    # Apply Fully-Connected Layer
    def create_fc(self, pool2):
        # Get the height, width, and # of channels if pool2 before flatten
        hwc = pool2.shape.as_list()[1:]
        # Get the pool2 data feature count
        # flattened_size = math.prod(hwc)
        flattened_size = hwc[0] * hwc[1] * hwc[2]
        pool2_flat = tf.python.reshape(pool2, [-1, flattened_size])
        dense = tf.python.layers.dense(pool2_flat, 1024, activation=tf.nn.relu, name='dense')
        return dense

    def get_logits(self, dropout):
        logits = tf.python.layers.dense(dropout, self.output_size, name='logits')
        return logits

    # CNN Layers
    def model_layers(self, inputs, is_training):
        # Convert input data into NHWC format
        reshaped_inputs = tf.reshape(inputs, (-1, self.input_dim, self.input_dim, 1))
        # Convolutional Layer 1
        conv1 = tf.python.layers.conv2d(reshaped_inputs, 32, [5,5], padding='same', activation=tf.nn.relu, name='conv1')
        # Pooling Layer 1
        pool1 = tf.python.layers.max_pooling2d(conv1, [2,2], 2, name='pool1')
        # Convolutional Layer 2
        conv2 = tf.python.layers.conv2d(pool1, 64, [5, 5], padding='same', activation=tf.nn.relu, name='conv2')
        # Pooling Layer 2
        pool2 = tf.python.layers.max_pooling2d(conv2, [2, 2], 2, name='pool2')

def run_model_setup(self, inputs, labels, is_training):
    logits = self.model_layers(inputs, is_training)

    # convert logits to probabilities with softmax activation
    self.probs = tf.nn.softmax(logits, name='probs')
    # round probabilities
    self.predictions = tf.argmax(
        self.probs, axis=-1, name='predictions')
    class_labels = tf.argmax(labels, axis=-1)
    # find which predictions were correct
    is_correct = tf.equal(
        self.predictions, class_labels)
    is_correct_float = tf.cast(
        is_correct,
        tf.float32)
    # compute ratio of correct to incorrect predictions
    self.accuracy = tf.reduce_mean(
        is_correct_float)
    # train model
    if self.is_training:
        labels_float = tf.cast(
            labels, tf.float32)
        # compute the loss using cross_entropy
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_float,
            logits=logits)
        self.loss = tf.reduce_mean(
            cross_entropy)
        # use adam to train model
        adam = tf.python.train.AdamOptimizer()
        self.train_op = adam.minimize(
            self.loss, global_step=self.global_step)