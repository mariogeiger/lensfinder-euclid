# pylint: disable=C0103,R0204,C0111
"""This module defines basic layers for tensorflow"""
import tensorflow as tf

def relu(x):
    x = (tf.nn.relu(x) - 0.3989422804014327) * 1.712858550449663
    tf.summary.histogram("relu", x)
    return x


def scaleandshift(x):
    f = x.get_shape().as_list()[-1]
    with tf.name_scope("scaleandshift"):
        a = tf.Variable(tf.constant(1.0, shape=[f]), name="g")
        b = tf.Variable(tf.constant(0.0, shape=[f]), name="b")
        tf.summary.histogram("scale", a)
        tf.summary.histogram("shift", b)
        return a * x + b


def fullyconnected(x, f_out=None, activation=relu):
    f_in = x.get_shape().as_list()[1]
    if f_out is None:
        f_out = f_in

    with tf.name_scope("fc_{}_{}".format(f_in, f_out)):
        W0 = tf.nn.l2_normalize(tf.random_normal([f_in, f_out]), [0])
        W = tf.Variable(W0, name="W")
        tf.summary.histogram("weights", W)
        x = tf.matmul(x, W)

        b = tf.Variable(tf.constant(0.0, shape=[f_out]), name="b")
        tf.summary.histogram("bias", b)
        x = x + b

        if activation is not None:
            x = activation(x)
        return x

#pylint: disable=R0913
def convolution(x, f_out=None, s=1, w=3, padding='VALID', activation=relu):
    f_in = x.get_shape().as_list()[3]

    if f_out is None:
        f_out = f_in

    with tf.name_scope("conv_{}_{}".format(f_in, f_out)):
        F0 = tf.nn.l2_normalize(tf.random_normal([w, w, f_in, f_out]), [0, 1, 2])
        F = tf.Variable(F0, name="F")
        tf.summary.histogram("filters", F)
        x = tf.nn.conv2d(x, F, [1, s, s, 1], padding)

        b = tf.Variable(tf.constant(0.0, shape=[f_out]), name="b")
        tf.summary.histogram("bias", b)
        x = x + b

        if activation is not None:
            x = activation(x)
        return x


def max_pool(x):
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return (x - 1.1) / 1.2


def batch_normalization(x, acc):
    shape = x.get_shape().as_list()
    f_in = shape[-1]

    def moments(x, axes):
        m = tf.reduce_mean(x, axes)
        v = tf.reduce_mean(tf.square(x), axes) - tf.square(m)
        return m, v

    acc = tf.convert_to_tensor(acc, dtype=tf.float32, name="accumulator")

    with tf.name_scope("bn_{}".format(f_in)):
        m, v = moments(x, axes=list(range(len(shape) - 1)))

        acc_m = tf.Variable(tf.constant(0.0, shape=[f_in]), trainable=False, name="acc_m")
        acc_v = tf.Variable(tf.constant(1.0, shape=[f_in]), trainable=False, name="acc_v")

        m = tf.assign(acc_m, (1.0 - acc) * acc_m + acc * m)
        v = tf.assign(acc_v, (1.0 - acc) * acc_v + acc * v)
        m.set_shape([f_in]) # pylint: disable=E1101
        v.set_shape([f_in]) # pylint: disable=E1101

        beta = tf.Variable(tf.constant(0.0, shape=[f_in]))
        gamma = tf.Variable(tf.constant(1.0, shape=[f_in]))
        return tf.nn.batch_normalization(x, m, v, beta, gamma, 1e-3)
