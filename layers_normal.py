# pylint: disable=C0103,R0204,C0111
"""This module defines basic layers for tensorflow
Two principles are repected:
    - after initialisation, the normalisation in maintained
"""
import math
import tensorflow as tf


def relu(x):
    return (tf.nn.relu(x) - 0.3989422804014327) * 1.712858550449663


def scaleandshift(x, a0=1, b0=0):
    f = x.get_shape().as_list()[-1]
    with tf.name_scope("scaleandshift"):
        a = tf.Variable(tf.constant(a0, dtype=tf.float32, shape=[f]), name="g")
        b = tf.Variable(tf.constant(b0, dtype=tf.float32, shape=[f]), name="b")
        tf.summary.histogram("scale", a)
        tf.summary.histogram("shift", b)
        return a * x + b


def fullyconnected(x, f_out=None, activation=relu, name='fc'):
    f_in = x.get_shape().as_list()[1]
    if f_out is None:
        f_out = f_in

    with tf.name_scope("{}-{}-{}".format(name, f_in, f_out)):
        W0 = tf.random_normal([f_in, f_out])
        W0 = W0 / math.sqrt(f_in)
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
def convolution(x, f_out=None, s=1, w=3, padding='VALID', activation=relu, name='conv'):
    f_in = x.get_shape().as_list()[3]

    if f_out is None:
        f_out = f_in

    with tf.name_scope("{}-{}-{}".format(name, f_in, f_out)):
        F0 = tf.random_normal([w, w, f_in, f_out])
        F0 = F0 / math.sqrt(w * w * f_in)
        F = tf.Variable(F0, name="F")
        tf.summary.histogram("filter", F)
        x = tf.nn.conv2d(x, F, [1, s, s, 1], padding)

        b = tf.Variable(tf.constant(0.0, shape=[f_out]), name="b")
        tf.summary.histogram("bias", b)
        x = x + b

        if activation is not None:
            x = activation(x)
        return x


def max_pool(x):
    with tf.name_scope("max_pool"):
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return scaleandshift(x, 0.83, -0.92)


def batch_normalization(x, acc):
    shape = x.get_shape().as_list()
    f_in = shape[-1]

    def moments(x, axes):
        m = tf.reduce_mean(x, axes)
        v = tf.reduce_mean(tf.square(x), axes) - tf.square(m)
        return m, v

    acc = tf.convert_to_tensor(acc, dtype=tf.float32, name="accumulator")

    with tf.name_scope("bn-{}".format(f_in)):
        m, v = moments(x, axes=list(range(len(shape) - 1)))

        acc_m = tf.Variable(tf.constant(0.0, shape=[f_in]), trainable=False, name="acc_m")
        acc_v = tf.Variable(tf.constant(1.0, shape=[f_in]), trainable=False, name="acc_v")
        tf.summary.histogram("acc_m", acc_m)
        tf.summary.histogram("acc_v", acc_v)

        m = tf.assign(acc_m, (1.0 - acc) * acc_m + acc * m)
        v = tf.assign(acc_v, (1.0 - acc) * acc_v + acc * v)
        m.set_shape([f_in]) # pylint: disable=E1101
        v.set_shape([f_in]) # pylint: disable=E1101

        beta = tf.Variable(tf.constant(0.0, shape=[f_in]))
        gamma = tf.Variable(tf.constant(1.0, shape=[f_in]))
        tf.summary.histogram("beta", beta)
        tf.summary.histogram("gamma", gamma)
        return tf.nn.batch_normalization(x, m, v, beta, gamma, 1e-3)
