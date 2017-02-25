# pylint: disable=C0103,R0204
"""This module defines equivariant layers for tensorflow under the dihedral group
Two principles are repected:
    - after initialisation, the normalisation in maintained
"""
import math
import tensorflow as tf
import numpy as np


def relu(x):
    """ReLU compatible with normalization propagation"""
    return (tf.nn.relu(x) - 0.3989422804014327) * 1.712858550449663


def scaleandshift(x, a0=1, b0=0, input_repr='regular'):
    """Scale and shift a tensor en keep its representation"""
    assert input_repr == 'regular' or input_repr == 'invariant'

    f = x.get_shape().as_list()[-1]
    with tf.name_scope("scaleandshift"):
        if input_repr == 'invariant':
            a = tf.Variable(tf.constant(a0, dtype=tf.float32, shape=[f]), name="g")
            b = tf.Variable(tf.constant(b0, dtype=tf.float32, shape=[f]), name="b")
        if input_repr == 'regular':
            assert f % 8 == 0
            a = tf.tile(tf.Variable(tf.constant(
                a0, dtype=tf.float32, shape=[f // 8]), name="g"), [8])
            b = tf.tile(tf.Variable(tf.constant(
                b0, dtype=tf.float32, shape=[f // 8]), name="b"), [8])
        tf.summary.histogram("scale", a)
        tf.summary.histogram("shift", b)
        return a * x + b


def fullyconnected(x, f_out=None, output_repr='regular', activation=relu, name='fc'):
    """Fully connect a tensor which transforms with the regular representation
    with an output tensor which transforms either with the regular or invariant representation"""
    assert output_repr == 'regular' or output_repr == 'invariant'

    f_in = x.get_shape().as_list()[1]
    assert f_in % 8 == 0
    if f_out is None:
        f_out = f_in

    if output_repr == 'regular':
        assert f_out % 8 == 0

        with tf.name_scope("{}-8x{}-8x{}".format(name, f_in // 8, f_out // 8)):
            W0 = tf.random_normal([f_in, f_out // 8])
            W0 = W0 / math.sqrt(f_in)
            W = tf.Variable(W0, name="W")
            tf.summary.histogram("weights", W)
            W = tf.split(W, 8, 0)

            mt = np.array([
                [0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6],
                [2, 3, 0, 1, 6, 7, 4, 5], [3, 2, 1, 0, 7, 6, 5, 4],
                [4, 6, 5, 7, 0, 2, 1, 3], [5, 7, 4, 6, 1, 3, 0, 2],
                [6, 4, 7, 5, 2, 0, 3, 1], [7, 5, 6, 4, 3, 1, 2, 0]])
            # tau[mt[a,b]] = tau[a] o tau[b]

            iv = np.array([0, 1, 2, 3, 4, 6, 5, 7])
            # tau[iv[a]] is the inverse of tau[a]

            W = tf.concat([  # merge 8 part of the output
                tf.concat([  # merge 8 part of the input
                    W[mt[iv[j], i]]
                    for i in range(8)], 0)
                for j in range(8)], 1)

            x = tf.matmul(x, W)

            b = tf.Variable(tf.constant(0.0, shape=[f_out // 8]), name="b")
            tf.summary.histogram("bias", b)
            x = x + tf.tile(b, [8])

            if activation is not None:
                x = activation(x)
            return x

    if output_repr == 'invariant':
        with tf.name_scope("{}-8x{}-{}".format(name, f_in // 8, f_out)):
            W0 = tf.random_normal([f_in // 8, f_out])
            W0 = W0 / math.sqrt(f_in)
            W = tf.Variable(W0, name="W")
            tf.summary.histogram("weights", W)
            W = tf.tile(W, [8, 1])
            x = tf.matmul(x, W)

            b = tf.Variable(tf.constant(0.0, shape=[f_out]), name="b")
            tf.summary.histogram("bias", b)
            x = x + b

            if activation is not None:
                x = activation(x)
            return x

# pylint: disable=R0913, R0915
def convolution(x, f_out=None, s=1, w=3,
                input_repr='regular', output_repr='regular', padding='VALID',
                activation=relu, name='conv'):
    """The input and output tensor must tranform with the
     defining x (invariant / regular) representation
    where the x represent the tensor product between
    the spacial componant and the channels componant."""
    assert input_repr == 'regular' or input_repr == 'invariant'
    assert output_repr == 'regular' or output_repr == 'invariant'

    f_in = x.get_shape().as_list()[3]

    # pylint: disable=C0111
    def filters(d_in, d_out, n_mul=None):
        F0 = tf.random_normal([w, w, d_in, d_out])
        if n_mul is None:
            n_mul = w * w * d_in
        F0 = F0 / math.sqrt(n_mul)

        F = tf.Variable(F0, name="F")
        tf.summary.histogram("filter", F)

        if w > 1:
            Fs = [None] * 8
            Fs[0] = F  # tau[0]
            Fs[1] = tf.reverse(F, [1])  # tau[1]
            Fs[2] = tf.reverse(F, [0])  # tau[2]
            Fs[3] = tf.reverse(F, [0, 1])  # tau[3]
            Fs[4] = tf.transpose(F, [1, 0, 2, 3])  # tau[4]
            Fs[5] = tf.reverse(Fs[4], [1])  # tau[5]
            Fs[6] = tf.reverse(Fs[4], [0])  # tau[6]
            Fs[7] = tf.reverse(Fs[4], [0, 1])  # tau[7]
            # Fs[j] = tau[j] F
        else:
            Fs = [F] * 8

        return Fs

    if input_repr == 'regular' and output_repr == 'regular':
        if f_out is None:
            f_out = f_in
        assert f_in % 8 == 0 and f_out % 8 == 0

        with tf.name_scope("{}-8x{}-8x{}".format(name, f_in // 8, f_out // 8)):
            Fs = [tf.split(F, 8, 2) for F in filters(f_in, f_out // 8)]
            # Fs[j][i] = tau[j] F_i

            mt = np.array([
                [0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6],
                [2, 3, 0, 1, 6, 7, 4, 5], [3, 2, 1, 0, 7, 6, 5, 4],
                [4, 6, 5, 7, 0, 2, 1, 3], [5, 7, 4, 6, 1, 3, 0, 2],
                [6, 4, 7, 5, 2, 0, 3, 1], [7, 5, 6, 4, 3, 1, 2, 0]])
            # tau[mt[a,b]] = tau[a] o tau[b]

            iv = np.array([0, 1, 2, 3, 4, 6, 5, 7])
            # tau[iv[a]] is the inverse of tau[a]

            F = tf.concat([  # merge 8 part of the output
                tf.concat([  # merge 8 part of the input
                    Fs[j][mt[iv[j], i]]
                    for i in range(8)], 2)
                for j in range(8)], 3)

            # y = Conv(x, W)
            x = tf.nn.conv2d(x, F, [1, s, s, 1], padding)

            b = tf.Variable(tf.constant(0.0, shape=[f_out // 8]), name="b")
            tf.summary.histogram("bias", b)
            x = x + tf.tile(b, [8])

            if activation is not None:
                x = activation(x)
            return x

    if input_repr == 'invariant' and output_repr == 'regular':
        if f_out is None:
            f_out = 8 * f_in
        assert f_out % 8 == 0

        with tf.name_scope("{}-{}-8x{}".format(name, f_in, f_out // 8)):
            F = tf.concat(filters(f_in, f_out // 8), 3)
            x = tf.nn.conv2d(x, F, [1, s, s, 1], padding)

            b = tf.Variable(tf.constant(0.0, shape=[f_out // 8]), name="b")
            tf.summary.histogram("bias", b)
            x = x + tf.tile(b, [8])

            if activation is not None:
                x = activation(x)
            return x

    if input_repr == 'regular' and output_repr == 'invariant':
        if f_out is None:
            f_out = f_in // 8
        assert f_in % 8 == 0

        with tf.name_scope("{}-8x{}-{}".format(name, f_in // 8, f_out)):
            F = tf.concat(filters(f_in // 8, f_out, n_mul=w * w * f_in), 2)
            x = tf.nn.conv2d(x, F, [1, s, s, 1], padding)

            b = tf.Variable(tf.constant(0.0, shape=[f_out]), name="b")
            tf.summary.histogram("bias", b)
            x = x + b

            if activation is not None:
                x = activation(x)
            return x


def max_pool(x, input_repr='regular'):
    """max pool compatible with normalization propagation"""
    with tf.name_scope("max_pool"):
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return scaleandshift(x, 0.77, -1, input_repr=input_repr)


def pool(x):
    """Take a tensor which transforms with the regular representation
    and output a tensor which transforms with the invariant representation"""
    shape = x.get_shape().as_list()
    f_in = shape[-1]
    assert f_in % 8 == 0

    with tf.name_scope("pool-8x{}".format(f_in // 8)):
        xs = tf.split(x, 8, len(shape) - 1)
        return tf.add_n(xs)


def concat(xs):
    """Concatenate a list of tensors which transform with the
    regular representation into one"""
    ss = [x.get_shape().as_list() for x in xs]
    assert all([s[-1] % 8 == 0 for s in ss])
    assert all([len(s) == len(ss[0]) for s in ss])
    assert all([s[:-1] == ss[0][:-1] for s in ss])
    f_dim = len(ss[0]) - 1
    xs = [tf.split(xs[i], 8, f_dim) for i in range(len(xs))]
    xs = [tf.concat(ys, f_dim) for ys in zip(*xs)]
    return tf.concat(xs, f_dim)

# pylint: disable=E1101
def batch_normalization(x, acc, input_repr='regular'):
    """Perform the amortized batch normalization, the mean and variance
    are accumulated according to the parameter acc.
    Typically acc = exp(- training steps / some number).
    The representation of the output is the same as which of the input"""
    assert input_repr == 'regular' or input_repr == 'invariant'

    shape = x.get_shape().as_list()
    f_in = shape[-1]

    # pylint: disable=C0111
    def moments(x, axes):
        m = tf.reduce_mean(x, axes)
        v = tf.reduce_mean(tf.square(x), axes) - tf.square(m)
        return m, v

    acc = tf.convert_to_tensor(acc, dtype=tf.float32, name="accumulator")

    if input_repr == 'invariant':
        with tf.name_scope("bn-{}".format(f_in)):
            m, v = moments(x, axes=list(range(len(shape) - 1)))

            acc_m = tf.Variable(tf.constant(0.0, shape=[f_in]), trainable=False, name="acc_m")
            acc_v = tf.Variable(tf.constant(1.0, shape=[f_in]), trainable=False, name="acc_v")
            tf.summary.histogram("acc_m", acc_m)
            tf.summary.histogram("acc_v", acc_v)

            m = tf.assign(acc_m, (1.0 - acc) * acc_m + acc * m)
            v = tf.assign(acc_v, (1.0 - acc) * acc_v + acc * v)
            m.set_shape([f_in])
            v.set_shape([f_in])

            beta = tf.Variable(tf.constant(0.0, shape=[f_in]))
            gamma = tf.Variable(tf.constant(1.0, shape=[f_in]))
            tf.summary.histogram("beta", beta)
            tf.summary.histogram("gamma", gamma)
            return tf.nn.batch_normalization(x, m, v, beta, gamma, 1e-3)
    if input_repr == 'regular':
        assert f_in % 8 == 0
        with tf.name_scope("bn-8x{}".format(f_in // 8)):
            m, v = moments(pool(x), axes=list(range(len(shape) - 1)))
            m = m / 8.0
            v = v / 8.0

            acc_m = tf.Variable(tf.constant(0.0, shape=[f_in // 8]), trainable=False, name="acc_m")
            acc_v = tf.Variable(tf.constant(1.0, shape=[f_in // 8]), trainable=False, name="acc_v")
            tf.summary.histogram("acc_m", acc_m)
            tf.summary.histogram("acc_v", acc_v)

            new_acc_m = tf.assign(acc_m, (1.0 - acc) * acc_m + acc * m)
            new_acc_v = tf.assign(acc_v, (1.0 - acc) * acc_v + acc * v)

            m = tf.tile(new_acc_m, [8])
            v = tf.tile(new_acc_v, [8])
            m.set_shape([f_in])
            v.set_shape([f_in])

            beta = tf.tile(tf.Variable(tf.constant(0.0, shape=[f_in // 8])), [8])
            gamma = tf.tile(tf.Variable(tf.constant(1.0, shape=[f_in // 8])), [8])
            tf.summary.histogram("beta", beta)
            tf.summary.histogram("gamma", gamma)
            return tf.nn.batch_normalization(x, m, v, beta, gamma, 1e-3)
