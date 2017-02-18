# pylint: disable=C,R,no-member
import tensorflow as tf
import numpy as np
import dihedral as nn

def res_layer(x, f_out=None, w=3, n=1, input_repr='regular', output_repr='regular'):
    f_in = x.get_shape().as_list()[3]
    assert w % 2 == 1
    if f_out is None:
        f_out = f_in

    with tf.name_scope("res_layer"):
        b = (w // 2) * n
        s = x[:, b:-b, b:-b, :] # VALID padding
        if f_in != f_out or input_repr != output_repr:
            s = nn.convolution(s, f_out, w=1, input_repr=input_repr, output_repr=output_repr, name="shortcut")

        if n == 1:
            x = nn.convolution(x, f_out, w=w, input_repr=input_repr, output_repr=output_repr)
        else:
            x = nn.convolution(x, f_out, w=w, input_repr=input_repr, output_repr='regular')
            for _ in range(n - 2):
                x = nn.convolution(x, f_out, w=w, input_repr='regular', output_repr='regular')
            x = nn.convolution(x, f_out, w=w, input_repr='regular', output_repr=output_repr)

        x = 0.7071067811865475 * (s + x)
        return x

class CNN:
    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        self.tfx = None
        self.tfy = None
        self.tfp = None
        self.xent = None
        self.tftrain_step = None
        self.tfkp = None
        self.tfacc = None
        self.train_counter = 0
        self.test = None
        self.embedding_input = None


    def NN(self, x):
        assert x.get_shape().as_list()[:3] == [None, 101, 101]
        x = nn.convolution(x, 8*4, w=2, input_repr='invariant') # 100

        ########################################################################
        assert x.get_shape().as_list() == [None, 100, 100, 8*4]
        x = res_layer(x, n=2) # 96
        x = nn.batch_normalization(x, self.tfacc)
        x = res_layer(x, n=2) # 92
        x = nn.max_pool(x)

        ########################################################################
        assert x.get_shape().as_list() == [None, 46, 46, 8*4]
        x = res_layer(x, 8*8, n=3) # 40
        x = nn.batch_normalization(x, self.tfacc)
        x = res_layer(x, n=2) # 36
        x = nn.max_pool(x)

        ########################################################################
        assert x.get_shape().as_list() == [None, 18, 18, 8*8]
        x = res_layer(x, 8*16, n=2) # 14
        x = nn.batch_normalization(x, self.tfacc)

        x = res_layer(x, 8*23, n=2) # 10
        x = nn.batch_normalization(x, self.tfacc)

        x = res_layer(x, 8*32, n=3) # 4
        x = nn.batch_normalization(x, self.tfacc)

        ########################################################################
        assert x.get_shape().as_list() == [None, 4, 4, 8*32]
        x = nn.convolution(x, 8*128, w=4)
        x = tf.nn.dropout(x, self.tfkp)

        ########################################################################
        assert x.get_shape().as_list() == [None, 1, 1, 8*128]
        x = tf.reshape(x, [-1, x.get_shape().as_list()[-1]])

        self.embedding_input = x

        x = nn.fullyconnected(x, 8*256)
        x = tf.nn.dropout(x, self.tfkp)

        x = nn.fullyconnected(x, 8*256)
        x = nn.batch_normalization(x, self.tfacc)
        self.test = x

        x = nn.fullyconnected(x, 1, output_repr='invariant', activation=None)
        return x

    ########################################################################
    def create_architecture(self, bands):
        self.tfkp = tf.placeholder_with_default(tf.constant(1.0, tf.float32), [])
        self.tfacc = tf.placeholder_with_default(tf.constant(0.0, tf.float32), [])
        x = self.tfx = tf.placeholder(tf.float32, [None, 101, 101, bands])
        # mean = 0 and std = 1

        if bands == 1:
            tf.summary.image("input", x, 3)
        else:
            tf.summary.image("input", x[:,:,:,:3], 3)

        with tf.name_scope("nn"):
            x = self.NN(x)

        ########################################################################
        assert x.get_shape().as_list() == [None, 1]
        self.tfp = tf.nn.sigmoid(tf.reshape(x, [-1]))

        with tf.name_scope("xent"):
            self.tfy = tf.placeholder(tf.float32, [None])
            xent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=tf.reshape(self.tfy, [-1, 1]))
            # [None, 1]
            self.xent = tf.reduce_mean(xent)
            tf.summary.scalar("xent", self.xent)

        with tf.name_scope("train"):
            self.tftrain_step = tf.train.AdamOptimizer(1e-4).minimize(self.xent)

    @staticmethod
    def split_test_train(path):
        # extract the files names and split them into test and train set
        import os
        files = ['{}/{}'.format(path, f) for f in sorted(os.listdir(path))]
        return files[:3000], files[3000:]

    @staticmethod
    def load(files):
        # load some files into numpy array ready to be eaten by tensorflow
        xs = np.stack([np.load(f)['image'] for f in files])
        return CNN.prepare(xs)

    @staticmethod
    def prepare(images):
        images[images == 100] = 0.0
        if images.shape[-1] == 1:
            images = (images - 4.337e-13) / 5.504e-12
        elif images.shape[-1] == 4:
            images = (images - 1.685e-12) / 5.122e-11
        else:
            print("No statistics to prepare this kind of data")
        return images

    @staticmethod
    def batch(files, labels):
        # pick randomly some files, load them and make augmentation
        id0 = np.where(labels == 0)[0]
        id1 = np.where(labels == 1)[0]

        k = 15
        idn = np.random.choice(id0, k, replace=False)
        idp = np.random.choice(id1, k, replace=False)

        xs = CNN.load([files[i] for i in idp] + [files[i] for i in idn])
        ys = np.concatenate((labels[idp], labels[idn]))

        for i in range(len(xs)):
            s = np.random.uniform(0.8, 1.2)
            u = np.random.uniform(-0.1, 0.1)
            xs[i] = xs[i] * s + u

        return xs, ys

    def train(self, session, xs, ys, options=None, run_metadata=None, tensors=None):
        if tensors is None:
            tensors = []

        acc = 0.6 ** (self.train_counter / 1000.0)
        kp = 0.5 + 0.5 * 0.5 ** (self.train_counter / 2000.0)

        output = session.run([self.tftrain_step, self.xent] + tensors,
            feed_dict={self.tfx: xs, self.tfy: ys, self.tfkp: kp, self.tfacc: acc},
            options=options, run_metadata=run_metadata)

        self.train_counter += 1
        return output[1], output[2:]

    def predict(self, session, images):
        return session.run(self.tfp, feed_dict={self.tfx: images})

    def predict_xentropy(self, session, images, labels):
        return session.run([self.tfp, self.xent], feed_dict={self.tfx: images, self.tfy: labels})
