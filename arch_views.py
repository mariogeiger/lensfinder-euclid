# pylint: disable=C,R,no-member
import tensorflow as tf
import numpy as np
import layers_normal as nn


def dihedral(x, i):
    if len(x.shape) == 3:
        if i & 4:
            y = np.transpose(x, (1, 0, 2))
        else:
            y = x.copy()

        if i&3 == 0:
            return y
        if i&3 == 1:
            return y[:, ::-1]
        if i&3 == 2:
            return y[::-1, :]
        if i&3 == 3:
            return y[::-1, ::-1]

    if len(x.shape) == 4:
        if i & 4:
            y = np.transpose(x, (0, 2, 1, 3))
        else:
            y = x.copy()

        if i&3 == 0:
            return y
        if i&3 == 1:
            return y[:, :, ::-1]
        if i&3 == 2:
            return y[:, ::-1, :]
        if i&3 == 3:
            return y[:, ::-1, ::-1]

def summary_images(x, name):
    for i in range(min(4, x.get_shape().as_list()[3])):
        tf.summary.image("{}-{}".format(name, i), x[:, :, :, i:i+1])

class CNN:
    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        self.tfx = None
        self.tfy = None
        self.tfp = None
        self.xent = None
        self.tftrain1 = None
        self.tftrain2 = None
        self.tftrain_all = None
        self.tfkp = None
        self.tfacc = None
        self.train_counter = 0
        self.test = None
        self.embedding_input = None


    def NN1(self, x):
        assert x.get_shape().as_list()[:3] == [None, 101, 101]
        x = nn.convolution(x, 16, w=4) # 98
        x = nn.convolution(x) # 96
        x = nn.max_pool(x)
        x = nn.batch_normalization(x, self.tfacc)

        ########################################################################
        assert x.get_shape().as_list() == [None, 48, 48, 16]
        x = nn.convolution(x, 32) # 46
        x = nn.convolution(x) # 44
        x = nn.max_pool(x)
        x = nn.batch_normalization(x, self.tfacc)

        ########################################################################
        assert x.get_shape().as_list() == [None, 22, 22, 32]
        x = nn.convolution(x, 64) # 20
        x = nn.convolution(x) # 18
        x = nn.max_pool(x)
        x = nn.batch_normalization(x, self.tfacc)
        x = tf.nn.dropout(x, self.tfkp)

        ########################################################################
        assert x.get_shape().as_list() == [None, 9, 9, 64]
        x = nn.convolution(x, 128) # 7
        x = tf.nn.dropout(x, self.tfkp)

        x = nn.convolution(x) # 5
        summary_images(x, "nn1")
        x = nn.batch_normalization(x, self.tfacc)
        x = tf.nn.dropout(x, self.tfkp)

        ########################################################################
        assert x.get_shape().as_list() == [None, 5, 5, 128]
        x = nn.convolution(x, 1024, w=5)

        ########################################################################
        assert x.get_shape().as_list() == [None, 1, 1, 1024]
        x = tf.reshape(x, [-1, x.get_shape().as_list()[-1]])

        x = tf.nn.dropout(x, self.tfkp)

        x = nn.fullyconnected(x, 1024)
        x = tf.nn.dropout(x, self.tfkp)

        x = nn.fullyconnected(x, 1024)
        x = nn.batch_normalization(x, self.tfacc)
        self.test = x

        x = nn.fullyconnected(x, 2, activation=None)
        return x

    ########################################################################
    def NN2(self, x):
        assert x.get_shape().as_list()[:3] == [None, 101, 101]

        x = x[:, 28:73, 28:73, :]

        ############################################################
        assert x.get_shape().as_list()[:3] == [None, 45, 45]
        x = nn.convolution(x, 16, w=5) # 41
        x = nn.convolution(x, 19, w=5) # 37
        x = nn.convolution(x, 23, w=5) # 33
        x = nn.batch_normalization(x, self.tfacc)
        x = nn.convolution(x, 27, w=5) # 29
        x = nn.convolution(x, 33, w=5) # 25
        x = nn.convolution(x, 39, w=5) # 21
        x = nn.batch_normalization(x, self.tfacc)
        x = nn.convolution(x, 47, w=5) # 17
        x = nn.convolution(x, 57, w=5) # 13
        x = nn.convolution(x, 68, w=5) # 9
        x = nn.batch_normalization(x, self.tfacc)
        x = nn.convolution(x, 82, w=5) # 5
        summary_images(x, "nn2")

        ############################################################
        assert x.get_shape().as_list() == [None, 5, 5, 82]
        x = tf.nn.dropout(x, self.tfkp)
        x = nn.convolution(x, 256, w=5) # 1

        ############################################################
        assert x.get_shape().as_list() == [None, 1, 1, 256]
        x = tf.reshape(x, [-1, 256])

        x = tf.nn.dropout(x, self.tfkp)
        x = nn.fullyconnected(x, 512)

        x = tf.nn.dropout(x, self.tfkp)
        x = nn.fullyconnected(x, 512)
        x = nn.batch_normalization(x, self.tfacc)

        x = nn.fullyconnected(x, 2, activation=None)
        return x


    def create_architecture(self, bands):
        self.tfkp = tf.placeholder_with_default(tf.constant(1.0, tf.float32), [], name="kp")
        self.tfacc = tf.placeholder_with_default(tf.constant(0.0, tf.float32), [], name="acc")
        x = self.tfx = tf.placeholder(tf.float32, [None, 101, 101, bands], name="input")
        # mean = 0 and std = 1

        summary_images(x, "input")

        with tf.name_scope("nn1"):
            x1 = self.NN1(x)
        tv1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        with tf.name_scope("nn2"):
            x2 = self.NN2(x)
        tv2 = [x for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if not x in tv1]

        with tf.name_scope("nn3"):
            x = tf.concat([x1, x2], 1)
            self.embedding_input = x
            x = nn.fullyconnected(x, 8)
            x = nn.fullyconnected(x, 8)
            x = nn.fullyconnected(x, 1, activation=None)
        tv_join = [x for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if not x in tv1 and not x in tv2]

        ########################################################################
        assert x.get_shape().as_list() == [None, 1]
        self.tfp = tf.nn.sigmoid(tf.reshape(x, [-1]))

        with tf.name_scope("xent"):
            self.tfy = tf.placeholder(tf.float32, [None])
            xent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=tf.reshape(self.tfy, [-1, 1]))
            # [None, 1]
            self.xent = tf.reduce_mean(xent)

        with tf.name_scope("train"):
            self.tftrain1 = tf.train.AdamOptimizer(1e-4).minimize(
                self.xent, var_list=tv1 + tv_join)
            self.tftrain2 = tf.train.AdamOptimizer(1e-4).minimize(
                self.xent, var_list=tv2 + tv_join)
            self.tftrain_all = tf.train.AdamOptimizer(1e-4).minimize(
                self.xent, var_list=tv1 + tv2 + tv_join)

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
            xs[i] = dihedral(xs[i], np.random.randint(8)) * s + u

        return xs, ys

    def train(self, session, xs, ys, options=None, run_metadata=None, tensors=None):
        if tensors is None:
            tensors = []

        if self.train_counter < 10000:
            train = self.tftrain1
            x = self.train_counter
        elif self.train_counter < 20000:
            train = self.tftrain2
            x = self.train_counter - 10000
        else:
            train = self.tftrain_all
            x = self.train_counter - 20000

        acc = 0.6 ** (x / 1000.0)
        kp = 0.5 + 0.5 * 0.5 ** (x / 2000.0)

        output = session.run([train, self.xent] + tensors,
            feed_dict={self.tfx: xs, self.tfy: ys, self.tfkp: kp, self.tfacc: acc},
            options=options, run_metadata=run_metadata)

        self.train_counter += 1
        return output[1], output[2:]

    def predict_naive(self, session, images):
        return session.run(self.tfp, feed_dict={self.tfx: images})

    def predict_naive_xentropy(self, session, images, labels):
        return session.run([self.tfp, self.xent], feed_dict={self.tfx: images, self.tfy: labels})

    def predict(self, session, images):
        # exploit symmetries to make better predictions
        ps = self.predict_naive(session, images)

        for i in range(1, 8):
            ps *= self.predict_naive(session, dihedral(images, i))

        return ps

    def predict_xentropy(self, session, images, labels):
        # exploit symmetries to make better predictions
        ps, xent = self.predict_naive_xentropy(session, images, labels)

        for i in range(1, 8):
            ps *= self.predict_naive(session, dihedral(images, i))

        return ps, xent
