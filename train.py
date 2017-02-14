# pylint: disable=C,R,no-member

# Usage
# python3 neural_train.py arch?/arch.py path_to_npz_files output_path number_of_iteration

import tensorflow as tf
import numpy as np
from sys import argv
from time import time, sleep
import queue
import threading
from sklearn import metrics
from astropy.io import fits
import importlib.util
from shutil import copy2
import os
import sys
import glob

def make_fits(files, labels, predict, path='.', suffix='', maxi=500, band=0):
    # pylint: disable=no-member
    assert labels.dtype == 'bool'
    assert predict.dtype == 'bool'

    fp = np.where(np.logical_and(labels == False, predict == True))[0]
    fn = np.where(np.logical_and(labels == True,  predict == False))[0]
    tp = np.where(np.logical_and(labels == True,  predict == True))[0]
    tn = np.where(np.logical_and(labels == False, predict == False))[0]

    np.savetxt('{}/fp{}.txt'.format(path, suffix), [np.load(files[i])['id'] for i in fp], "%s  ")
    np.savetxt('{}/fn{}.txt'.format(path, suffix), [np.load(files[i])['id'] for i in fn], "%s  ")
    np.savetxt('{}/tp{}.txt'.format(path, suffix), [np.load(files[i])['id'] for i in tp], "%s  ")
    np.savetxt('{}/tn{}.txt'.format(path, suffix), [np.load(files[i])['id'] for i in tn], "%s  ")

    fits.PrimaryHDU([np.load(files[i])['image'][:,:,band] for i in fp[:maxi]]).writeto('{}/fp{}.fits'.format(path, suffix), overwrite=True)
    fits.PrimaryHDU([np.load(files[i])['image'][:,:,band] for i in fn[:maxi]]).writeto('{}/fn{}.fits'.format(path, suffix), overwrite=True)
    fits.PrimaryHDU([np.load(files[i])['image'][:,:,band] for i in tp[:maxi]]).writeto('{}/tp{}.fits'.format(path, suffix), overwrite=True)
    fits.PrimaryHDU([np.load(files[i])['image'][:,:,band] for i in tn[:maxi]]).writeto('{}/tn{}.fits'.format(path, suffix), overwrite=True)


def predict_all(session, CNN, cnn, files, f, step=50):
    q = queue.Queue(20)  # batches in the queue
    ps = np.zeros(len(files), np.float64)
    xent_list = []

    def compute():
        for j in range(0, len(files), step):
            t0 = time()

            rem = len(files) // step - j // step
            if q.qsize() < min(2, rem):
                while q.qsize() < min(20, rem):
                    sleep(0.05)

            xs, ys = q.get()
            t1 = time()

            k = min(j + step, len(files))
            ps[j:k], xent = cnn.predict_xentropy(session, xs, ys)
            xent_list.append(xent * (k-j))

            t2 = time()
            f.write('{}/{} ({}) {: >6.3f}s+{:.3f}s\n'.format(
                j, len(files), q.qsize(), t1 - t0, t2 - t1))
            f.flush()

            q.task_done()

    t = threading.Thread(target=compute)
    t.daemon = True
    t.start()

    for j in range(0, len(files), step):
        k = min(j + step, len(files))
        xs = CNN.load(files[j:k])
        ys = np.array([np.load(f)['is_lens'] for f in files[j:k]]).astype(np.float32)
        q.put((xs, ys))

    q.join()

    return ps, np.sum(xent_list) / len(files)


def main(arch_path, images_path, output_path, n_iter):
    time_total_0 = time()
    if os.path.isdir(output_path):
        resume = True
        if not os.path.isdir(output_path + '/iter'):
            sys.exit("Try to resume computation : no iter dir in the directory")
        if not arch_path.startswith(output_path):
            sys.exit("Try to resume computation : you need to resume with the same architecture")
        f = open(output_path + '/log.txt', 'a')
        fs = open(output_path + '/stats_test.txt', 'a')
        fst = open(output_path + '/stats_train.txt', 'a')
        fm = open(output_path + '/metrics.txt', 'a')
        fx = open(output_path + '/xent_batch.txt', 'a')
    else:
        resume = False
        os.makedirs(output_path)
        os.makedirs(output_path + '/iter')
        f = open(output_path + '/log.txt', 'w')
        fs = open(output_path + '/stats_test.txt', 'w')
        fst = open(output_path + '/stats_train.txt', 'w')
        fm = open(output_path + '/metrics.txt', 'w')
        fx = open(output_path + '/xent_batch.txt', 'w')

        f.write("{}\n".format(argv))
        f.flush()

        copy2(arch_path, output_path + '/arch.py')

    f.write("Loading {}...".format(arch_path))
    f.flush()

    spec = importlib.util.spec_from_file_location("module.name", arch_path)
    neural = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(neural)
    CNN = neural.CNN

    cnn = CNN()

    f.write(" Done\nSplit data set...")
    f.flush()

    files_test, files_train = CNN.split_test_train(images_path)

    f.write(" Done\n")
    f.write("{: <6} images into train set\n".format(len(files_train)))
    f.write("{: <6} images into test set\n".format(len(files_test)))
    f.write("Create TF session...")
    f.flush()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    f.write(" Done\nCreate graph...")
    f.flush()

    bands = np.load(files_train[0])['image'].shape[2]
    cnn.create_architecture(bands=bands)
    f.write("Done\n{} bands\n".format(bands))
    f.flush()

    saver = tf.train.Saver(max_to_keep=20)

    f.write("Extract test labels...")
    f.flush()
    labels_test = []
    for fi in files_test:
        with np.load(fi) as data:
            labels_test.append(data['is_lens'])
    labels_test = np.array(labels_test, np.float32)
    f.write(" Done\nExtract train labels...")
    f.flush()
    labels_train = []
    for fi in files_train:
        with np.load(fi) as data:
            labels_train.append(data['is_lens'])
    labels_train = np.array(labels_train, np.float32)
    f.write(" Done\n")
    f.flush()

    if not resume:
        fs.write("# predictions on the test set\n")
        fs.write("# iteration | probabilities\n")
        fs.write("-1 {}\n".format(" ".join([str(l) for l in labels_test])))
        fs.flush()

        fst.write("# predictions on the training set\n")
        fst.write("# iteration | probabilities\n")
        fst.write("-1 {}\n".format(" ".join([str(l) for l in labels_train[:len(labels_test)]])))
        fst.flush()

        fm.write("# iteration xent_test auc_test xent_train auc_train\n")
        fx.write("# iteration xent_batch \n")

    if resume:
        f.write("Restore session...")
        f.flush()
        backup = sorted(glob.glob(output_path + '/iter/*.data.index'))[-1]
        backup = backup.rsplit('.', 1)[0] # remove .index
        resume_iter = int(backup.split('/')[-1].split('.')[0])
        saver.restore(session, backup)
        f.write(' Done\nBackup file : {}\n'.format(backup))
        f.flush()
    else:
        f.write("Initialize variables...")
        f.flush()
        resume_iter = 0
        session.run(tf.global_variables_initializer())
        tf.train.write_graph(session.graph_def, output_path, "graph.pb", False)
        f.write(" Done\n")
        f.flush()

    cnn.train_counter = resume_iter

    def print_log(xs, ys):
        ps, xent = cnn.predict_xentropy(session, xs, ys)
        ys = ys.astype(np.int32)

        f.write('{}\n'.format(' '.join(['{:.2f}/{}'.format(p, y) for (p, y) in zip(ps, ys) if y == 1])))
        f.write('{}\n'.format(' '.join(['{:.2f}/{}'.format(p, y) for (p, y) in zip(ps, ys) if y == 0])))
        f.write('=> xent = {:.4}\n'.format(xent))
        f.write('< pred|label=1 > = {:.3g}\n'.format(np.sum(ps * ys) / np.sum(ys == 1)))
        f.write('< pred|label=0 > = {:.3g}\n'.format(np.sum(ps * (1. - ys)) / np.sum(ys == 0)))
        ps = (ps > 0.5).astype(np.int32)
        tp = np.sum(np.logical_and(ys == 1, ps == 1))
        tn = np.sum(np.logical_and(ys == 0, ps == 0))
        fp = np.sum(np.logical_and(ys == 0, ps == 1))
        fn = np.sum(np.logical_and(ys == 1, ps == 0))
        f.write('tp:{} tn:{} fp:{} fn:{}\n'.format(tp, tn, fp, fn))
        f.flush()

    def save_statistics(i):
        if (i // 1000) % 2 == 1:
            save_path = saver.save(session, '{}/iter/{:06d}.data'.format(output_path, i))
            f.write('Model saved in file: {}\n'.format(save_path))

        ps_test, xentropy_test = predict_all(session, CNN, cnn, files_test, f)
        ps_train, xentropy_train = predict_all(session, CNN, cnn, files_train[:len(files_test)], f)

        auc_test = metrics.roc_auc_score(labels_test, ps_test)
        auc_train = metrics.roc_auc_score(labels_train[:len(files_test)], ps_train)

        fm.write("{} {:.8g} {:.8g} {:.8g} {:.8g}\n".format(i, xentropy_test, auc_test, xentropy_train, auc_train))
        fm.flush()

        f.write("     |  TEST    |  TRAIN\n")
        f.write("-----+----------+-------\n")
        f.write("xent |  {: <8.4}|  {:.4}\n".format(xentropy_test, xentropy_train))
        f.write("auc  |  {: <8.4}|  {:.4}\n".format(auc_test, auc_train))
        f.flush()

        fs.write("{} {}\n".format(i, " ".join(["{:.12g}".format(p) for p in ps_test])))
        fs.flush()
        fst.write("{} {}\n".format(i, " ".join(["{:.12g}".format(p) for p in ps_train])))
        fst.flush()

        #make_fits(files_test, labels_test == 1, ps_test > 0.5, output_path + '/iter', "_{:06d}".format(i))

    f.write("Start daemon...")
    f.flush()

    # Use a Queue to generate batches and train in parallel
    q = queue.Queue(50)  # batches in the queue

    def trainer():
        for i in range(resume_iter, resume_iter + n_iter + 1):
            t0 = time()

            rem = resume_iter + n_iter + 1 - i
            if q.qsize() < min(3, rem):
                while q.qsize() < min(50, rem):
                    sleep(0.05)

            xs, ys = q.get()
            t1 = time()

            if i % 100 == 0 and i != 0:
                f.write("Before the training\n")
                f.write("===================\n")
                f.flush()
                print_log(xs, ys)

            if i == 102 or i == 1002:
                from tensorflow.python.client import timeline
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                xentropy = cnn.train(session, xs, ys, options=run_options, run_metadata=run_metadata)
                # google chrome : chrome://tracing/

                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open(output_path + '/timeline.json', 'w') as tlf:
                    tlf.write(ctf)
            else:
                xentropy = cnn.train(session, xs, ys)

            fx.write('{}    {:.6} \n'.format(i, xentropy))

            if i % 100 == 0 and i != 0:
                f.write("\nAfter the training\n")
                f.write("==================\n")
                print_log(xs, ys)
                fx.flush()

            if i % 1000 == 0 and i != 0:
                save_statistics(i)

            t2 = time()
            f.write('{:06d}: ({}) {: >6.3f}s+{:.3f}s {} xent_batch={:.3f}\n'.format(
                i, q.qsize(), t1 - t0, t2 - t1, xs.shape, xentropy))
            f.flush()

            q.task_done()

    t = threading.Thread(target=trainer)
    t.daemon = True
    t.start()

    f.write(" Done\nStart feeders...")
    f.flush()

    # the n+1
    xs, ys = CNN.batch(files_train, labels_train)
    q.put((xs, ys))

    n_feeders = 2
    assert n_iter % n_feeders == 0
    def feeder():
        for _ in range(n_iter // n_feeders):
            xs, ys = CNN.batch(files_train, labels_train)
            q.put((xs, ys))

    threads = [threading.Thread(target=feeder) for _ in range(n_feeders)]
    for t in threads:
        t.start()
    f.write("Done\n")
    f.flush()
    for t in threads:
        t.join()

    q.join()
    session.close()

    t = time() - time_total_0
    f.write("total time : {}h {}min".format(t // 3600, (t % 3600) // 60))

    f.close()
    fs.close()
    fm.close()
    fx.close()


if __name__ == '__main__':
    main(argv[1], argv[2], argv[3], int(argv[4]))
