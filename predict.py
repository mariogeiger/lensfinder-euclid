# pylint: disable=C,R,E1101
from sys import argv
import tensorflow as tf
from astropy.io import fits
from os import listdir, path
import queue
import threading
import numpy as np
from time import time
import importlib.util

def main(arch_path, restorepath, output, directory_bands):
    assert len(directory_bands) >= 1

    print("ls directories...", end='', flush=True)
    sorted_files = [sorted(listdir(p)) for p in directory_bands]
    assert all([len(sorted_files[0]) == len(x) for x in sorted_files])
    print(" done", flush=True)


    print("load arch python script...", end='', flush=True)
    spec = importlib.util.spec_from_file_location("module.name", arch_path)
    neural = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(neural)
    CNN = neural.CNN
    print(" done", flush=True)

    print("TF session...", end='', flush=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    print(" done", flush=True)

    print("create graph...", end='', flush=True)
    c = CNN()
    c.create_architecture(bands=len(directory_bands))
    print(" done", flush=True)

    print("restore variables...", end='', flush=True)
    saver = tf.train.Saver()
    saver.restore(sess, restorepath)
    print(" done", flush=True)

    print("start daemon...", end='', flush=True)
    fout = open(output, 'w')

    q = queue.Queue(20)

    def predict_batch():
        while True:
            ids, xs = q.get()
            time_start = time()
            ps = c.predict(sess, xs)
            time_end = time()
            print("{:.3}s \t{} - {}".format(time_end - time_start, ids[0], ids[-1]))

            for i, p in zip(ids, ps):
                fout.write("{} {:.9}\n".format(i, p))
            q.task_done()

    t = threading.Thread(target=predict_batch)
    t.daemon = True
    t.start()
    print(" done", flush=True)

    # feeding the Queue
    zipped_files = list(zip(*sorted_files))
    for i in range(0, len(zipped_files), 100):
        ids = []
        xs = []
        for j in range(i, min(i + 100, len(zipped_files))):
            bands = ()
            numbers = []
            for k in range(0, len(directory_bands)):
                filename = zipped_files[j][k]

                f = fits.open(path.join(directory_bands[k], filename), memmap=False)
                bands += (f[0].data, )
                f.close()

                numbers.append(int(filename.split('.')[0].split('-')[-1]))

            assert all([n == numbers[0] for n in numbers])
            im = np.stack(bands, axis=2)

            ids.append(numbers[0])
            xs.append(im)

        xs = np.array(xs)
        xs = CNN.prepare(xs)
        q.put((ids, xs))

    q.join()
    sess.close()
    fout.close()
    print("finished")


if __name__ == '__main__':
    main(argv[1], argv[2], argv[3], argv[4:])
