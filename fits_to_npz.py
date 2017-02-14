# pylint: disable=C,R,E1101

# Usage
# python3 fits_to_npz.py output_empty_directory classifications.csv band1 [band2, band3, ...]

import csv
import numpy as np
import os
from astropy.io import fits
from sys import argv

def read_fits(list_of_paths, flags, output):
    lfn = list(zip(*[sorted(os.listdir(p)) for p in list_of_paths]))

    assert len(lfn) == len(flags)

    for fns, flag in zip(lfn, flags):
        print("read {}".format(fns))
        bands = ()
        for i, p in enumerate(list_of_paths):
            f = fits.open(os.path.join(p, fns[i]), memmap=False)
            bands += (f[0].data, )
            f.close()

        im = np.stack(bands, axis=2)

        if np.isnan(im).any() or np.isinf(im).any():
            print("Warning ! image containt Inf or NaN")

        ids = [int(f.split('.')[0].split('-')[-1]) for f in fns]

        # images of different bands must have the same ID
        assert [i == ids[0] for i in ids]

        # the ID from the CSV must match with the ID of the fits files
        assert ids[0] == int(flag[0])

        dic = {}
        dic['id'] = int(flag[0])
        dic['image'] = im
        dic['is_lens'] = int(flag[1])
        dic['einstein_area'] = float(flag[2])
        dic['numb_pix_lensed_image'] = int(flag[3])
        dic['flux_lensed_image_in_sigma'] = int(flag[4])

        np.savez('{}/{}.npz'.format(output, str(ids[0])), **dic)


def read_csv(path):
    with open(path) as f:
        reader = csv.reader(f)
        rows = [r for r in reader][1:]
        rows = [[float(x) for x in r] for r in rows]
    return np.array(rows)

def main(output, csv_path, list_of_paths):
    flags = read_csv(csv_path)
    read_fits(list_of_paths, flags, output)


if __name__ == '__main__':
    main(argv[1], argv[2], argv[3:])
