# lensfinder euclid

## required software

python v3 is needed with the following libraries:
- `tensorflow`
- `numpy`
- `astropy`
- `sklearn`

    sudo apt-get install python3-numpy python3-astropy python3-sklearn python3-pip
    pip install tensorflow # because no package in debian at the time i wrote this

## architectures

- `arch_baseline.py` baseline architecture
- `arch_views.py` two NN feed with the full image and only the central part connected at the end
- `arch_invariant.py` invariant NN under the dihedral group

## make a prediction

    python3 predict.py arch_baseline.py trained_variables/baseline output.txt samples/space_based/lens
    
## train the NN

The training set can be download at [the challenge page](http://metcalf1.bo.astro.it/blf-portal/gg_challenge.html).

Then run the script `fits_to_npz.py` with the apporpiate argument.
It will generate an npz file for each fits file containing the image and the label.

Finally start the training for 50000 iteration of SGD

    python3 train.py arch_baseline.py npz_files ouput_directory 50000
