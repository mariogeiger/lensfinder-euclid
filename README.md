# lensfinder euclid

## required software

python 3.5+ is needed with the following libraries:
- `tensorflow` ([install tensorflow](https://www.tensorflow.org/install/))
- `numpy` (`sudo apt-get install python3-numpy`)
- `astropy` (`sudo apt-get install python3-astropy`)
- `sklearn` (`pip3 install sklearn`)
- `scipy` (`sudo apt-get install python3-scipy`)

## architectures

- `arch_baseline.py` baseline architecture
- `arch_views.py` two NN feed with the full image and only the central part connected at the end
- `arch_invariant.py` invariant NN under the dihedral group
- `arch_residual.py` based on resnet

## jupyter notebooks

([install jupyter](http://jupyter.org/))
- `get_started.ipynb` usage examples
- `predictions.ipynb` make perdictions from fits images
- `invariance_check.ipynb` test the invariance of invariant NN
- `parameters_amount.ipynb` shows the amount of trained parameters in each architecture

## prediction script

    python3 predict.py arch_baseline.py trained_variables/space_based/baseline output.txt samples/space_based/lens

## train script

The training set can be download at [the challenge page](http://metcalf1.bo.astro.it/blf-portal/gg_challenge.html).

Then run the script `fits_to_npz.py` with the appropriate arguments.
It will generate npz files containing the images with labels.

Finally start the training for 50000 iteration of SGD

    python3 train.py arch_baseline.py npz_files output 50000
    
To see the progress, run the following in another terminal 

    tail -f output/log.txt
