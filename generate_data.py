import h5py
import numpy as np


# TODO generator for the test and validation dataset


def generate_10k(batch_size=32):
    # Let say we are just writing for the train dataset
    path = '/data4/gopal.sharma/datasets/mnt/ramdisk/max/90kDICT32px/test_chunks/'
    while (1):
        for i in range( 2 ):
            with h5py.File( path + 'test_data%d.hdf5' % i, 'r' ) as hf:
                data = hf.get( "X" )
                X = np.array( data )
                data = hf.get( "Words" )
                Words = np.array( data )
                data = hf.get( "y" )
                y = np.array( data )
            for j in range( 0, 10000 - batch_size, batch_size ):
                yield [X[j:j + batch_size],
                       Words[j:j + batch_size]], \
                      y[j:j + batch_size]


def generate_val(batch_size=32):
    # Let say we are just writing for the train dataset
    path = '/data4/gopal.sharma/datasets/mnt/ramdisk/max/90kDICT32px/val_chunks/'
    while (1):
        for i in range( 2 ):
            with h5py.File( path + 'test_data%d.hdf5' % i, 'r' ) as hf:
                data = hf.get( "X" )
                X = np.array( data )
                data = hf.get( "Words" )
                Words = np.array( data )
                data = hf.get( "y" )
                y = np.array( data )
            for j in range( 0, 10000 - batch_size, batch_size ):
                yield [X[j:j + batch_size],
                       Words[j:j + batch_size]], \
                      y[j:j + batch_size]
