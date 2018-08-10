from __future__ import division
from joblib import load, dump
import numpy as np
import sys

#NOTES:
#1. expect matrix of n_tr, n_features
#2. write helper function to preprocess time-series data into n_tr, n_features
#3. write audio helper function to preprocess time-series data into n_tr, n_features or n_tr, n_timesteps, n_features

def lag_features(features, n_samples_lag=3, n_samples_offset=1):
    '''Lags features with a lag of n_samples_lag and removes
    n_samples_offset most recent samples.
    '''
    try:
        n_samples, n_features = features.shape
    except ValueError:
        raise ValueError('features needs to be of shape (n_samples, n_features)')
    strides = (features.strides[0],) + features.strides

    # rolling window of length n_samples_lag
    shape = (n_samples - n_samples_lag + 1, n_samples_lag, n_features)

    features = np.lib.stride_tricks.as_strided(features[::-1,:].copy(),
                                              shape=shape,
                                              strides=strides)[::-1, :, :]
    features = np.reshape(features, (n_samples, -1))
    if n_samples_offset > 0:
        features = features[:, :-(n_samples_offset*n_features)]

    # remove n_samples_offset features
    return features

def remove_transitions(features, duration=[902,882,876,976,924,878,1084,676]):
    pass

# the length of the movie segments without the transition TRs 
# (like they are saved in patches)
movieseg_duration = duration[:]
movieseg_duration[0] -= 8
movieseg_duration[-1] -= 8
movieseg_duration[1:-1] -= 16

mvcs = np.cumsum(movieseg_duration)

# we need to remove the last 2s of the second to last stimulus
to_delete = (mvcs[-2]-2)*10 + np.arange(20)

patches = np.delete(patches, to_delete, axis=0)
# shape of TR samples
# note: column ordering is now oldest --> newest in steps of 50
patches = np.reshape(patches, (-1, 200*20))

strides = (patches.strides[0],) + patches.strides

# rolling window of length 4 samples
shape = (patches.shape[0] - 4 + 1, 4, patches.shape[1])

patches = np.lib.stride_tricks.as_strided(patches[::-1,:].copy(),
                                          shape=shape,
                                          strides=strides)[::-1, :, :]

patches = np.reshape(patches, (patches.shape[0], -1))

# we kick out the most recent sample
patches = patches[:, :-4000]

dump(patches,
     spenc_dir+'MaThe/prepro/logBSC_H200_stimuli.pkl')

