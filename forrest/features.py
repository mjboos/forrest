from __future__ import division
import numpy as np
import os
import librosa as lbr
import sys
import argparse
import joblib
import glob
import pandas as pd

#TODO: document inner workings of function

def convert_annotation_to_TR(annotations, which_col=None):
    '''Converts annotations to an array where each row denotes a TR of the movie.
    annotations can either be a filename of an annotation file or a pandas DataFrame.
    which_col specificies which annotation to use, None means all.
    '''
    if isinstance(annotations, basestring):
        annotations = pd.DataFrame.from_csv(annotations, sep='\t', index_col=None)
    if which_col is not None:
        annotations = annotations[['onset', 'duration', which_col]]

    tr_vector = np.zeros((3542, annotations.shape[-1]-2))
    for i, annotation in annotations.iterrows():
        tr_vector[time_to_TR(annotation['onset'], annotation['duration'])] = annotation.values[2:]
    return tr_vector


def time_to_TR(time, duration):
    '''Indicate which TRs are active at a given time segment'''
    start_tr = np.floor(time / 2.0)
    return (start_tr + np.arange(np.ceil(duration/2.0).astype('int'))).astype('int')


def extract_audio(wavname, extraction_func=lbr.feature.melspectrogram, **kwargs):
    from scipy.io.wavfile import read
    sr, wavcontent = read(wavname)
    if len(wavcontent.shape) > 1:
        wavcontent = np.mean(wavcontent,axis=1)
    return extraction_func(wavcontent, **kwargs).T

def extract_mel(wavname, bins=48, hop_length=441, n_fft=882, fmax=8000):
    '''Extracts the mel-frequency spectrogram from file wavname'''
    from scipy.io.wavfile import read
    sr, wavcontent = read(wavname)
    if len(wavcontent.shape) > 1:
        wavcontent = np.mean(wavcontent,axis=1)
    return lbr.feature.melspectrogram(wavcontent, hop_length=hop_length, sr=sr,
                                      n_fft=n_fft, n_mels=bins, fmax=fmax).T

def folder_to_audio_generator(folder_path, audio_extractor=extract_mel,
                              tr_reshape=True, **kwargs):
    '''generator of (audio representation, file name) tuples for all wav files in folder_path
    '''
    files = sorted(glob.glob(os.path.join(folder_path, '*.wav')))
    file_names = [fn.split('/')[-1].split('.')[0] for fn in files]
    for fpath_wav, fname in zip(files, file_names):
        audio = audio_extractor(fpath_wav, **kwargs)
        if tr_reshape:
            n_tr = np.floor(lbr.get_duration(filename=fpath_wav) / 2.0).astype('int')
            try:
                audio = np.reshape(audio, (n_tr, -1, audio.shape[-1]))
            except ValueError:
                audio = np.reshape(audio[:-int(audio.shape[0] % n_tr)],
                                   (n_tr, -1, audio.shape[-1]))
        yield audio

def extract_audio_rep(folder_path, audio_extractor=extract_mel, **kwargs):
    '''Extracts Mel specgram for all wave files in folder and saves them'''
    from . import preprocessing
    audio_rep = reduce(preprocessing.cut_out_overlap,
                               folder_to_audio_generator(folder_path,
                                                         audio_extractor=audio_extractor,
                                                         **kwargs))
    # audio of the last 2s of the second-to-last run were not recorded so we remove them
    audio_rep = np.delete(audio_rep, -335, axis=0)
    assert audio_rep.shape[0] == 3542, 'Audio representation is not aligned with fMRI'
    return audio_rep

def lag_features(features, n_samples_lag=4, n_samples_offset=1):
    '''Lags features with a lag of n_samples_lag and removes
    n_samples_offset most recent samples.
    '''
    if len(features.shape) > 2:
        features = np.squeeze(features)
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
    features = np.reshape(features, (shape[0], -1))

    # remove n_samples_offset features
    if n_samples_offset > 0:
        features = features[:, :-(n_samples_offset*n_features)]

    return features

