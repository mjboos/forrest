from __future__ import division
import numpy as np
import librosa as lbr
import sys
import argparse
import joblib

def recursive_glob(folder_name, expr='*.wav'):
    '''Returns list of files in folder_name and all subfolders that match expr'''
    import fnmatch
    import os
    files = []
    for root, dirname, filenames in os.walk(folder_name):
        for filename in fnmatch.filter(filenames, expr):
            files.append(os.path.join(root, filename))
    return files

def extract_mel(wavname, bins=48, stepsize=441, n_fft=882, fmax=8000):
    '''Extracts the mel-frequency spectrogram from file wavname'''
    from scipy.io.wavfile import read
    sr, wavcontent = read(wavname)
    if len(wavcontent.shape) > 1:
        wavcontent = np.mean(wavcontent,axis=1)
    return lbr.feature.melspectrogram(wavcontent,hop_length=stepsize,sr=sr,n_mels=bins,fmax=fmax).T

def mel_to_zscored_patches(mel, patch_size=(10, 48), scaler=None):
    '''Extracts patches and preprocesses them by taking the logarithm and z-scoring'''
    from sklearn.preprocessing import StandardScaler
    patches = mel_to_patches(mel, patch_size=patch_size)
    if scaler is None:
        scaler = StandardScaler().fit(patches)
    patches = scaler.transform(patches)
    return patches

def mel_to_patches(mel, patch_size=(10, 48)):
    '''Extracts patches and preprocesses them by taking the logarithm and z-scoring'''
    from sklearn.feature_extraction.image import extract_patches_2d
    patches = extract_patches_2d(mel, patch_size=patch_size)[::patch_size[0], :]
    patches = np.reshape(patches, (-1, patch_size[0]*patch_size[1]))
    patches[patches<1] = 1
    patches = np.log(patches)
    return patches

def folder_to_mel_list(folder_path, scaler=None, **mel_args):
    '''generator of (mel content, file name) tuples for all wav files in folder_path'''
    import glob
    files = sorted(recursive_glob(folder_path, '*.wav'))
    file_names = [fn.split('/')[-1].split('.')[0] for fn in files]
    for fpath_wav, fname in zip(files, file_names):
        mel = extract_mel(fpath_wav, **mel_args)
        # we don't care about clips that are <6s
        if mel.shape[0] >= 600:
            yield (mel, fname)

def audio_preprocess_folder(folder_path, folder_name, preprocessed_data_folder='/data/mboos/encoding/audio_preprocessed/', scaler=None, **mel_args):
    '''Extracts Mel specgram for all wave files in folder and saves them'''
    import os
    if not os.path.exists(preprocessed_data_folder + folder_name):
        os.makedirs(preprocessed_data_folder+folder_name)
    final_folder = preprocessed_data_folder + folder_name
    for mel, fname in folder_to_mel_list(folder_path, scaler=scaler, **mel_args):
        joblib.dump(mel_to_zscored_patches(mel, scaler=scaler), final_folder+'/'+fname+'.pkl')
