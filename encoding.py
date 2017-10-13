# coding: utf-8
from __future__ import division
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import joblib
from joblib import Parallel, delayed
import sys
from os.path import join

def get_ridge_predictions_coefs(stimulus, fmri, **ridge_params):
    '''fits ridge regression to predict fmri from stimulus.
    Returns tuple with predictions and coefficients'''
    kfold = KFold(n_splits=8)
    predictions = np.vstack([RidgeCV(**ridge_params
        ).fit(stimulus[train], fmri[train]
        ).predict(stimulus[test]).astype('float32')
        for train, test in kfold.split(stimulus, fmri)])

    ridge = RidgeCV(**ridge_params).fit(stimulus, fmri).coef_.astype('float32')
    return predictions, ridge

#TODO: allow parallelization
def encoding_for_subject(subj, stimulus_name='logBSC_H200', folder='/data/mboos/encoding', memory=joblib.Memory(cachedir=None), **ridge_params):
    '''Fits voxel-wise encoding models for subj using stimuli in stimulus_name
    and returns a tuple of (predictions, Ridge-model)'''
    from sklearn.model_selection import KFold
    fit_model_func = memory.cache(get_ridge_predictions_coefs)
    stimulus = joblib.load(join(folder,'stimulus','preprocessed', '{}_stimuli.pkl'.format(stimulus_name)),
                           mmap_mode='r')

    fmri = joblib.load(join(folder,'fmri','fmri_subj_{}.pkl'.format(subj)),
                             mmap_mode='r')
    #TODO: parallelization
    for split_nr, fmri_split_file in enumerate(fmri_files):
       fmri = joblib.load(fmri_split_file, mmap_mode='r+')
       predictions, coefs = fit_model_func(stimulus, fmri, **ridge_params)

