# coding: utf-8
from __future__ import division
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import joblib
from joblib import Parallel, delayed
import sys
from os.path import join

#TODO: do it with memory?

def get_ridge_predictions_coefs(stimulus, fmri, **ridge_params):
    '''fits ridge regression to predict fmri from stimulus.
    Returns tuple with predictions and coefficients'''
    kfold = KFold(n_splits=8)
    predictions = np.vstack([RidgeCV(**ridge_params
        ).fit(stimulus[train], fmri[train]
        ).predict(stimulus[test]).astype('float32')
        for train, test in kfold.split(stimulus, fmri)])

    coefs = RidgeCV(**ridge_params).fit(stimulus, fmri).coef_.astype('float32')
    return predictions, coefs

def encoding_for_subject(subj, stimulus_name='logBSC_H200', folder='/data/mboos/encoding', **ridge_params):
    '''Fits voxel-wise encoding models for subj using stimuli in stimulus_name
    Saves the predictions, coefficients, and scores to disk.'''

    stimulus = joblib.load(join(folder,'stimulus','preprocessed',
            '{}_stimuli.pkl'.format(stimulus_name)), mmap_mode='r+')

    fmri_files = [join(folder,'fmri','fmri_subj_{}_split_{}.pkl'.format(subj, split))
                  for split in xrange(10)]

    for split_nr, fmri_split_file in enumerate(fmri_files):
       fmri = joblib.load(fmri_split_file, mmap_mode='r+')
       predictions, coefs = get_ridge_predictions_coefs(stimulus, fmri,
                                                        **ridge_params)
       scores = np.array([np.corrcoef(predictions[:, i], fmri[:, i])[0,1]
                          for i in xrange(fmri.shape[1])])
       joblib.dump(predictions,
                   join(folder, 'predictions',
                       'preds_{}_subj_{}_split_{}.pkl'.format(stimulus_name, subj, split_nr)))
       joblib.dump(coefs,
                   join(folder,'coefs',
                       'coefs_{}_subj_{}_split_{}.pkl'.format(stimulus_name, subj, split_nr)))
       joblib.dump(scores,
                   join(folder, 'scores',
                       'scores_{}_subj_{}_split_{}.pkl'.format(stimulus_name, subj, split_nr)))
