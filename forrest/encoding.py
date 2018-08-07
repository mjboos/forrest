# coding: utf-8
from __future__ import division
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import joblib
from joblib import Parallel, delayed
import sys
from os.path import join
from sklearn.multioutput import MultiOutputEstimator, _fit_estimator
from sklearn.utils.fixes import parallel_helper
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import RegressorMixin

def coef_to_float32(estimator):
    estimator.coef_ = estimator.coef_.astype('float32')
    return estimator

class BlockMultiOutput(MultiOutputEstimator, RegressorMixin):
    """Multi target regression with block-wise fit

    This strategy consists of splitting the targets in blocks and fitting one regressor per block.
    The estimator used needs to natively support multioutput.
    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` and `predict` and supporting multioutput.

    n_splits : int, optional, default=10
        The number of splits for the target variable.

    n_jobs : int, optional, default=1
        The number of jobs to run in parallel for `fit`. If -1,
        then the number of jobs is set to the number of cores.
        When individual estimators are fast to train or predict
        using `n_jobs>1` can result in slower performance due
        to the overhead of spawning processes.
    """

    def __init__(self, estimator, n_splits=10, n_jobs=1):
        self.estimator = estimator
        self.n_splits = n_splits
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """ Fit the model to data.
        Fit a separate model for each chunk of output.

        Parameters
        ----------
        X : (sparse) array-like, shape (n_samples, n_features)
            Data.

        y : (sparse) array-like, shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.

        sample_weight : array-like, shape = (n_samples) or None
            Sample weights. If None, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        Returns
        -------
        self : object
            Returns self.
        """
        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement a fit method")

        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions for "
                             "multi-output regression but has only one.")

        if (sample_weight is not None and
                not has_fit_parameter(self.estimator, 'sample_weight')):
            raise ValueError("Underlying estimator does not support"
                             " sample weights.")
        kfold = KFold(n_splits=self.n_splits)
        smpl_X, smpl_y = np.zeros((y.shape[1],1)), np.zeros((y.shape[1],1))
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(
                self.estimator, X, y[:, block], sample_weight)
            for _, block in kfold.split(smpl_X, smpl_y))
        self.estimators_ = [coef_to_float32(estimator) for estimator in self.estimators_]
        return self

    def partial_predict(self, X):
        """Predict multi-output variable using a model
         trained for each target variable block and yields predictions for each block as an iterator.

        Parameters
        ----------
        X : (sparse) array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        y : (sparse) array-like, shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.
        """
        check_is_fitted(self, 'estimators_')
        if not hasattr(self.estimator, "predict"):
            raise ValueError("The base estimator should implement a predict method")

        X = check_array(X, accept_sparse=True)

        for estimator in self.estimators_:
            yield estimator.predict(X)

    def predict(self, X):
        """Predict multi-output variable using a model
         trained for each target variable block.

        Parameters
        ----------
        X : (sparse) array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        y : (sparse) array-like, shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.
        """
        check_is_fitted(self, 'estimators_')
        if not hasattr(self.estimator, "predict"):
            raise ValueError("The base estimator should implement a predict method")

        X = check_array(X, accept_sparse=True)

        y = Parallel(n_jobs=self.n_jobs)(
            delayed(parallel_helper)(e, 'predict', X)
            for e in self.estimators_)

        return np.hstack(y)

    def score(self, X, y):
        """Returns the correlation of the prediction with the target for each output.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test samples.

        y : array-like, shape (n_samples) or (n_samples, n_outputs)
            True values for X.

        Returns
        -------
        score : float
            Correlation of self.predict(X) wrt. y.
        """
        from sklearn.preprocessing import StandardScaler
        from itertools import izip
        kfold = KFold(n_splits=self.n_splits)
        smpl_X, smpl_y = np.zeros((y.shape[1],1)), np.zeros((y.shape[1],1))
        scores = []
        for prediction, (_, block) in izip(self.partial_predict(X), kfold.split(smpl_X, smpl_y)):
            mx = StandardScaler().fit_transform(prediction).astype('float32')
            my = StandardScaler().fit_transform(y[:, block]).astype('float32')
            n = mx.shape[0]
            r = (1/(n-1))*(mx*my).sum(axis=0)
            scores.append(r)
        return np.concatenate(scores)

def get_ridge_predictions_model(stimulus, fmri, n_splits=10, n_jobs=1, **ridge_params):
    '''fits ridge regression to predict fmri from stimulus.
    Returns tuple with predictions and BlockMultioutput model'''
    estimator = RidgeCV(**ridge_params)
    kfold = KFold(n_splits=8)
    predictions = np.vstack([BlockMultiOutput(estimator, n_splits=n_splits, n_jobs=n_jobs
        ).fit(stimulus[train], fmri[train]
        ).predict(stimulus[test]).astype('float32')
        for train, test in kfold.split(stimulus, fmri)])

    model = BlockMultiOutput(estimator, n_splits=n_splits, n_jobs=n_jobs)
    model.fit(stimulus, fmri)
    return predictions, model

def encoding_for_subject(subj, stimulus_name='logBSC_H200', folder='/data/mboos/encoding', n_splits=10, n_jobs=1, memory=joblib.Memory(cachedir=None), **ridge_params):
    '''Fits voxel-wise encoding models for subj using stimuli in stimulus_name
    and returns a tuple of (predictions, Ridge-model)'''
    fit_model_func = memory.cache(get_ridge_predictions_model)
    stimulus = joblib.load(join(folder,'stimulus','preprocessed', '{}_stimuli.pkl'.format(stimulus_name)))
    fmri = joblib.load(join(folder,'fmri','fmri_subj_{}.pkl'.format(subj)),
                             mmap_mode='c')
    return fit_model_func(stimulus, fmri, n_splits=n_splits, n_jobs=n_jobs, **ridge_params)

def pearson_r(x,y):
    from sklearn.preprocessing import StandardScaler
    x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(y)
    n = x.shape[0]
    r = (1/(n-1))*(x*y).sum(axis=0)
    return r

def r2_score_predictions(predictions, subj, folder='/data/mboos/encoding', n_splits=10):
    '''Helper functions to score a matrix of obs X voxels of predictions without loading all fmri data into memory'''
    from sklearn.metrics import r2_score
    from sklearn.feature_selection import VarianceThreshold
    file_name = join(folder, 'fmri', 'fmri_subj_{}.pkl'.format(subj))
    fmri = joblib.load(file_name, mmap_mode='c')
    fmri = fmri[:predictions.shape[0]]
    split_indices = np.array_split(np.arange(predictions.shape[1]), n_splits)
    scores = []
    for indices in split_indices:
        r2_scores = r2_score(fmri[:, indices], predictions[:, indices], multioutput='raw_values')
        r2_scores[np.var(fmri[:, indices], axis=0)==0.0] = 0
        scores.append(r2_scores)
    return np.concatenate(scores)

def score_predictions(predictions, subj, folder='/data/mboos/encoding', n_splits=10):
    '''Helper functions to score a matrix of obs X voxels of predictions without loading all fmri data into memory'''
    file_name = join(folder, 'fmri', 'fmri_subj_{}.pkl'.format(subj))
    fmri = joblib.load(file_name, mmap_mode='c')
    fmri = fmri[:predictions.shape[0]]
    split_indices = np.array_split(np.arange(predictions.shape[1]), n_splits)
    scores = []
    for indices in split_indices:
        scores.append(pearson_r(predictions[:, indices], fmri[:, indices]))
    return np.concatenate(scores)

