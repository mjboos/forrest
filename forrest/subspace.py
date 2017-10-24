from __future__ import division
import numpy as np
from sklearn.decomposition import PCA
import joblib
from itertools import izip
from functools import partial
from sklearn.model_selection import BaseCrossValidator

class LeaveOneGroupCombinationOut(BaseCrossValidator):
    def _iter_test_masks(self, X, y, groups):
        from sklearn.utils.validation import check_array
        from itertools import product
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, copy=True, ensure_2d=False, dtype=None)
        unique_subgroups = [np.unique(subgroup) for subgroup in groups.T]
        prod = product(*unique_subgroups)
        for i in prod:
            yield [subgroups==group_label for subgroups, group_label in izip(groups.T, i)]


    def split(self, X, y=None, groups=None):
        from sklearn.utils import indexable
        from sklearn.utils.validation import _num_samples
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for test_index_per_group in self._iter_test_masks(X, y, groups):
            train_index = np.logical_not(np.logical_or.reduce(test_index_per_group))
            train_index = indices[train_index]
            test_index = indices[np.logical_and.reduce(test_index_per_group)]
            yield train_index, test_index


    def get_n_splits(self, X=None, y=None, groups=None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, copy=True, ensure_2d=False, dtype=None)
        return np.multiply.reduce([len(np.unique(group)) for group in groups.T])


def fit_cross_subject_model(targets, stimulus_name, estimator, n_targets=3, n_runs=8, cv_args=None):
    '''Fits an estimator to targets using the features specified in stimulus_name.
    Returns a tuple of a fitted estimator and the cross-validated predictions'''
    from sklearn.model_selection import cross_val_predict
    if cv_args is None:
        cv_args = {'cv' : LeaveOneGroupCombinationOut()}
    # make runs equal in length
    targets = targets[:3536,:,:n_targets]
    n_samples, n_subj = targets.shape[:2]
    run_length = n_samples / n_runs
    features = joblib.load('/data/mboos/encoding/stimulus/preprocessed/{}_stimuli.pkl'.format(stimulus_name)).astype('float32')[:n_samples]
    # ordering is now [n_runs * samples_per_run * n_subj] with C-ordering (last index changes fastest
    targets = np.reshape(targets, (-1, n_targets))
    features = np.repeat(features, n_subj, axis=0)
    groups_subj = np.tile(np.arange(n_subj), n_samples)
    groups_runs = np.repeat(np.arange(n_runs), run_length*n_subj)
    groups = np.array([groups_subj, groups_runs]).T
    predictions = cross_val_predict(estimator, features, targets, groups=groups, **cv_args)
    return predictions, estimator.fit(features, targets)

def adjust_r(r, n=3539, **fdr_params):
    from statsmodels.sandbox.stats.multicomp import fdrcorrection0
    from scipy.stats import betai
    df = n-2
    t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
    prob = betai(0.5*df, 0.5, df / (df+t_squared))
    return fdrcorrection0(prob)

def which_to_flip(mat):
    '''expects mat to be of shape [ observations X participants X components ]'''
    # finds the participant to flip against
    anchors = []
    for comp_activations in mat.T:
        corrs = np.abs(np.corrcoef(comp_activations))
        corrs_agmin = np.argsort(corrs, axis=1)
        anchors.append(np.argmax([corrs_row[mins[:2]].mean() for corrs_row, mins in izip(corrs, corrs_agmin)]))

    correlations_smaller_zero = [np.corrcoef(comp_activations)[anchor] < 0
                                 for anchor, comp_activations in izip(anchors, mat.T)]
    rev_flags = [np.diag([-1 if wh else 1 for wh in corr_lz])
                 for corr_lz in correlations_smaller_zero]
    return rev_flags

def select_significant_fdr_voxels(subj, predictions, **kwargs):
    '''Returns the significant voxels adjusted by fdr'''
    from forrest.encoding import score_predictions
    scores = score_predictions(predictions, subj, **kwargs)
    scores[np.isnan(scores)] = 0
    scores[scores<0] = 0
    select_voxels = adjust_r(scores)[0]
    return select_voxels

def pca_sign_flip(mat, rev_flags=None):
    '''aligns the signs in mat [ observations X participants X components ] for n_pb'''
    if rev_flags is None:
        rev_flags = which_to_flip(mat)
    for comp, flags in enumerate(rev_flags):
        mat[:, :, comp] = mat[:, :, comp].dot(flags)
    return mat

def transform_select_voxels(X, select_voxels=None):
    return X[:, select_voxels]

def inverse_transform_select_voxels(X_transformed, select_voxels=None):
    X = np.zeros((X_transformed.shape[0], select_voxels.shape[0]))
    X[:, select_voxels] = X_transformed
    return X

def pca_reduction(predictions, select_voxels=None, **pc_params):
    '''PCA analysis for (subset of) predicted voxel activity in predictions.'''
    from os.path import join
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer

    if select_voxels is not None:
        func_transf = FunctionTransformer(func=partial(transform_select_voxels, select_voxels=select_voxels),
                inverse_func=partial(inverse_transform_select_voxels, select_voxels=select_voxels))
        pca = make_pipeline(func_transf, PCA(**pc_params))
    else:
        pca = PCA(**pc_params)
    pc_predictions = pca.fit_transform(predictions)
    return pc_predictions, pca


def project_fmri_on_pcs(subj, pca, select_voxels=None, folder='/data/mboos/encoding'):
    '''Projects fmri data from subj onto the pca and returns it'''
    from os.path import join
    fmri = joblib.load(join(folder, 'fmri', 'fmri_subj_{}.pkl'.format(subj)))
    return pca.transform(fmri)


def scores_from_pc(pc, pc_predictions, pca, subj, n_splits=10, folder='/data/mboos/encoding'):
    '''Computes scores for individual voxels for subj'''
    from os.path import join
    from sklearn.preprocessing import StandardScaler
    from forrest.encoding import score_predictions
    pc_predictions
    pc_predictions[:, np.arange(pc_preds.shape[-1]) != pc] = 0
    file_name = join(folder, 'fmri', 'fmri_subj_{}.pkl'.format(subj))
    fmri = joblib.load(fl, mmap_mode='c')
    predictions = pca.inverse_transform(pc_predictions).astype('float32')
    return score_predictions(predictions, subj, folder=folder, n_splits=n_splits)

def select_all_voxels(subj, predictions, **kwargs):
    return None

def pca_on_predictions(subj, stimulus_name='logBSC_H200', select_func=select_all_voxels,
                       folder='/data/mboos/encoding', memory=joblib.Memory(cachedir=None),
                       ridge_params={}, pc_params={}):
    '''does pca analysis on predictions for stimuli in subj'''
    from forrest.encoding import encoding_for_subject
    pca_for_subj_cached = memory.cache(pca_reduction)
    predictions, _ = encoding_for_subject(subj, stimulus_name=stimulus_name, folder=folder, memory=memory, **ridge_params)
    return pca_for_subj_cached(predictions, select_voxels=select_func(subj, predictions, folder=folder), **pc_params)

def get_pc_preds_pca_list(stimulus_name,
        subjects = [1, 2, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19],
        select_func=select_all_voxels, memory=joblib.Memory(cachedir=None),
        ridge_params={}, pc_params={}):
    '''Returns the (unflipped) predictions and pca as a list'''
    pc_predictions, pca_list = zip(*[pca_on_predictions(subj, stimulus_name=stimulus_name,
                                     select_func=select_func, memory=memory,
                                     ridge_params=ridge_params, pc_params=pc_params) for subj in subjects])

    pc_predictions = np.concatenate([preds[:, None, :] for preds in pc_predictions], axis=1)
    return pc_predictions, pca_list


def get_flipped_pcs(stimulus_name,
        subjects = [1, 2, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19],
        select_func=select_all_voxels, memory=joblib.Memory(cachedir=None),
        ridge_params={}, pc_params={}):
    '''Returns all PCs, fliped and concatenated'''
    pc_predictions, pca_list = get_pc_preds_pca_list(stimulus_name, subjects=subjects, select_func=select_func,
                                                     memory=memory, ridge_params=ridge_params, pc_params=pc_params)
    flips = which_to_flip(pc_predictions)
    pc_predictions = pca_sign_flip(pc_predictions, flips)
    return pc_predictions

def get_flipped_fmri(stimulus_name,
        subjects = [1, 2, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19],
        select_func=select_all_voxels, memory=joblib.Memory(cachedir=None),
        ridge_params={}, pc_params={}):
    '''Returns all fmri data projected onto the PCs, flipped and concatenated'''
    pc_predictions, pca_list = get_pc_preds_pca_list(stimulus_name, subjects=subjects, select_func=select_func,
                                                     memory=memory, ridge_params=ridge_params, pc_params=pc_params)
    pc_fmri = [project_fmri_on_pcs(subj, pca) for subj, pca in izip(subjects, pca_list)]
    pc_fmri = np.concatenate([fmri[:, None, :] for fmri in pc_fmri], axis=1)
    flips = which_to_flip(pc_predictions)
    pc_fmri = pca_sign_flip(pc_fmri, flips)
    return pc_fmri

def get_flips(stimulus_name,
        subjects = [1, 2, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19],
        select_func=select_all_voxels, memory=joblib.Memory(cachedir=None), **pc_params):
    '''Returns all fmri data projected onto the PCs, flipped and concatenated'''
    pc_predictions, pca_list = get_pc_preds_pca_list(stimulus_name, subjects=subjects, select_func=select_func,
                                                     memory=memory, ridge_params=ridge_params, pc_params=pc_params)
    flips = which_to_flip(pc_predictions)
    return flips
