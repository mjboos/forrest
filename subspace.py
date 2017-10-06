from __future__ import division
import numpy as np
from sklearn.decomposition import PCA
import joblib
from itertools import izip
import matplotlib as mpl
import mkl
from functools import partial
from sklearn.model_selection import BaseCrossValidator

#check with sklearn about memory object

mkl.set_num_threads(4)

mpl.use('Agg')

cachedir = '/data/mboos/joblib'
memory = joblib.Memory(cachedir=cachedir)

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

@memory.cache
def fit_cross_subject_model(targets, model, estimator, n_targets=3, n_runs=8, cv_args=None):
    '''Fits an estimator to targets using the features specified in model.
    Returns a tuple of a fitted estimator and the cross-validated predictions'''
    from sklearn.model_selection import cross_val_predict
    if cv_args is None:
        cv_args = {'cv' : LeaveOneGroupCombinationOut()}
    # make runs equal in length
    targets = targets[:3536,:,:n_targets]
    n_samples, n_subj = targets.shape[:2]
    run_length = n_samples / n_runs
    features = joblib.load('/data/mboos/encoding/stimulus/preprocessed/{}_stimuli.pkl'.format(model)).astype('float32')[:n_samples]
    # ordering is now [n_runs * samples_per_run * n_subj] with C-ordering (last index changes fastest
    targets = np.reshape(targets, (-1, n_targets))
    features = np.repeat(features, n_subj, axis=0)
    groups_subj = np.tile(np.arange(n_subj), n_samples)
    groups_runs = np.repeat(np.arange(n_runs), run_length*n_subj)
    groups = np.array([groups_subj, groups_runs]).T
    predictions = cross_val_predict(estimator, features, targets, groups=groups, **cv_args)
    return estimator.fit(features, targets), predictions

def load_scores(subj, model, folder='/data/mboos/encoding'):
    from os.path import join
    scores = np.concatenate([joblib.load(
        join(folder, 'scores', 'scores_{}_subj_{}_split_{}.pkl'.format(model, subj, split))) for split in xrange(10) ])
    return scores

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

def select_significant_fdr_voxels(subj, model):
    '''Returns the significant voxels adjusted by fdr'''
    scores = load_scores(subj, model)
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

@memory.cache
def pca_analysis_for_subj(subj, model='logBSC_H200', select_voxels=None, folder='/data/mboos/encoding', **pc_params):
    '''Does PCA Analysis for one subject and one model'''
    from os.path import join
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer
    file_names = [join(folder, 'predictions',
        'preds_{}_subj_{}_split_{}.pkl'.format(model, subj, split)) for split in xrange(10)]
    predictions = np.hstack([joblib.load(fl) for fl in file_names])

    if select_voxels is not None:
        func_transf = FunctionTransformer(func=partial(transform_select_voxels, select_voxels=select_voxels),
                inverse_func=partial(inverse_transform_select_voxels, select_voxels=select_voxels))
        pca = make_pipeline(func_transf, PCA(**pc_params))
    else:
        pca = PCA(**pc_params)
    pc_predictions = pca.fit_transform(predictions)
    return pc_predictions, pca

@memory.cache
def project_fmri_on_pcs(subj, pca, select_voxels=None, folder='/data/mboos/encoding'):
    '''Projects fmri data from subj onto the pca and returns it'''
    from os.path import join
    file_names = [join(folder, 'fmri',
        'fmri_subj_{}_split_{}.pkl'.format(subj, split)) for split in xrange(10)]
    fmri = np.hstack([joblib.load(fl) for fl in file_names])
    return pca.transform(fmri)

@memory.cache
def scores_from_pc(pc, subj, model, folder='/data/mboos/encoding', **pca_analysis_args):
    '''Computes scores for individual voxels for subj'''
    from os.path import join
    from sklearn.preprocessing import StandardScaler
    pc_preds, pca = pca_analysis_for_subj(subj, model, **pca_analysis_args)
    pc_preds[:, np.arange(pc_preds.shape[-1]) != pc] = 0
    file_names = [join(folder, 'fmri',
        'fmri_subj_{}_split_{}.pkl'.format(subj, split)) for split in xrange(10)]
    fmri_splits = [joblib.load(fl, mmap_mode='r') for fl in file_names]
    predictions = pca.inverse_transform(pc_preds).astype('float32')
    predictions = np.array_split(predictions, len(fmri_splits), axis=1)
    scores = []
    for preds, fmri in izip(predictions, fmri_splits):
        mx = StandardScaler().fit_transform(preds).astype('float32')
        my = StandardScaler().fit_transform(fmri).astype('float32')
        n = mx.shape[0]
        r = (1/(n-1))*(mx*my).sum(axis=0)
        scores.append(r)
    return np.concatenate(scores)

def scores_from_predictions(predictions, subj, model, select=None, folder='/data/mboos/encoding'):
    from os.path import join
    from sklearn.preprocessing import StandardScaler
    if select is None:
        select = np.ones(predictions.shape[:1], dtype=bool)
    file_names = [join(folder, 'fmri',
        'fmri_subj_{}_split_{}.pkl'.format(subj, split)) for split in xrange(10)]
    fmri_splits = [joblib.load(fl, mmap_mode='r') for fl in file_names]
    predictions = np.array_split(predictions, len(fmri_splits), axis=1)
    scores = []
    for preds, fmri in izip(predictions, fmri_splits):
        mx = StandardScaler().fit_transform(preds).astype('float32')
        my = StandardScaler().fit_transform(fmri[select]).astype('float32')
        n = mx.shape[0]
        my = my[:n]
        r = (1/(n-1))*(mx*my).sum(axis=0)
        scores.append(r)
    return np.concatenate(scores)

def get_pc_scores(pcs, model, subjects=[1,2,5,6,7,8,9,11,12,14,15,16,17,18,19], **pca_analysis_args):
    for pc in pcs:
        scores_act = []
        for subj in subjects:
            scores = scores_from_pc(pc, subj, model, **pca_analysis_args)
            scores[np.isnan(scores)] = 0
            scores[scores<0] = 0
            scores_act.append(scores)
        yield scores_act

def select_all_voxels(subj, model):
    return None

@memory.cache
def get_flipped_pcs(model,
        subjects = [1, 2, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19],
        select_func=select_all_voxels, **pc_params):
    '''Returns all PCs, fliped and concatenated'''
    pc_predictions, pca_list = zip(*[pca_analysis_for_subj(subj, model=model,
                                     select_voxels=select_func(subj, model), **pc_params) for subj in subjects])
    pc_predictions = np.concatenate([preds[:, None, :] for preds in pc_predictions], axis=1)
    flips = which_to_flip(pc_predictions)
    pc_predictions = pca_sign_flip(pc_predictions, flips)
    return pc_predictions

@memory.cache
def get_flipped_fmri(model,
        subjects = [1, 2, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19],
        select_func=select_all_voxels, **pc_params):
    '''Returns all fmri data projected onto the PCs, flipped and concatenated'''
    pc_predictions, pca_list = zip(*[pca_analysis_for_subj(subj, model=model,
                                                            select_voxels=select_func(subj, model), **pc_params) for subj in subjects])
    pc_predictions = np.concatenate([preds[:, None, :] for preds in pc_predictions], axis=1)
    pc_fmri = [project_fmri_on_pcs(subj, pca) for subj, pca in izip(subjects, pca_list)]
    pc_fmri = np.concatenate([fmri[:, None, :] for fmri in pc_fmri], axis=1)
    flips = which_to_flip(pc_predictions)
    pc_fmri = pca_sign_flip(pc_fmri, flips)
    return pc_fmri

@memory.cache
def get_PCA_list(model,
        subjects = [1, 2, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19],
        select_func=select_all_voxels, **pc_params):
    '''Returns all PCAs for subjects as a list'''
    pc_predictions, pca_list = zip(*[pca_analysis_for_subj(subj, model=model, select_voxels=select_func(subj, model), **pc_params) for subj in subjects])
    return pca_list

@memory.cache
def get_flips(model,
        subjects = [1, 2, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19], select_func=select_all_voxels, **pc_params):
    '''Returns all fmri data projected onto the PCs, flipped and concatenated'''
    pc_predictions, pca_list = zip(*[pca_analysis_for_subj(subj, model=model, select_voxels=select_func(subj, model), **pc_params) for subj in subjects])
    pc_predictions = np.concatenate([preds[:, None, :] for preds in pc_predictions], axis=1)
    flips = which_to_flip(pc_predictions)
    return flips

def area_ecdf(distribution, a=-1, b=1):
    from statsmodels.distributions import ECDF
    from scipy.integrate import quad
    ecdf = ECDF(distribution)
    return quad(ecdf, a=a, b=b)[0]

def feature_selectivity_index(activations, stimuli, n_stim=25, n_bootstrap=100):
    '''Computes the feature selectivity index for the activations'''
    activations_idx_sorted = np.argsort(activations)
    highest_idx = activations_idx_sorted[-n_stim:]
    lowest_idx = activations_idx_sorted[:n_stim]
    mean_high = np.mean(stimuli[highest_idx], axis=0)
    mean_low = np.mean(stimuli[lowest_idx], axis=0)
    corrs_high = np.array([np.corrcoef(mean_high, stimulus)[0,1]
        for stimulus in stimuli[highest_idx]])
    corrs_low = np.array([np.corrcoef(mean_low, stimulus)[0,1]
        for stimulus in stimuli[lowest_idx]])
    area_high = area_ecdf(corrs_high)
    area_low = area_ecdf(corrs_low)
    area_rnd_low = []
    area_rnd_high = []
    for i in xrange(n_bootstrap):
        random_stimuli = np.random.permutation(np.arange(stimuli.shape[0]))[:n_stim]
        corrs_random_high = np.array([np.corrcoef(mean_high, stimulus)[0,1]
                for stimulus in stimuli[random_stimuli]])
        corrs_random_low = np.array([np.corrcoef(mean_low, stimulus)[0,1]
                for stimulus in stimuli[random_stimuli]])
        area_rnd_high.append(area_ecdf(corrs_random_high))
        area_rnd_low.append(area_ecdf(corrs_random_low))
    mean_area_rnd_high = np.mean(area_rnd_high)
    mean_area_rnd_low = np.mean(area_rnd_low)
    fsi_high = (mean_area_rnd_high-area_high)/mean_area_rnd_high
    fsi_low = (mean_area_rnd_low-area_low)/mean_area_rnd_low
    return fsi_high, fsi_low

def feature_corrs(activations, stimuli, n_stim=25, n_bootstrap=100):
    '''Computes the feature selectivity index for the activations'''
    activations_idx_sorted = np.argsort(activations)
    highest_idx = activations_idx_sorted[-n_stim:]
    lowest_idx = activations_idx_sorted[:n_stim]
    mean_high = np.mean(stimuli[highest_idx], axis=0)
    mean_low = np.mean(stimuli[lowest_idx], axis=0)
    corrs_high = np.array([np.corrcoef(mean_high, stimulus)[0,1]
        for stimulus in stimuli[highest_idx]])
    corrs_low = np.array([np.corrcoef(mean_low, stimulus)[0,1]
        for stimulus in stimuli[lowest_idx]])
    return (np.mean(corrs_high), np.mean(corrs_low))

def separability(spectrogram):
    from scipy.linalg import svd
    U, s, Vh = svd(spectrogram)
    return s[0]**2 / (s**2).sum()

