import numpy as np
import matplotlib.pyplot as plt
from nilearn import image as img
import pandas as pd
import joblib
import seaborn as sns
from copy import deepcopy
from coef_helper_functions import remove_BF_from_coefs, make_df_for_lineplot

def test_latent_space_reconstruction(feature, latent_activity,
                                     estimator=None, **kwargs):
    '''Returns the cross-validated explained variance (averaged across 8 folds)
    for predicting feature from latent_activity'''
    from sklearn.model_selection import cross_validate
    from sklearn.linear_model import RidgeCV
    if estimator is None:
        estimator = RidgeCV(alphas=[1e-5, 1e-3, 1e-1, 1, 1e3, 1e5])
    cv_result = cross_validate(estimator, latent_activity, feature,
                               scoring='explained_variance', cv=8, **kwargs)
    if 'estimator' in cv_result:
        return cv_result['test_score'], cv_result['estimator']
    else:
        return cv_result['test_score']

def get_feature_scores(feature_dict, latent_activity, ratings_idx,
                       estimator=None, **kwargs):
    scores_dict = dict()
    feature_names = ['Time-Frequency Separability', 'Sound level (db)',
                     'Speech duration (s)']
    for label in feature_names:
        scores_dict[label] = test_latent_space_reconstruction(
            feature_dict[label], latent_activity, estimator=estimator, **kwargs)
    scores_dict['Noise rating'] = test_latent_space_reconstruction(
            feature_dict['Noise rating'], latent_activity[ratings_idx],
            estimator=estimator, **kwargs)
    return scores_dict

def get_average_estimator():
    import dill
#    dill._dill._reverse_typemap[b'TypeType'] = type
#    dill._dill._reverse_typemap[b'CodeType']= dill._dill._reverse_typemap['CodeType']
    with open('average_estimator_py3.pkl', 'rb') as fn:
        estimator = dill.load(fn, encoding="bytes")
    return estimator

def get_feature_dict():
    bsc = joblib.load('../semisupervised/logBSC_H200_stimuli.pkl')
    separability = joblib.load('mean_sep.pkl')
    separability_pos = joblib.load('sep_of_pos_Ws_only.pkl')
    separability_pos[np.isnan(separability_pos)] = 0
    decibel = joblib.load('db_dict.pkl')
    decibel = np.array([decibel[str(i)][1] for i in range(3539)])
    speech_overlap = joblib.load('speech_overlap.pkl')
    ratings_dict = joblib.load('ratings_dict.pkl')
    feature_dict = {'Time-Frequency Separability': separability,
                    'Sound level (decibel)': decibel,
                    'Positive separability': separability_pos,
                    'Speech duration (s)': speech_overlap,
                    'Noise rating': ratings_dict['ratings'], 'BSC': bsc}
    return feature_dict

def get_cluster_infos(means_file='cluster_means_reordered.pkl',
                      idx_file='compressed_cluster_identity_reordered.pkl'):
    cluster_means = joblib.load(means_file)
    cluster_idx = joblib.load(idx_file)
    return {'means': cluster_means, 'index': cluster_idx}

def get_corr_df(joint_pcs, cluster_means, cluster_idx):
    corrs = [{cl:np.corrcoef(joint_pcs[:,pc], cluster_means[i])[0,1]
            for i, cl in enumerate(np.unique(cluster_idx))} for pc in range(3)]
    correlations = np.array([corr.values() for corr in corrs])
    corr_df = pd.concat([pd.DataFrame(correlations).melt(), pd.Series(np.tile(['PC 1', 'PC 2', 'PC 3'], 7), name='PC')], axis=1, ignore_index=True)
    corr_df.columns = ['Cluster', 'Correlation', 'PC']
    corr_df.Cluster = corr_df.Cluster.map({0:'1', 1:'2', 2:'3', 3:'4', 4:'5',5:'6',6:'7'})
    return corr_df

def cluster_reordering(cluster_means, cluster_idx):
    """Reorders cluster_means and cluster_idx and saves them as new_files"""
    reorder_dict = {1:0, 0:1, 6:2, 2:3, 8:4, 9:5, 5:6}
    reorder_list = [1,0,4,2,5,6,3]
    cluster_means_reordered = cluster_means[reorder_list]
    cluster_idx_new = deepcopy(cluster_idx)
    for current, to_be in reorder_dict.items():
        cluster_idx_new[cluster_idx==current] = to_be
    joblib.dump(cluster_means_reordered, 'cluster_means_reordered.pkl')
    joblib.dump(cluster_idx_new, 'compressed_cluster_identity_reordered.pkl')


def get_seps(features, separability, excl_idx=None):
    features = np.reshape(features, (60, 200))
    separabilities_sample = []
    for ft in features:
        if ft.any():
            if excl_idx is not None:
                separabilities_sample.append(
                        np.array([separability[loc] for loc in np.where(ft)[0] if not np.isin(loc, excl_idx)]))
            else:
                separabilities_sample.append(separability[np.where(ft)[0]])
    return np.concatenate(separabilities_sample)


#TODO: write unit test
def compute_MPS_from_STFT(BF, n_fft=882, sr=44100, fmax=8000, n_mels=48):
    import librosa as lbr
    mel_filters = lbr.filters.mel(sr=sr, n_fft=n_fft, fmax=fmax, n_mels=n_mels)
    Ws_stft = BF.dot(mel_filters)[:,:161]
    return compute_MPS(Ws_stft)

def compute_MPS(specgram):
    return np.abs(np.fft.fftshift(np.fft.rfft2(specgram), axes=0))

def compute_mps_time_and_freq_labels(n_fft=882, sr=44100, fmax=8000, n_mels=48):
    '''Returns the labels for time and frequency modulation for the input parameters'''
    import librosa as lbr
    fft_freq = lbr.core.fft_frequencies(sr=44100, n_fft=882)[:161]
    # in cycles/KHz
    mps_freqs = np.fft.rfftfreq(fft_freq.shape[0], np.diff(fft_freq)[0])*1000
    mps_times = np.fft.fftshift(np.fft.fftfreq(10,
                                1. / 10.))
    return mps_times, mps_freqs


def bin_component_indices(component, n_bins=5):
    '''Computes n_bins bins of component and returns the bin edges and indices'''
    assert len(component.shape) == 1
    _, edges = np.histogram(component, bins=n_bins)
    indices = np.digitize(component, edges)
    return edges, indices


def get_features_in_sample(bsc, feature):
    '''Melts the occurences of feature in each row of bsc into a list of lists 
    IN:
       bsc      -   ndarray of shape (samples, 12000)
       feature  -   ndarray of shape (200,)
    OUT:
        feature_list    -   list of lists
    '''
    feature_list = []
    bsc = np.reshape(bsc, (-1, 60, 200))
    for bsc_sample in bsc:
        temp_list = []
        for bsc_ts in bsc_sample:
            active_BFs = np.where(bsc_ts)[0]
            if active_BFs.size > 0:
                temp_list.append(feature[active_BFs])
        feature_list.append(np.concatenate(temp_list))
    return feature_list


def feature_list_to_df(feature_list, indices_samples, feature_name='value'):
    '''Converts a feature list to a melted dataframe
    annotated by the bins from indices_samples'''
    list_of_indices = [[idx]*len(feature_list[i])
                       for i, idx in enumerate(indices_samples)]
    return pd.DataFrame({'bin': np.concatenate(list_of_indices),
                         feature_name: np.concatenate(feature_list)})


def make_df_for_feature_sensitivity(bf_feature_dict, bsc, component, n_bins=5):
    '''Creates a melted pandas DataFrame for each feature in bf feature_dict
    IN:
        bf_feature_dict -   dictionary with auditory feature names as keys and
                            shape (200,) ndarrays quantifying the feature for each
                            BSC basis function
        bsc             -   ndarray of the Binary Sparse Coding basis function activations
        component       -   ndarray of component activation in each sample
        n_bins          -   number of bins for each principal component
    '''
    from functools import reduce
    _, indices = bin_component_indices(component, n_bins=n_bins)
    list_of_dfs = [feature_list_to_df(get_features_in_sample(bsc, feature), indices,
                                      feature_name=feature_name)
                   for feature_name, feature in bf_feature_dict.items()]
    joint_df = reduce(lambda x, y: pd.concat([x,y.drop('bin', axis=1)], axis=1), list_of_dfs)
    return joint_df


def annotate_df_with_col_value(df, col_value, col_name='PC'):
    '''Adds a column to df with col_value'''
    return pd.concat([df, pd.Series([col_value]*df.shape[0], name=col_name)], axis=1)


#TODO: think about how to do a unittest
def make_feature_pc_df(bf_feature_dict, bsc, pcs, n_bins=5):
    '''Creates a melted pandas DataFrame for each feature in bf feature_dict
    and each component in pcs.shape[1]
    IN:
        bf_feature_dict -   dictionary with auditory feature names as keys and
                            shape (200,) ndarrays quantifying the feature for each
                            BSC basis function
        bsc             -   ndarray of the Binary Sparse Coding basis function activations
        pcs             -   principal component values for each sample
        n_bins          -   number of bins for each principal component'''
    # test that the number of samples is the same
    assert pcs.shape[0] == bsc.shape[0]
    pc_df_list = [annotate_df_with_col_value(
        make_df_for_feature_sensitivity(bf_feature_dict, bsc, component, n_bins=n_bins), i+1)
                  for i, component in enumerate(pcs.T)]
    return pd.concat(pc_df_list, axis=0, ignore_index=True)


def melt_feature_df(feature_df, var_name='Cluster', **kwargs):
    '''Melts feature_df and keeps bin and PC'''
    return pd.melt(feature_df, id_vars=['bin', 'PC'], var_name=var_name, **kwargs)


def make_df_mean_feature_all_pcs(bf_feature_dict, bsc, pcs):
    '''Compute feature average per time point and put into dataframe with PCs
    IN:
        bf_feature_dict -   dictionary with auditory feature names as keys and
                            shape (200,) ndarrays quantifying the feature for each
                            BSC basis function
        bsc             -   ndarray of the Binary Sparse Coding basis function activations
        pcs             -   principal component values for each sample
    '''
    assert pcs.shape[0] == bsc.shape[0]
    average_ft_dict = make_average_feature_dict(bf_feature_dict, bsc)
    return pd.concat([pd.DataFrame(average_ft_dict), pd.DataFrame({'PC{}'.format(i+1): pcs[:,i] for i in range(pcs.shape[1])})], axis=1)

#TODO: write this
def make_df_mean_feature_all_participants(bf_feature_dict, bsc, pcs):
    '''Creates a melted pandas DataFrame for each feature in bf feature_dict
    IN:
        bf_feature_dict -   dictionary with auditory feature names as keys and
                            shape (200,) ndarrays quantifying the feature for each
                            BSC basis function
        bsc             -   ndarray of the Binary Sparse Coding basis function activations
        pcs             -   principal component values for each sample and each participant. ndarray of shape (participants, samples, components)
    '''
    #iterate through participant PC values, annotate with participant label, concat on axis 0
    list_of_dfs = [annotate_df_with_col_value(
        make_df_mean_feature_all_pcs(bf_feature_dict, bsc, pcs_participant), subj_i, col_name='Participant')
                   for subj_i, pcs_participant in enumerate(pcs)]
    return pd.concat(list_of_dfs, axis=0, ignore_index=True)


def make_df_feature_correlation(bf_feature_dict, bsc, pcs):
    '''Makes a dataframe of the mean value of each feature in feature dict for each pc
    IN:
        bf_feature_dict -   dictionary with auditory feature names as keys and
                            shape (200,) ndarrays quantifying the feature for each
                            BSC basis function OR dict of mean features values per time
        bsc             -   ndarray of the Binary Sparse Coding basis function activations
        pcs             -   principal component values for each sample and each participant
                            shape (participants, samples, components)'''
    # test that the number of samples is the same
    assert pcs.shape[1] == bsc.shape[0]
    pc_df_list = [annotate_df_with_col_value(
        make_df_feature_correlation_component(bf_feature_dict, bsc, component), i+1)
                  for i, component in enumerate(pcs.T)]
    return pd.concat(pc_df_list, axis=0, ignore_index=True)


def average_feature_list(feature_list):
    '''Averages the sublists in feature_list'''
    return [sub_list.mean() for sub_list in feature_list]

def make_df_feature_correlation_component(bf_feature_dict, bsc, component):
    '''Creates a melted pandas DataFrame for each feature in bf feature_dict
    IN:
        bf_feature_dict -   dictionary with auditory feature names as keys and
                            shape (200,) ndarrays quantifying the feature for each OR dict of mean feature values per time
                            BSC basis function
        bsc             -   ndarray of the Binary Sparse Coding basis function activations
        component       -   ndarray of shape (samples, participants) component activation in each sample
        n_bins          -   number of bins for each principal component
    '''
    from functools import reduce
    if list(bf_feature_dict.values())[0].shape[0] == 200:
        average_ft_dict = make_average_feature_dict(bf_feature_dict, bsc)
    else:
        average_ft_dict = bf_feature_dict
    corr_df = pd.DataFrame({ft_name: np.corrcoef(component, feature, rowvar=0)[-1, :-1]
                            for ft_name, feature in average_ft_dict.items()})
    return pd.concat([corr_df, pd.Series(np.arange(component.shape[1]), name='Participant')], axis=1)

def make_average_feature_dict(bf_feature_dict, bsc):
    '''Averages the feature lists from bf_feature_list'''
    average_ft_dict = {ft_name: average_feature_list(get_features_in_sample(bsc, feature))
           for ft_name, feature in bf_feature_dict.items()}
    return average_ft_dict

def melt_participant_feature_df(feature_df, var_name='Cluster', **kwargs):
    '''Melts feature_df and keeps Participant and PC'''
    return pd.melt(feature_df, id_vars=['Participant', 'PC'], var_name=var_name, **kwargs)

def pval_from_pearson_r(r, n=3539):
    '''returns p value for pearson correlation coefficient with Fisher transform'''
    from scipy.stats import norm
    z = (np.arctanh(r)-np.arctanh(0.))*np.sqrt(n-3)
    return 2*norm.sf(np.abs(z))
