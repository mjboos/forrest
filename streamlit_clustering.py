from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import seaborn
import seaborn as sns
import auditory_feature_helpers as aud
import coef_helper_functions as cfh
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm 
import coef_helper_functions as cfh
import auditory_feature_helpers as aud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import streamlit as st

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


Ws = joblib.load('logBSC_H200_W.pkl')
def get_Z(bsc_distmats):
    from scipy.spatial.distance import squareform

    dense_distance = squareform(bsc_distmats, checks=False)
    Z = linkage(dense_distance, 'ward', optimal_ordering=True)
    return Z


def distance_from_corrmat(corrmat):
    '''Transforms a correlation matrix into a distance matrix'''
    revcorrmat = -1*corrmat
    distmat = (revcorrmat - revcorrmat.min()) / (revcorrmat.max() - revcorrmat.min())
    return distmat

def compute_clstr_mean(ft, labels):
    return np.array([ft[labels==lbl].mean() for lbl in np.unique(labels)])

@st.cache
def compute_for_cluster(n_clusters):
    Ws = joblib.load('logBSC_H200_W.pkl')
    magnitude = Ws.mean(axis=(1,2))
    cluster_idx = joblib.load('cluster_idx.pkl')
    cluster_means = joblib.load('cluster_means.pkl')
    mps = np.concatenate([np.fft.fftshift(np.fft.rfft2(w))[None] for w in Ws], axis=0)
    mps_stft = np.concatenate([aud.compute_MPS_from_STFT(BF)[None] for BF in Ws], axis=0)
    import librosa as lbr
    fft_freq = lbr.core.fft_frequencies(sr=44100, n_fft=882)[:161]
    # in cycles/KHz
    mps_freqs = np.fft.rfftfreq(fft_freq.shape[0], 50)*1000
    best_time, best_freq = zip(*[np.unravel_index(np.argmax(np.abs(mps)[:,5:][i]), (5,25)) for i in range(200)])
    best_time = np.array(best_time)
    best_freq = np.array(best_freq)
    mean_time = [np.mean(best_time[cluster_idx==i]) for i in range(10)]
    mean_freq = [np.mean(best_freq[cluster_idx==i]) for i in range(10)]
    pcs = joblib.load('pcs.pkl')
    pcs[:,:,0] *= -1
    pcs[:,:,1] *= -1
    pcs[:,:,2] *= -1
    mean_pcs = pcs.mean(axis=0)[...,:3]
    bsc = joblib.load('../semisupervised/logBSC_H200_stimuli.pkl')
    times, freqs = aud.compute_mps_time_and_freq_labels()
    best_time_stft, best_freq_stft = zip(*[np.unravel_index(np.argmax(mps_stft[:,5:][i]), (5, 81)) for i in range(200)])
    best_time_stft = np.array(best_time_stft)
    best_freq_stft = np.array(best_freq_stft)
    best_freq_stft = freqs[best_freq_stft]
    best_time_stft = times[5:][best_time_stft]
    feature_dict = {'sBMF': best_freq_stft, 'tBMF': best_time_stft, 'magnitude': magnitude}
    average_feature_dict = aud.make_average_feature_dict(feature_dict, bsc)
    feature_corr_df = aud.make_df_feature_correlation(feature_dict, bsc, pcs[...,:3])
    melted_feature_df = aud.melt_participant_feature_df(feature_corr_df, var_name='Feature', value_name='Correlation')

    bscs = joblib.load('../semisupervised/logBSC_H200_stimuli.pkl')
    bscs = cfh.reshape_bsc(bscs)
    nonlagged_bscs = bscs[:,20:40,:]
    connectivity_bsc = np.corrcoef(bscs.mean(axis=1), rowvar=0)
    summed_bscs = bscs.sum(axis=1)
    speech_overlap = joblib.load('speech_overlap.pkl')
    contains_speech = speech_overlap > 5.5
    contains_less_speech = speech_overlap < 5.0
    corr_speech_BF = np.array([np.corrcoef(summed_bscs[:,i], speech_overlap)[0,1] for i in range(200)])
    sensitivity_to_speech = summed_bscs[contains_speech].sum(axis=0) / summed_bscs[contains_less_speech].sum(axis=0)
    bscs_per_time = {'100ms': np.corrcoef(np.reshape(bscs, (-1,200)), rowvar=0), '500ms': np.corrcoef(np.reshape(bscs, (12*3539, -1 ,200)).mean(axis=-2), rowvar=0), '200ms': np.corrcoef(np.reshape(bscs, (10*3539, -1 ,200)).mean(axis=-2), rowvar=0),
                     '1000ms': np.corrcoef(np.reshape(bscs, (6*3539, -1 ,200)).mean(axis=-2), rowvar=0), '2000ms': np.corrcoef(np.reshape(bscs, (3*3539, -1 ,200)).mean(axis=-2), rowvar=0),
                    '6000ms': np.corrcoef(np.reshape(bscs, (3539, -1 ,200)).mean(axis=-2), rowvar=0)}
    bscs_per_time_nonlagged = {'100ms': np.corrcoef(np.reshape(nonlagged_bscs, (-1,200)), rowvar=0), '200ms': np.corrcoef(np.reshape(nonlagged_bscs, (10*3539, -1 ,200)).mean(axis=-2), rowvar=0),
                               '500ms': np.corrcoef(np.reshape(nonlagged_bscs, (4*3539, -1 ,200)).mean(axis=-2), rowvar=0),
                     '1000ms': np.corrcoef(np.reshape(nonlagged_bscs, (2*3539, -1 ,200)).mean(axis=-2), rowvar=0), '2000ms': np.corrcoef(np.reshape(nonlagged_bscs, (3539, -1 ,200)).mean(axis=-2), rowvar=0)}

    speech_correlation = speech_overlap

    separability = joblib.load('separability.pkl')



    bscs_per_time_distmats = {time: distance_from_corrmat(corrmat) for time, corrmat in bscs_per_time.items()}

    Z = get_Z(bscs_per_time_distmats['500ms'])
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')


    cluster_means = {i: bscs.mean(axis=1)[..., labels==i].mean(axis=-1) for i in np.unique(labels)}

    cluster_means = np.vstack([cluster_means[i] for i in np.unique(labels)])

    cluster_features = {ft: compute_clstr_mean(feature_dict[ft], labels) for ft in feature_dict}

#    full_cluster_pcs_corr = np.array([[[np.corrcoef(cluster_mean, pcs[i, :, pc])[0,1] for i in range(15)]
#                              for cluster_mean in cluster_means]
#                              for pc in range(3)])
    full_cluster_pcs_corr = []
    for pc in range(3):
        full_cluster_pcs_corr.append([])
        for cluster_mean in cluster_means:
            full_cluster_pcs_corr[-1].append([np.corrcoef(cluster_mean, pcs[i, :, pc])[0,1] for i in range(15)])
    full_cluster_pcs_corr = np.array(full_cluster_pcs_corr)

    speech_corrs = np.array([np.corrcoef(speech_correlation, clstr)[0,1]
                    for clstr in cluster_means])
    separability_cluster = [separability[labels==lbl].mean()
                    for lbl in np.unique(labels)]
    mps_sep = np.abs(mps).max(axis=(1, 2))/np.abs(mps).sum(axis=(1,2))
    mps_sep_clstrs = [mps_sep[labels==lbl].mean()
                      for lbl in np.unique(labels)]



    clstrs_full = pd.DataFrame({'Cluster': labels,
                                'Magnitude':feature_dict['magnitude'],
                                'tBMF': feature_dict['tBMF'],
                                'sBMF': feature_dict['sBMF'],
                                'Speech correlation': speech_corrs[labels-1],
                                'Separability': separability,
                                'MPS Separability': mps_sep})


    # In[12]:


    pc_corr_clstrs = pd.DataFrame({'Correlation': full_cluster_pcs_corr.flatten(),
                                   'PC': np.repeat([1, 2, 3], 15*n_clusters),
                                   'Cluster': np.tile(np.repeat(np.arange(1, n_clusters+1), 15), 3),
                                   'Participant': np.tile(np.arange(1, 16), 3*n_clusters),
                                   'Magnitude': np.tile(np.repeat(cluster_features['magnitude'], 15), 3),
                                   'tBMF': np.tile(np.repeat(cluster_features['tBMF'], 15), 3),
                                   'sBMF': np.tile(np.repeat(cluster_features['sBMF'], 15), 3),
                                   'Speech correlation': np.tile(np.repeat(speech_corrs, 15), 3),
                                   'Separability': np.tile(np.repeat(separability_cluster, 15), 3),
                                   'MPS Separability': np.tile(np.repeat(mps_sep_clstrs, 15), 3) 
                                  })
    return labels, clstrs_full, pc_corr_clstrs




plt.style.use('mb')

plt.rcParams.update({'font.size': 15})


# In[4]:


from librosa import mel_frequencies
mel_freqs = mel_frequencies(48, fmax=8000)

def plot_Ws(Ws, corrs=None, vmax=None, vmin=None):
    if vmin is None or vmax is None:
        vmin = Ws.min()
        vmax = Ws.max()
    n_rows = np.ceil(Ws.shape[0] / 5).astype('int')
    fig, axes = plt.subplots(n_rows, 5, figsize=(20, n_rows * 3.75), constrained_layout=True)
    axes = axes.flatten()
    for n in range(Ws.shape[0]):
        mappable = axes[n].imshow(Ws[n].T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(mappable)
    if corrs:
        fig.suptitle('{0:.2f}'.format(corrs))
    return fig


def plot_Ws_more_info(Ws, data=None, vmax=None, vmin=None):
    if vmin is None or vmax is None:
        vmin = Ws.min()
        vmax = Ws.max()
    n_plots = Ws.shape[0]+1
    n_rows = np.ceil(n_plots / 5).astype('int')
    fig, axes = plt.subplots(n_rows, 5, figsize=(20, n_rows * 3), constrained_layout=True)
    axes = axes.flatten()
    for n in range(Ws.shape[0]):
        mappable = axes[n].imshow(Ws[n].T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[n+1].plot(mel_freqs, Ws.mean(axis=(0, 1)))
    axes[n+1].set_xscale('log')
    axes[n+1].set_xticks([250, 500, 1000, 2000, 4000, 8000])
    axes[n+1].set_xticklabels(['250', '500', '1k', '2k', '4k', '8k'], rotation=90)
    plt.colorbar(mappable)
    if data is not None:
        corrmean = data['Correlation'].mean()
        speech = data['Speech correlation'].iloc[0]
        sep = data['Separability'].iloc[0]
        fig.suptitle('r={0:.2f} Speech r={1:.2f} Separability={2:.2f}'.format(corrmean, speech, sep))
    return fig

def plot_cluster(Ws, data, vmax=None, vmin=None, max_plot_figs=10, seed=None, shuffle=True):
    # TODO: change value plot
    # TODO: tick size and axes labels
    if vmin is None or vmax is None:
        vmin = Ws.min()
        vmax = Ws.max()
    if max_plot_figs is None:
        max_plot_figs = Ws.shape[0]
    n_plots = min(Ws.shape[0], max_plot_figs)+4
    n_rows = np.ceil(n_plots / 5).astype('int')
    if seed is not None:
        np.random.seed(seed)
    if shuffle:
        w_order = np.arange(Ws.shape[0])
        np.random.shuffle(w_order)
        Ws = Ws[w_order]
    fig, axes = plt.subplots(n_rows, 5, figsize=(23.5, n_rows * 3.4), constrained_layout=True)
    axes = axes.flatten()
    for n in range(min(max_plot_figs, Ws.shape[0])):
        mappable = axes[n].imshow(Ws[n].T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[n].set_yticks([2, 8, 16, 27, 37, 47])
        axes[n].set_yticklabels(['128', '513', '1028', '2129', '4127', '8000'])
        axes[n].set_xticks([0, 4, 9])
        axes[n].set_xticklabels(['10', '50', '100'])
        if n % 5 == 0:
            axes[n].set_ylabel('Hz')
    plt.colorbar(mappable)
    corrmean = [data['Correlation'][data['PC']==pc] for pc in range(1, 4)]
    axes[n+1].boxplot(corrmean)
    axes[n+1].set_xlabel('PC')
    axes[n+1].set_ylabel('Correlation')
    axes[n+1].set_title('Correlation with PCs')
    # TODO: maybe take values across all bf in cluster
    axes[n+2].barh(np.arange(6), [data['Speech correlation'].iloc[0], data['Separability'].iloc[0], data['MPS Separability'].iloc[0],
                                  data['tBMF'].iloc[0], data['sBMF'].iloc[0], data['Magnitude'].iloc[0]])
    axes[n+2].set_yticks(np.arange(6))
    axes[n+2].set_yticklabels(['Speech correlation', 'Separability', 'MPS Separability', 'tBMF', 'sBMF', 'Magnitude'])
    axes[n+2].set_xlabel('Average value')
    axes[n+2].set_title('Cluster characterization')
    mean_freqs = Ws.mean(axis=(0, 1))
    axes[n+3].plot(mel_freqs, mean_freqs)
    std_freqs = Ws.std(axis=(0,1))
    axes[n+3].fill_between(mel_freqs, mean_freqs-std_freqs, mean_freqs+std_freqs, alpha=0.3)
    axes[n+3].set_xscale('log')
    axes[n+3].set_xticks([250, 500, 1000, 2000, 4000, 8000])
    axes[n+3].set_xticklabels(['250', '500', '1000', '2000', '4000', '8000'], rotation=45)
    axes[n+3].set_xlabel('Frequency (Hz)')
    axes[n+3].set_title('Average spectral profile')
    mean_temp = Ws.mean(axis=(0, 2))
    std_temp = Ws.std(axis=(0, 2))
    axes[n+4].plot(mean_temp)
    axes[n+4].fill_between(np.arange(10), mean_temp-std_temp, mean_temp+std_temp, alpha=0.3)
    axes[n+4].set_xticks([0, 4, 9])
    axes[n+4].set_xticklabels(['10', '50', '100'])
    axes[n+4].set_xlabel('Time (ms)')
    axes[n+4].set_title('Average temporal profile')
    for i in range(len(axes)-(n+5)):
        axes[i+n+5].axis('off')
    return fig


def summarize_PCs(data, vars_to_use=None):
    if vars_to_use is None:
        vars_to_use = ['Cluster']
    df_list = []
    for pc, df in data.groupby('PC'):
        df = df.rename(columns={'Correlation': 'PC {}'.format(pc)})
        df_list.append(df[['PC {}'.format(pc)]+vars_to_use])
    df_list[0]['PC 2'] = df_list[1]['PC 2'].values
    df_list[0]['PC 3'] = df_list[2]['PC 3'].values
    return df_list[0]


def melt_df(data, vars_to_use=None):
    if vars_to_use is None:
        vars_to_use = ['PC 1', 'PC 2', 'PC 3']
    return pd.melt(data, id_vars=['Cluster'], value_vars=vars_to_use, var_name='Correlation with', value_name='Correlation')

n_clusters = st.sidebar.number_input('Indicate the number of clusters to use:', min_value=2, max_value=200, value=5)
cluster_idx = st.sidebar.number_input('Which cluster to display:', min_value=1, max_value=n_clusters, value=1)


labels, clstrs_full, pc_corr_clstrs  = compute_for_cluster(n_clusters)

st.header('Exploring the hierarchy of cortical auditory processing')
melted_df = melt_df(summarize_PCs(pc_corr_clstrs))
st.subheader('Characterization of clusters')
fig1=sns.catplot(data=melted_df, x='Cluster', y='Correlation', col='Correlation with', kind='box', sharey=True)
st.pyplot(fig1)
melted_clstr_char = pd.melt(clstrs_full, id_vars=['Cluster'], var_name='Feature')
fig2=sns.catplot(data=melted_clstr_char, x='Cluster', y='value', col='Feature', kind='box', sharey=False, col_wrap=3)
st.pyplot(fig2)

st.subheader('Cluster {} exemplary basis functions and key attributes'.format(cluster_idx))
fig3 = plot_cluster(Ws[labels==cluster_idx],
                   pc_corr_clstrs[pc_corr_clstrs['Cluster']==cluster_idx],
                   vmin=Ws.min(), vmax=Ws.max())
st.pyplot(fig3)
