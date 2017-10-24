from __future__ import division
import numpy as np
import os
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from nilearn.masking import unmask, apply_mask
from nilearn.image import threshold_img

def save_map(scores, threshold=None, model='logBSC_H200', name='mean', mask='group_temporal_lobe_mask.nii.gz',
             folder='/home/mboos/encoding_paper', **kwargs):
    '''Saves brainmap as nifti file'''
    if threshold is not None:
        scores[scores<threshold] = 0
    unmasked = unmask(scores, mask)
    fname = os.path.join(folder, 'maps', '{}_{}_map.nii.gz'.format(name, model))
    unmasked.to_filename(fname)

def glassbrain_contours(map_dict, colors=['r', 'g', 'b', 'cyan', 'magenta', 'k'], cutoff=[90]):
    import nilearn.image as img
    from nilearn import plotting
    img_dict = { label : img.load_img(filename) for label, filename in map_dict.iteritems()}
    display = plotting.plot_glass_brain(None)
    for color, (label, image) in zip(colors, img_dict.iteritems()):
        display.add_contours(image, levels=cutoff, colors=color)
    return display

def plot_scores(scores, threshold=0.01, coords=None, folder='/home/mboos/encoding_paper',
                   data_path='/data/forrest_gump/phase1', mask='group_temporal_lobe_mask.nii.gz', **kwargs):
    '''plots subject scoremap using nilearn and returns display object'''
    background_img = os.path.join(data_path, 'templates','grpbold7Tp1/brain.nii.gz')
    scores = scores.copy()
    scores[scores<threshold] = 0
    unmasked = unmask(scores, mask)
    display = plot_stat_map(
                    unmasked, cut_coords=coords, bg_img=background_img,
                    dim=-1, aspect=1.25,
                    threshold=1e-6, **kwargs)
    fig = plt.gcf()
    fig.set_size_inches(12, 4)
    return display

def plot_diff_scores(scores, threshold=0.01, coords=None, folder='/home/mboos/encoding_paper',
                        data_path='/data/forrest_gump/phase1', mask='group_temporal_lobe_mask.nii.gz', **kwargs):
    '''plots subject scoremap using nilearn and returns display object'''
    background_img = os.path.join(data_path, 'templates','grpbold7Tp1/brain.nii.gz')
    scores = scores.copy()
    scores[np.abs(scores)<threshold] = 0
    unmasked = unmask(scores, mask)
    display = plot_stat_map(
                    unmasked, cut_coords=coords, bg_img=background_img,
                    dim=-1, aspect=1.25,
                    threshold=1e-6, **kwargs)
    fig = plt.gcf()
    fig.set_size_inches(12, 4)
    return display

def group_stability_plots(predictions, model='logBSC_H200', folder='/home/mboos/encoding_paper/plots'):
    '''Group stability plots for model and predictions'''
    import matplotlib.pyplot as plt
    import seaborn as sns
    from os.path import join
    import pandas as pd
    r_per_comp = dict()
    for nr, comp_act in enumerate(predictions.T):
        correlations = np.abs(np.corrcoef(comp_act))
        r_per_comp['PC {}'.format(nr+1)] = correlations[np.triu_indices_from(correlations)]
        fig, ax = plt.subplots()
        im= ax.imshow(correlations, origin='lower', aspect='auto',
                  interpolation='nearest', vmin=0, vmax=1, cmap='viridis')
        fig.colorbar(im)
        plt.xticks(np.arange(comp_act.shape[0]), np.arange(comp_act.shape[0])+1)
        plt.yticks(np.arange(comp_act.shape[0]), np.arange(comp_act.shape[0])+1)
        plt.xlabel('Participant')
        plt.ylabel('Participant')
        fig.savefig(join(folder, '{}_subj_r_comp_{}.svg'.format(model, nr)))
        plt.close()
    comp_df = pd.melt(pd.DataFrame(data=r_per_comp),
            value_vars=r_per_comp.keys(),
            var_name='Principal Component', value_name='Correlations between participants')
    fig = plt.figure()
    ax = sns.boxplot(data=comp_df, x='Principal Component', y='Correlations between participants', order=sorted(r_per_comp.keys()))
    fig.savefig(join(folder, 'inter_subject_corrs_comp_{}.svg'.format(model)))
    plt.close()

def feature_correlation_plots(activations, stimuli='logMFS', model='logBSC_H200', **fsi_args):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    fsi_dict = dict()
    stimuli_ft = joblib.load('/data/mboos/encoding/stimulus/preprocessed/{}_stimuli.pkl'.format(stimuli))
    pc_names = ['PC {}'.format(pc_nr+1) for pc_nr in xrange(activations.shape[-1])]
    for pc_name, pc in izip(pc_names, activations.T):
        fsi_high, fsi_low = zip(*[feature_corrs(pc_for_subj, stimuli_ft, **fsi_args)
                           for pc_for_subj in pc])
        fsi_dict[(pc_name, 'high')] = fsi_high
        fsi_dict[(pc_name, 'low')] = fsi_low
    fsi_df = pd.melt(pd.DataFrame(data=fsi_dict),
            value_vars=pc_names + ['low', 'high'],
            var_name=['Principal component', 'activation'], value_name='Mean feature correlation')
    fig = plt.figure()
    ax = sns.boxplot(data=fsi_df, x='activation', hue='Principal component', y='Mean feature correlation')
    fig.savefig('plots/Feature_corr_{}_using_{}.svg'.format(model, stimuli))
    plt.close()

def feature_selectivity_plots_over_n_stim(activations, stimuli='logMFS', model='logBSC_H200', **fsi_args):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    fsi_list = []
    n_stim_grid = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    stimuli_ft = joblib.load('/data/mboos/encoding/stimulus/preprocessed/{}_stimuli.pkl'.format(stimuli))
    pc_names = ['PC {}'.format(pc_nr+1) for pc_nr in xrange(activations.shape[-1])]
    for pc_name, pc in izip(pc_names, activations.T):
        fsi = [[np.mean(feature_selectivity_index(pc_for_subj, stimuli_ft, n_stim=n_stim_i, **fsi_args))
                           for n_stim_i in n_stim_grid] for pc_for_subj in pc]
        fsi_list.append(fsi)
    fsi_list = np.array(fsi_list)
    ax = sns.tsplot(data=fsi_list, time='Time',
            value='Feature Selectivity Index', condition='Principal component')
    return ax
#plt.savefig('plots/FSI_time_{}_using_{}.svg'.format(model, stimuli))
#    plt.close()


def feature_selectivity_plots_per_time(activations, n_ts=3, stimuli='logMFS', model='logBSC_H200', **fsi_args):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    fsi_dict = dict()
    stimuli_ft = joblib.load('/data/mboos/encoding/stimulus/preprocessed/{}_stimuli.pkl'.format(stimuli))
    stimuli_ft = np.reshape(stimuli_ft, (stimuli_ft.shape[0], n_ts, -1))
    pc_names = ['PC {}'.format(pc_nr+1) for pc_nr in xrange(activations.shape[-1])]
    for pc_name, pc in izip(pc_names, activations.T):
        fsi = [[np.mean(feature_selectivity_index(pc_for_subj, stimuli_ft[:,t,:], **fsi_args))
                           for pc_for_subj in pc] for t in xrange(n_ts)]
        for t in xrange(n_ts):
            fsi_dict[(pc_name, t)] = fsi[t]
    fsi_df = pd.melt(pd.DataFrame(data=fsi_dict),
            value_vars=pc_names+[str(i) for i in xrange(n_ts)],
            var_name=['Principal component', 'Time'], value_name='Feature Selectivity Index')
    g = sns.FacetGrid(fsi_df, col='Principal component')
    (g.map(sns.boxplot, 'Time', 'Feature Selectivity Index')).despine(left=True)
#    ax = sns.tsplot(data=fsi_df, time='Time',
#            value='Feature Selectivity Index', condition='Principal component', interpolate=False)
            #hue_order=pc_names, order=[str(i) for i in xrange(n_ts)])
    plt.savefig('plots/FSI_time_{}_using_{}.svg'.format(model, stimuli))
    plt.close()

def feature_selectivity_plots_mean(activations, stimuli='logMFS', model='logBSC_H200', **fsi_args):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    fsi_dict = dict()
    stimuli_ft = joblib.load('/data/mboos/encoding/stimulus/preprocessed/{}_stimuli.pkl'.format(stimuli))
    pc_names = ['PC {}'.format(pc_nr+1) for pc_nr in xrange(activations.shape[-1])]
    for pc_name, pc in izip(pc_names, activations.T):
        fsi = np.array([np.mean(feature_selectivity_index(pc_for_subj, stimuli_ft, **fsi_args))
                           for pc_for_subj in pc])
        fsi_dict[pc_name] = fsi
    fsi_df = pd.melt(pd.DataFrame(data=fsi_dict),
            value_vars=pc_names,
            var_name='Principal component', value_name='Feature Selectivity Index')
    fig = plt.figure()
    sns.boxplot(data=fsi_df, x='Principal component', y='Feature Selectivity Index')
    sns.swarmplot(data=fsi_df, x='Principal component', y='Feature Selectivity Index', color='.25')
    fig.savefig('plots/FSI_mean_{}_using_{}.svg'.format(model, stimuli))
    plt.close()

def feature_selectivity_plots(activations, stimuli='logMFS', model='logBSC_H200', **fsi_args):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    fsi_dict = dict()
    stimuli_ft = joblib.load('/data/mboos/encoding/stimulus/preprocessed/{}_stimuli.pkl'.format(stimuli))
    pc_names = ['PC {}'.format(pc_nr+1) for pc_nr in xrange(activations.shape[-1])]
    for pc_name, pc in izip(pc_names, activations.T):
        fsi_high, fsi_low = zip(*[feature_selectivity_index(pc_for_subj, stimuli_ft, **fsi_args)
                           for pc_for_subj in pc])
        fsi_dict[(pc_name, 'high')] = fsi_high
        fsi_dict[(pc_name, 'low')] = fsi_low
    fsi_df = pd.melt(pd.DataFrame(data=fsi_dict),
            value_vars=pc_names + ['low', 'high'],
            var_name=['Principal component', 'activation'], value_name='Feature Selectivity Index')
    fig = plt.figure()
    ax = sns.boxplot(data=fsi_df, x='activation', hue='Principal component', y='Feature Selectivity Index')
    fig.savefig('plots/FSI_{}_using_{}.svg'.format(model, stimuli))
    plt.close()

def make_pc_plots(pcs, model, folder='/home/mboos/encoding_paper/plots',
                  subjects=[1,2,5,6,7,8,9,11,12,14,15,16,17,18,19],
                  mean_map=False, plot_args={}, pca_analysis_args={}):
    from os.path import join
    mean_scores_list = []
    for pc, scores_list in enumerate(get_pc_scores(pcs, model, subjects=subjects, **pca_analysis_args)):
        for subj, scores in izip(subjects, scores_list):
            display = plot_scores(scores, **plot_args)
            display.savefig(join(folder, 'PC_{}_scores_subj_{}_model_{}.png'.format(pc+1, subj, model)))
            display.close()
        mean_scores = np.concatenate([scores[:, None] for scores in scores_list], axis=1).mean(axis=1)
        display = plot_scores(mean_scores, **plot_args)
        display.savefig(join(folder, 'PC_{}_mean_scores_model_{}.png'.format(pc+1, model)))
        display.close()
        mean_scores_list.append(mean_scores)
        if mean_map:
            save_map(mean_scores, model=model, name='mean_PC_{}'.format(pc+1))


