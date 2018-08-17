import matplotlib as mpl
import joblib
import numpy as np
import os
from forrest import encoding as enc
from forrest.preprocessing import process_subj
from forrest import features
from sklearn.linear_model import RidgeCV
import librosa as lbr
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial

memory = joblib.Memory(cachedir='/data/mboos/joblib')

# cache the audio extractor function so features get saved on disk after computation
features.extract_audio_rep = memory.cache(features.extract_audio_rep)
enc.get_predictions_and_fit_model = memory.cache(enc.get_predictions_and_fit_model)

path_to_fg = '/data/forrest_gump/phase1'
path_to_save = '/data/mboos/encoding/fmri'
path_to_researchcut_audio = '/data/mboos/audio'
path_to_fg_annotations = '/data/mboos/studyforrest-data-annotations'


subjects = [1]

# we use the root mean-squared error of the audio signal for prediction
audio_extractor = partial(features.extract_audio, extraction_func=lbr.feature.rmse)

# make simple RMSE audio features
audio_rmse = features.extract_audio_rep('/data/mboos/audio',
             hop_length=4410, frame_length=8820,
             audio_extractor=audio_extractor)

# now we lag the audio features
audio_rmse = features.lag_features(audio_rmse, n_samples_lag=4)

# additionally we can use the annotation of the studyforrest dataset to extract different variables, like emotional arousal
emotional_arousal = features.convert_annotation_to_TR('/data/mboos/studyforrest-data-annotations/researchcut/emotions_ao_1s_events.tsv', which_col='arousal')
emotional_arousal = features.lag_features(emotional_arousal, n_samples_lag=4)

# we also extract one-hot indicators for characters
annotation_df = pd.DataFrame.from_csv('/data/mboos/studyforrest-data-annotations/researchcut/emotions_ao_1s_events.tsv', sep='\t', index_col=None)
character_one_hot = pd.get_dummies(annotation_df, columns=['character'])
characters_and_onsets = pd.concat([annotation_df[['onset', 'duration']], character_one_hot[[col for col in character_one_hot.columns if col.startswith('character')]]], axis=1)
character_TR_ind = features.convert_annotation_to_TR(characters_and_onsets)
character_TR_ind = features.lag_features(character_TR_ind, n_samples_lag=4)

# we use RMSE as a baseline, because the loudness of the auditory stimulus will explain most variance and we want to control for that
# thus we add character and emotional features to the RMSE features and test what they explain beyond pure RMSE
feature_dict = {'RMSE' : audio_rmse,
                'character' : np.concatenate([audio_rmse, character_TR_ind], axis=1)}
                'arousal' : np.concatenate([audio_rmse, emotional_arousal], axis=1)}

# BlockMultiOutput fits a MultiOutputEstimator for each block of targets
# This needs much less memory.
model = enc.BlockMultiOutput(RidgeCV(alphas=[1e-2, 1e-1, 1, 1e2, 1e3]), n_blocks=10, n_jobs=1)

score_dicts = []

for subj in subjects:
    process_subj(subj, path_to_save, path_to_fg)
    fmri_data = joblib.load(path_to_save+'/'+'fmri_subj_{}_test.pkl'.format(subj), mmap_mode='c')

    # we will remove the first 6s because after lagging stimulus features lack the first (n_samples_lag - 1) samples
    fmri_data = fmri_data[3:]
    subj_score_dict = dict()
    for feature_name, stimulus_features in feature_dict.items():
        predictions, model = enc.get_predictions_and_fit_model(stimulus_features, fmri_data, model)

        scores_tmp = enc.r_score_predictions(predictions, fmri_data)
        # we set all correlations to zero that are non-significant after false-discovery rate correction
        scores_tmp[np.logical_not(enc.adjust_r(scores_tmp, n=predictions.shape[0])[0])] = 0
        subj_score_dict[feature_name] = scores_tmp
    score_dicts.append(subj_score_dict)

score_differences = []
for i, score_dict in enumerate(score_dicts):
    score_diff_dict = {}
    for feature_name in score_dict.keys():
        if feature_name == 'RMSE':
            continue
        score_diff_dict[feature_name] = score_dict[feature_name] - score_dict['RMSE']
    score_differences.append(score_diff_dict)

# plot scores for the first subject
from nilearn.plotting import plot_glass_brain
fig, axes = plt.subplots(len(feature_dict.keys()), figsize=(12,20))

vmin, vmax = np.min(score_dicts[0].values()), np.max(score_dicts[0].values())
subj_mask_path = os.path.join(path_to_fg, 'sub{0:03d}'.format(subjects[0]), 'templates', 'bold7Tp1', 'in_grpbold7Tp1', 'brain_mask.nii.gz')
for ax_ft, feature_name in zip(axes, feature_dict.keys()):
    fn = 'scores_{}_in_mni_{}.nii.gz'.format(feature_name, subjects[0])
    # convert array of scores to mni map and save as fn
    enc.array_to_mni_map(score_dicts[i][feature_name], fn, path_to_fg, subj_mask_path)
    plot_glass_brain(fn, axes=ax_ft, title='{}'.format(feature_name),
                     vmin=vmin, vmax=vmax, colorbar=True, threshold=0.1)

fig.savefig('scores.png')

fig, axes = plt.subplots(len(score_differences[0].keys()), figsize=(12,20))

subj_mask_path = os.path.join(path_to_fg, 'sub{0:03d}'.format(subjects[0]), 'templates', 'bold7Tp1', 'in_grpbold7Tp1', 'brain_mask.nii.gz')
for ax_ft, feature_name in zip(axes, score_differences[0].keys()):
    fn = 'score_diffs_{}_in_mni_{}.nii.gz'.format(feature_name, subjects[0])
    # convert array of scores to mni map and save as fn
    enc.array_to_mni_map(score_differences[i][feature_name], fn, path_to_fg, subj_mask_path)
    plot_glass_brain(fn, axes=ax_ft, title='{}'.format(feature_name),
                     colorbar=True, threshold=1e-5)

fig.savefig('score_differences.png')
