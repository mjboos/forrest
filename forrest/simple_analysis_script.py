import joblib
import numpy as np
from forrest import encoding as enc
from forrest.preprocessing import process_subj
from forrest import features
from sklearn.linear_model import RidgeCV
import mkl
import librosa as lbr

mkl.set_num_threads(4)

memory = joblib.Memory(cachedir='/data/mboos/joblib')

ridge_params = {
        'alphas' : [1, 1e2, 1e3, 1e4, 1e5]}

subjects=[1]

scores = []

# make simple RMSE audio features
audio_rmse = features.extract_audio_rep('/data/mboos/audio', hop_length=4410, frame_length=8820, audio_extractor=partial(features.extract_audio, extraction_func=lbr.feature.rmse))
audio_rmse = features.lag_features(audio_rmse)

for subj in subjects:
    pre.process_subj(subj, group_mask='/home/mboos/encoding_paper/group_temporal_lobe_mask.nii.gz')
    predictions, model = enc.encoding_for_subject(subj, stimulus_name=stimuli, memory=memory, **ridge_params)
    # only use significant fdr corrected scores for average, every other score is used as 0
    scores_tmp = enc.score_predictions(predictions, subj)
    scores_tmp[np.logical_not(adjust_r(scores_tmp)[0])] = 0
    scores.append(scores_tmp)

scores = np.mean(scores, axis=0)

display = plotting.plot_scores(scores, mask='/home/mboos/encoding_paper/group_temporal_lobe_mask.nii.gz')
display.savefig('mean_bsc_highest_score_slice.svg')
plotting.arr_to_mni_map(scores, 'mean_bsc_mni.nii.gz', mask='/home/mboos/encoding_paper/group_temporal_lobe_mask.nii.gz')
