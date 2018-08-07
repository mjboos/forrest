import joblib
import numpy as np
from forrest import encoding as enc
from forrest import plotting
import preprocessing as pre
from sklearn.linear_model import RidgeCV
import mkl
from subspace import adjust_r

mkl.set_num_threads(4)

memory = joblib.Memory(cachedir='/data/mboos/joblib')

ridge_params = {
        'alphas' : [1, 1e2, 1e3, 1e4, 1e5]}

stimuli = 'logBSC_H200'
subjects=[1,2,5,6,7,8,9,11,12,14,15,16,17,18,19]

scores = []

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
