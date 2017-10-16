import joblib
import numpy as np
import encoding as enc
import preprocessing as pre
from sklearn.linear_model import RidgeCV
import mkl

mkl.set_num_threads(4)

memory = joblib.Memory(cachedir='/data/mboos/joblib')

ridge_params = {
        'alphas' : [1, 1e2, 1e3, 1e4, 1e5]}

stimuli = 'logBSC_H200'
subjects=[1,2,5,6,7,8,9,11,12,14,15,16,17,18,19]

for subj in subjects[1:]:
    pre.process_subj(subj, group_mask='/home/mboos/encoding_paper/group_temporal_lobe_mask.nii.gz')
    enc.encoding_for_subject(subj, stimulus_name=stimuli, memory=memory, **ridge_params)
    break
