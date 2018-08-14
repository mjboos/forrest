import joblib
import numpy as np
from forrest import encoding as enc
from forrest.preprocessing import process_subj
from forrest import features
from sklearn.linear_model import RidgeCV
import mkl
import librosa as lbr
import matplotlib.pyplot as plt

mkl.set_num_threads(4)

memory = joblib.Memory(cachedir='/data/mboos/joblib')

path_to_fg = '/data/forrest_gump/phase1'
path_to_save = '/data/mboos/encoding/fmri'

subjects=[1, 2]
scores = []

# make simple RMSE audio features
audio_rmse = memory.cache(features.extract_audio_rep('/data/mboos/audio',
             hop_length=4410, frame_length=8820,
             audio_extractor=partial(features.extract_audio, extraction_func=lbr.feature.rmse)))

# now we lag the audio featurs
audio_rmse = features.lag_features(audio_rmse)

# BlockMultiOutput fits a MultiOutputEstimator for each block of targets
# This makes the estimator n_blocks-times less memory-hungry
model = enc.BlockMultiOutput(RidgeCV(alphas=[1, 1e2, 1e3, 1e4, 1e5]), n_blocks=10, n_jobs=1)

for subj in subjects:
    pre.process_subj(subj, path_to_save, path_to_fg)
    fmri_data = joblib.load(path_to_save+'/'+'fmri_subj_{}.pkl'.format(subj), mmap_mode='c')
    predictions, model = enc.get_ridge_predictions_model(audio_rmse, fmri_data, model)

    scores_tmp = enc.score_predictions(predictions, subj)
    # we cut at zero correlation because a negative correlation is likely due to overfitting
    scores_tmp[scores_tmp<0] = 0
    scores.append(scores_tmp)

from nilearn.plotting import plot_glass_brain
vmin, vmax = np.min(scores), np.max(scores)
fig, axes = plt.subplots(2,1)

for i, ax in enumerate(axes):
    fn = 'scores_in_mni_{}.nii.gz'.format(i)
    enc.array_to_mni_map(scores[i], fn, path_to_fg)
    plot_glass_brain(fn, axes=ax, title='Correlation scores of subject {}'.format(subjects[i]))

plt.show()
