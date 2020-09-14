import coef_helper_functions as cfh
import auditory_feature_helpers as aud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
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

