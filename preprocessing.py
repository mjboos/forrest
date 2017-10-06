from __future__ import division
import sys
import os
import glob
import mvpa2.suite as mvpa
import numpy as np
import joblib
from sklearn.cross_validation import KFold
from nilearn.masking import unmask
from nibabel import save

#TODO: code review
#TODO: check if these functions work
#TODO: re-factor split fmri with split stimulus?

def create_lagged_features(patches):
    '''Creates lagged features'''
    patches_pre = patches[:int(np.floor(patches.shape[0] / 20)*20), :]
    # shape of TR samples
    # note: column ordering is now oldest --> newest in steps of 200
    patches_pre = np.reshape(patches_pre, (-1, 200*20))

    strides = (patches_pre.strides[0],) + patches_pre.strides

    # rolling window of length 3 samples
    shape = (patches_pre.shape[0] - 3 + 1, 3, patches_pre.shape[1])

    patches_pre = np.lib.stride_tricks.as_strided(patches_pre[::-1,:].copy(),
                                              shape=shape,
                                              strides=strides)[::-1, :, :]
    patches_pre = np.reshape(patches_pre, (patches_pre.shape[0], -1))

    # we kick out the most recent sample
#    patches_pre = patches_pre[:, :-4000]
    return patches_pre


def preprocess_and_tmp_save_fmri(datapath, task, subj, model, scratch_path):
    '''preprocesses one subject from Forrest Gump
    aligns to group template
    run-wise linear de-trending and z-scoring'''
    dhandle = mvpa.OpenFMRIDataset(datapath)
    #mask_fname = os.path.join('/home','mboos','SpeechEncoding','temporal_lobe_mask_brain_subj' + str(subj) + 'bold.nii.gz')

    flavor = 'dico_bold7Tp1_to_subjbold7Tp1'
    group_brain_mask = 'brainmask_group_template.nii.gz'
    group_temp_lobe_mask = 'group_temporal_lobe_mask.nii.gz'
    mask_fname = os.path.join(datapath, 'sub{0:03d}'.format(subj), 'templates', 'bold7Tp1', 'brain_mask.nii.gz')
    for run_id in dhandle.get_task_bold_run_ids(task)[subj]:
        run_ds = dhandle.get_bold_run_dataset(subj,task,run_id,chunks=run_id-1,mask=mask_fname,flavor=flavor)
        filename = 'brain_subj_{}_run_{}.nii.gz'.format(subj, run_id)
        tmp_path = scratch_path + filename
        save(unmask(run_ds.samples.astype('float32'), mask_fname), tmp_path)
        os.system('fsl5.0-applywarp -i {0} -o {1} -r /data/forrest_gump/phase1/templates/grpbold7Tp1/brain.nii.gz -w /data/forrest_gump/phase1/sub{2:03}/templates/bold7Tp1/in_grpbold7Tp1/subj2tmpl_warp.nii.gz --interp=nn'.format(tmp_path, scratch_path+'group_'+filename,subj))
        os.remove(tmp_path)
        run_ds = mvpa.fmri_dataset(scratch_path+'group_'+filename, mask=group_temp_lobe_mask, chunks=run_id-1)
        mvpa.poly_detrend(run_ds, polyord=1)
        mvpa.zscore(run_ds)
        joblib.dump(run_ds.samples.astype('float32'),
                    scratch_path+'brain_subj_{}_run_{}.pkl'.format(subj, run_id))
        os.remove(scratch_path+'group_'+filename)
    return run_ds.samples.shape[1]

def split_fmri_memmap(file_mmaps, cv):
    '''
    Generator function for voxel-splits
    IN:
    file_mmaps - list of memmapped runs
    cv         - KFold object with splits
    OUT:
    fmri_data split by voxels specified in cv
    '''
    duration = np.array([902,882,876,976,924,878,1084,676])

    # i did not kick out the first/last 4 samples per run yet
    slice_nr_per_run = [dur/2 for dur in duration]

    # use broadcasting to get indices to delete around the borders
    idx_borders = np.cumsum(slice_nr_per_run[:-1])[:,np.newaxis] + \
                  np.arange(-4,4)[np.newaxis,:]

    for i, (_, voxels) in enumerate(cv):
        fmri_data = np.vstack([fl_mm[:, voxels] for fl_mm in file_mmaps])
        fmri_data = np.delete(fmri_data, idx_borders, axis=0)
        # and we're going to remove the last fmri slice
        # since it does not correspond to a movie part anymore
        fmri_data = fmri_data[:-1, :]

        # shape of TR samples
        fmri_data = fmri_data[3:]
        yield fmri_data

def process_subj(subj, scratch_path='/data/mboos/tmp/',
                 save_path='/data/mboos/encoding/fmri/',
                 datapath='/data/forrest_gump/phase1'):
    '''this function preprocesses subj run-wise and then saves it in voxel-splits'''
    task = 1
    model = 1

    # preprocess participant and save data temporary for each run
    voxel_nr = preprocess_and_tmp_save_fmri(datapath, task, subj, model, scratch_path)
    cv = KFold(n=voxel_nr, n_folds=10)
    file_mmaps = [joblib.load(scratch_path+'brain_subj_{}_run_{}.pkl'.format(subj, run_id), mmap_mode='r') for run_id in xrange(1, 9)]

    # instead of splitting in time (per run), split in space (voxel-chunks) 
    # so voxel-wise encoding needs less memory
    for i, fmri_data in enumerate(split_fmri_memmap(file_mmaps, cv)):
            joblib.dump(fmri_data,
                 save_path+'fmri_subj_{}_split_{}.pkl'.format(subj, i))
    # delete temporary saved run data
    for run_id in xrange(1,9):
        os.remove(scratch_path+'brain_subj_{}_run_{}.pkl'.format(subj, run_id))
