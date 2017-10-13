from __future__ import division
import sys
import os
import glob
import mvpa2.suite as mvpa
import numpy as np
import joblib
from nilearn.masking import unmask
from nibabel import save

#TODO: review preprocessing

def preprocess_and_tmp_save_fmri(datapath, task, subj, model, scratch_path, group_mask='group_temporal_lobe_mask.nii.gz'):
    '''preprocesses one subject from Forrest Gump
    aligns to group template
    run-wise linear de-trending and z-scoring'''
    from nipype.interfaces import fsl
    dhandle = mvpa.OpenFMRIDataset(datapath)
    #mask_fname = os.path.join('/home','mboos','SpeechEncoding','temporal_lobe_mask_brain_subj' + str(subj) + 'bold.nii.gz')

    flavor = 'dico_bold7Tp1_to_subjbold7Tp1'
    group_brain_mask = 'brainmask_group_template.nii.gz'
    mask_fname = os.path.join(datapath, 'sub{0:03d}'.format(subj), 'templates', 'bold7Tp1', 'brain_mask.nii.gz')
    for run_id in dhandle.get_task_bold_run_ids(task)[subj]:
        run_ds = dhandle.get_bold_run_dataset(subj,task,run_id,chunks=run_id-1,mask=mask_fname,flavor=flavor)
        filename = 'brain_subj_{}_run_{}.nii.gz'.format(subj, run_id)
        tmp_path = scratch_path + filename
        save(unmask(run_ds.samples.astype('float32'), mask_fname), tmp_path)
        warp = fsl.ApplyWarp()
        warp.inputs.in_file = tmp_path
        warp.inputs.out_file = scratch_path+'group_'+filename
        warp.inputs.ref_file = '/data/forrest_gump/phase1/templates/grpbold7Tp1/brain.nii.gz'
        warp.inputs.field_file = '/data/forrest_gump/phase1/sub{:03}/templates/bold7Tp1/in_grpbold7Tp1/subj2tmpl_warp.nii.gz'.format(subj)
        warp.inputs.interp = 'nn'
        warp.run()
        os.remove(tmp_path)
        run_ds = mvpa.fmri_dataset(scratch_path+'group_'+filename, mask=group_mask, chunks=run_id-1)
        mvpa.poly_detrend(run_ds, polyord=1)
        mvpa.zscore(run_ds)
        os.remove(scratch_path+'group_'+filename)
        yield run_ds.samples.astype('float32')

def cut_out_overlap(run_a, run_b):
    '''cuts out the overlap between run_a and run_b'''
    return np.vstack([run_a[:-4], run_b[4:]])

def process_subj(subj, scratch_path='/data/mboos/tmp/',
                 save_path='/data/mboos/encoding/fmri/',
                 datapath='/data/forrest_gump/phase1', **kwargs):
    '''this function preprocesses subj run-wise and then saves it in voxel-splits'''
    task = 1
    model = 1

    # preprocess participant and concatenate runs
    fmri_data = reduce(cut_out_overlap, preprocess_and_tmp_save_fmri(datapath, task, subj, model, scratch_path, **kwargs))

    # and we're going to remove the last fmri slice
    # since it does not correspond to a movie part anymore
    fmri_data = fmri_data[:-1]

    # and also the first six seconds since not enough stimulus was presented to predict yet
    fmri_data = fmri_data[3:]

    joblib.dump(fmri_data, save_path+'fmri_subj_{}.pkl'.format(subj))
