from __future__ import division
import os
import mvpa2.suite as mvpa
import numpy as np
import joblib
from nilearn.masking import unmask
from nibabel import save

def preprocess_and_tmp_save_fmri(data_path, task, subj, model, tmp_path,
                                 group_mask=None):
    '''
    Generator for preprocessed fMRI runs from  one subject of Forrest Gump
    aligns to group template
    run-wise linear de-trending and z-scoring
    IN:
        data_path    -   string, path pointing to the Forrest Gump directory
        task        -   string, which part of the Forrest Gump dataset to load
        subj        -   int, subject to pre-process
        tmp_path    -   string, path to save the dataset temporarily to
    OUT:
        preprocessed fMRI samples per run'''
    from nipype.interfaces import fsl
    dhandle = mvpa.OpenFMRIDataset(data_path)

    flavor = 'dico_bold7Tp1_to_subjbold7Tp1'
    if group_mask is None:
        group_mask = os.path.join(data_path, 'sub{0:03d}'.format(subj), 'templates',
                              'bold7Tp1', 'in_grpbold7Tp1', 'brain_mask.nii.gz')
    mask_fname = os.path.join(data_path, 'sub{0:03d}'.format(subj), 'templates',
                              'bold7Tp1', 'brain_mask.nii.gz')
    for run_id in dhandle.get_task_bold_run_ids(task)[subj]:
        run_ds = dhandle.get_bold_run_dataset(subj, task, run_id,
                                              chunks=run_id-1, mask=mask_fname,
                                              flavor=flavor)
        filename = 'brain_subj_{}_run_{}.nii.gz'.format(subj, run_id)
        tmp_file = os.path.join(tmp_path, filename)
        save(unmask(run_ds.samples.astype('float32'), mask_fname), tmp_file)
        warp = fsl.ApplyWarp()
        warp.inputs.in_file = tmp_file
        warp.inputs.out_file = os.path.join(tmp_path, 'group_'+filename)
        warp.inputs.ref_file = os.path.join(data_path, 'templates',
                              'grpbold7Tp1', 'brain.nii.gz')
        warp.inputs.field_file = os.path.join(data_path, 'sub{0:03d}'.format(subj),
                                              'templates', 'bold7Tp1', 'in_grpbold7Tp1',
                                              'subj2tmpl_warp.nii.gz')
        warp.inputs.interp = 'nn'
        warp.run()
        os.remove(tmp_file)
        run_ds = mvpa.fmri_dataset(os.path.join(tmp_path, 'group_'+filename), mask=group_mask, chunks=run_id-1)
        mvpa.poly_detrend(run_ds, polyord=1)
        mvpa.zscore(run_ds)
        os.remove(os.path.join(tmp_path, 'group_'+filename))
        yield run_ds.samples.astype('float32')

def cut_out_overlap(run_a, run_b):
    '''cuts out the overlap between run_a and run_b'''
    return np.vstack([run_a[:-4], run_b[4:]])

def process_subj(subj, save_path, data_path, tmp_path=None, **kwargs):
    '''Preprocesses one subject run-wise and saves fMRI as a joblib pickle
    IN:
        subj        -   int, number of subject to pre-process
        save_path   -   string, path to the directory in which pre-processed data will be saved
        data_path   -   string, path to the directory of the Forrest Gump dataset
        tmp_path    -   string or None, optional, path to the directory in which temporary results will be saved'''
    # Forrest Gump, auditory version
    task = 1
    model = 1

    if tmp_path is None:
        tmp_path = save_path

    # preprocess participant and concatenate runs
    fmri_data = reduce(cut_out_overlap, preprocess_and_tmp_save_fmri(data_path, task, subj, model, tmp_path, **kwargs))

    # and we're going to remove the last fmri slice
    # since it does not correspond to a movie part anymore
    fmri_data = fmri_data[:-1]

    joblib.dump(fmri_data, os.path.join(save_path, 'fmri_subj_{}.pkl'.format(subj)))
