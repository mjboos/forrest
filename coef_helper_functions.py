import numpy as np
from copy import deepcopy

def remove_BF_from_coefs(estimator, remove_bf):
    '''Returns estimator with all coefficients set to zero that correspond to basis functions in remove bool'''
    remove_coef = np.tile(remove_bf, 60)
    estimator_new = deepcopy(estimator)
    estimator_new.transformedcomp[remove_coef,:] = 0
    return estimator_new

def get_cluster_coefs_from_estimator(transformedcomp, cluster_idx_bool):
    '''Expects transformedcomp to be of shape (12000, 5)'''
    coefs = np.reshape(transformedcomp.T, (5, 3, 20, 200))
    # because the 20 time bins are from oldest to newest
    # AND the 3 time bins are from newest to oldest we need reorder
    coefs = coefs[:,::-1]
    coefs = np.reshape(coefs, (5, 3*20, 200))
    # now the dimension with 60 bins is 1...60
    return coefs[:, : ,cluster_idx_bool]

def reshape_bsc(bsc):
    '''Reshapes the BSC BF activations from (samples, 12000) to (samples, 60, 200)
    with bins increasing from oldest to newest'''
    reshaped_bsc = np.reshape(bsc, (bsc.shape[0], 3, 20, 200))[:, ::-1]
    return np.reshape(reshaped_bsc, (bsc.shape[0], 3*20, 200))

def make_df_for_lineplot(cluster_array):
    import pandas as pd
    dfs = [pd.DataFrame(cluster_array[pc]) for pc in range(3)]
    for df in dfs:
        df.index = np.arange(-10, -4, step=0.1)
    df = pd.concat([df.T.melt() for df in dfs], ignore_index=True)
    df_pc_idx = pd.Series(np.repeat(['PC {}'.format(i+1) for i in range(3)], cluster_array[0].size), name='PC')
    df = pd.concat([df, df_pc_idx], axis=1)
    df = df.rename(columns={'variable' : 'Time (s)', 'value': 'Regression coefficient'})
    return df

def test_reshaping():
    # shape of TR samples
    # note: column ordering is now oldest --> newest in steps of 50
    original = np.arange(3542*200*20)
    patches = np.reshape(original, (-1, 200*20))

    strides = (patches.strides[0],) + patches.strides

    # rolling window of length 4 samples
    shape = (patches.shape[0] - 4 + 1, 4, patches.shape[1])

    patches = np.lib.stride_tricks.as_strided(patches[::-1,:].copy(),
                                              shape=shape,
                                              strides=strides)[::-1, :, :]

    patches = np.reshape(patches, (patches.shape[0], -1))

    # we kick out the most recent sample
    patches = patches[:, :-4000]
    cluster_idx = np.array([True]*200)
    testcase = np.tile(patches[0][None], (5,1)).T
    reordered_testcase = get_cluster_coefs_from_estimator(testcase, cluster_idx)
    assert np.allclose(reordered_testcase[0].flatten(),np.arange(4000, 16000))

if __name__=='__main__':
    pass

