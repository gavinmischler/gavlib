import numpy as np
from scipy.ndimage import gaussian_filter1d
try:
    import torch
except:
    pass # soft requirement, since may not be needed 

from ..stats import fratio
from ..utils import _parse_outstruct_args

def get_label_change_points(x):
    '''
    Find the indices where x changes from one categorical (integer) value to another.
    For example, if x is [0, 0, 0, 1, 1, 3, 3], locations returned will be [3, 5]
    
    Parameters
    ----------
    x : np.ndarray of shape (time,)
        Label vector over time.
    
    Returns
    -------
    locs : array of shape (n_changes, )
        Indices where labels change.
    labels : array of shape (n_changes, )
        New label (from input `x`) after each transition.
    prior_labels : array of shape (n_changes, )
        Old label (from input `x`) before each transition.
    '''
    
    diff_x = np.concatenate([np.array([0]), np.diff(x)])
    locs = np.argwhere(diff_x!=0).squeeze()
    labels = x[locs]
    labels_prior = x[locs-1]
    return locs, labels, labels_prior

def segment_around_labeltransitions(outstruct=None, data=None, labels=None, prechange_samples=50, postchange_samples=300, elec_lag=None):
    '''
    Cut x around the transition points given by the changes in the labels.
    
    Parameters
    ----------
    outstruct : naplib.OutStruct object, optional
        OutStruct containing data to be normalized in one of the field. If not given, must give
        the data to be normalized directly as the ``data`` argument. 
    data : list or array-like, or a string specifying a field of the outstruct
        Data to be segmented based on the data[i] is shape (time, electrode, *, ...)
    labels : string, or list of np.ndarrays or array-like with the same length as data, or a tuple of lists or array-likes each the same length as data
        If a string, specifies a field of the oustruct containing labels to use for
        segmenting. If a single list or np.ndarray (not a tuple), then labels[i]
        is shape (time, ) giving integer label of each trial.
        If a tuple, then labels[0][i] is shape (time, ) giving integer label of each sample, and the
        transition points of these labels are used to segment. The other lists of labels (like labels[1])
        are not used for segmentation but their values surrounding the transition point are returned in a tuple of labels
    prechange_samples : int, default=50
        Number of samples to include in each segment that come before a label transition.
    postchange_samples : int, default=300
        Number of samples to include in each segment that come after a label transition.
    elec_lag : list or array-like, default=None
        Provides the lag (in samples) of each electrode, must be length=n_electrodes=data[i].shape[1].
        Only used if not None. This can be computed with many methods, for example, using
        ``nl.segmentation.electrode_lags_fratio``.
        
    Returns
    -------
    segments : array of shape (n_segments, time, electrode, *, ...)
        Segments cut from data surrounding label transition points.
    labels : array-like
        If input labels is just a list, then is of shape (n_segments,) providing new label after transition point
        If input labels is a tuple of lists, then this is a tuple of array-likes, where the first is the
        same as described above, and the others are arrays of shape (n_segments, time)
            
    prior_labels : array-like, shape (n_segments,)
        Gives label that came before the transition in the main labels array used for segmentation

    Examples
    --------
    >>> from naplib.segmentation import segment_around_labeltransitions
    >>> # In this example, we imagine we only have 1 trial of data, so all the lists of data and labels
    >>> # are length 1. In practice, they could be any length though.
    >>> arr = np.arange(20).reshape(10,2) # array of shape (time, features/electrodes)
    >>> arr
    array([[ 0,  1],
           [ 2,  3],
           [ 4,  5],
           [ 6,  7],
           [ 8,  9],
           [10, 11],
           [12, 13],
           [14, 15],
           [16, 17],
           [18, 19]])
    >>> # use a label of categorical values to segment the array based on
    >>> # transitions in the categorical label
    >>> label = np.array([0,0,1,1,1,0,0,3,3,3])
    >>> segments, labels, prior_labels = segment_around_labeltransitions(data=[arr], labels=[label],
    ...                                                                  prechange_samples=0,
    ...                                                                  postchange_samples=3)
    >>> segments
    array([[[ 4,  5],
            [ 6,  7],
            [ 8,  9]],
           [[10, 11],
            [12, 13],
            [14, 15]],
           [[14, 15],
            [16, 17],
            [18, 19]]])
    >>> labels, prior_labels
    (array([1, 0, 3]), array([0, 1, 0]))
    >>> # We can also get the full segment value of these labels or another set of
    >>> # labels around the transition, rather than just the single values of the
    >>> # labels at the transition point, simply by passing other label vectors
    >>> # that we want to be segmented as the 2nd through nth value in a tuple of labels
    >>> label2 = np.array([0,1,2,3,4,5,6,7,8,9]) # another set of labels of interest
    >>> label_tuple = ([label], [label], [label2])
    >>> segments, labels, prior_labels = segment_around_labeltransitions(data=[arr], labels=label_tuple, prechange_samples=0, postchange_samples=3)
    >>> labels # this time we get a tuple of 3 labels. The first is the same as before, and the others are fully segmented 
    (array([1, 0, 3]),
     array([[1, 1, 1],
            [0, 0, 3],
            [3, 3, 3]]),
     array([[2, 3, 4],
            [5, 6, 7],
            [7, 8, 9]]))


    '''
    
    x, labels = _parse_outstruct_args(outstruct, data, labels)
    
    if isinstance(labels, tuple):
        labels, other_labels = labels[0], labels[1:]
        return_multiple_labels = True
    else:
        other_labels = (labels,)
        return_multiple_labels = False
    
    segments = []
    new_labels = []
    prior_labels = []
    other_labels_to_return = [[] for _ in range(len(other_labels))]
    for i, tmp_unpack in enumerate(zip(x, labels, *other_labels)):
        x_i, labels_i, other_labels_i = tmp_unpack[0], tmp_unpack[1], tmp_unpack[2:]
        if not isinstance(x_i, np.ndarray) and isinstance(x_i, torch.Tensor): # only check if a torch Tensor if not np array
            x_i = x_i.numpy()
        label_changepoints, labels_at_changepoints, labels_before_changepoints = get_label_change_points(labels_i)
        
        label_changepoints = label_changepoints.astype('int')
        labels_before_changepoints = labels_before_changepoints.astype('int')
        
        for i_c, (change_point, new_lab, prior_lab) in enumerate(zip(label_changepoints, labels_at_changepoints, labels_before_changepoints)):
            if change_point > prechange_samples and change_point+postchange_samples <= x_i.shape[0]:
                if elec_lag is not None:
                    tmp_x_i_region = np.array([x_i[change_point-prechange_samples+elec_lag[elec_idx]:change_point+postchange_samples+elec_lag[elec_idx], elec_idx] for elec_idx in range(len(elec_lag))])
                    segments.append(tmp_x_i_region.transpose())
                else:
                    segments.append(x_i[change_point-prechange_samples:change_point+postchange_samples])
                new_labels.append(new_lab)
                prior_labels.append(prior_lab)

                for i, other_labs in enumerate(other_labels_i):
                    other_labels_to_return[i].append(other_labs[change_point-prechange_samples:change_point+postchange_samples])
                    
    if return_multiple_labels:
         return np.array(segments), (np.array(new_labels), *[np.array(tt) for tt in other_labels_to_return]), np.array(prior_labels)
    else:
        return np.array(segments), np.array(new_labels), np.array(prior_labels)
    
def electrode_lags_fratio(outstruct=None, data=None, labels=None, max_lag=20, return_fratios=False):
    '''
    Compute lags of each electrode based on peak of fratio to a given label, such as phoneme labels.
    
    Parameters
    ----------
    outstruct : naplib.OutStruct object, optional
        OutStruct containing data to be normalized in one of the field. If not given, must give
        the data to be normalized directly as the ``data`` argument. 
    data : list or array-like, length 18
        data[i] is shape (time, n_electrodes, *, ...)
    labels : list or array-like, same length as data
    return_fratios : bool
        whether or not to return smoothed_fratios along with lags
    
    Returns
    -------
    lags : np.ndarray, shape=(n_electrodes,)
        lag of each electrode, in samples
    smoothed_fratios : np.ndarray, shape=(n_electrodes, time)
        fratio of each electrode over time, after gaussian smoothing.
        Only returned if ``return_fratios``==True
    '''
        
    segments, labels, _ = segment_around_labeltransitions(outstruct=outstruct, data=data, labels=labels, prechange_samples=0, postchange_samples=max_lag)
    
    fratios_lags = fratio(segments.transpose((2,1,0)), labels, elec_mode='individual')
    
    fratios_lags_smooth = gaussian_filter1d(fratios_lags, 0.5, mode='constant')

    lags = fratios_lags_smooth.argmax(-1)
    
    if return_fratios:
        return lags, fratios_lags_smooth
    else:
        return lags
    
