import numpy as np
import pandas as pd
from numpy.linalg import svd

from ..array_ops import concat_apply
from ..utils import _parse_outstruct_args



def rereference(arr, data=None, field='resp', method='avg', return_reference=False):
    """
    Rereference responses based on the specification of a connection array defining which
    electrodes should be used to define the "reference" for each electrode.
    
    Parameters
    ----------
    arr : np.ndarray
        Square matrix defining connections between electrodes and their groupings. Arr should have
        dtype float, but can be entirely 1s and 0s or can encode weights with intermediate values.
        This can be created by one of the helper functions, like ``make_contact_rereference_arr``.
    data : naplib.Data object, optional
        Data object containing data to be normalized in one of the field. If not given, then the
        the data to be normalized must be passed directly as a list of trial arrays
        to the ``field`` argument instead of a string. 
    field : string | list of np.ndarrays or a multidimensional np.ndarray, default='resp'
        Field to normalize. If a string, it must specify one of the fields of the Data
        provided in the first argument. If a multidimensional array, first dimension
        indicates the trial/instances which will be concatenated over to compute
        normalization statistics. If a list, each array must be a multidimensional array
        of shape (time_i, channels)
    method : string, default='avg'
        Method for computing the reference over a group of electrodes. Options are 'avg' (average),
        'med' (median), or 'pca' (first principle component). Note, PCA method will whiten responses
        first.
    return_reference : bool, default=False
        If True, also return the reference computed for each electrode, which will be a list of
        numpy arrays, just like the rereferenced_data. So rereferenced_data[i]+reference[i] will
        reproduce field[i].
    
    Returns
    -------
    rereferenced_data : list of np.ndarrays
        Re-referenced data.
    reference : list of np.ndarrays
        Reference for each electrode. Only returned if ``return_reference=True``

    See Also
    --------
    make_contact_rereference_arr
    """
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f'arr must be a square matrix, but got arr of shape {arr.shape}')

    data_ = _parse_outstruct_args(data, field)

    if arr.shape[0] != data_[0].shape[1]:
        raise ValueError(f'arr must have shape channels * channels, but arr has shape {arr.shape} for response of shape {data_[0].shape}')

    def _rereference(data_arr, method='avg', return_ref=False):
        """Helper function to perform rereferencing on single array"""
        if data_arr.ndim < 2:
            return data_arr

        cached_ref = None
        if method == 'pca':
            data_to_use = (data_arr - data_arr.mean(1, keepdims=True)) / data_arr.std(1, keepdims=True)
        else:
            data_to_use = data_arr

        re_ref_data = np.zeros(data_arr.shape, dtype=data_arr.dtype)
        for channel in range(arr.shape[0]):
            ref_channels = arr[channel] # now a 1D array of shape (channels,)
            is_cached = cached_ref is not None and np.allclose(ref_channels, arr[channel-1])

            if method == 'avg':
                if not is_cached:
                    weighted_data = data_arr[:,ref_channels!=0]
                    cached_ref = np.nanmean(weighted_data, axis=1)

                ref = cached_ref

            elif method == 'pca':
                if not is_cached:
                    weighted_data = data_arr[:,ref_channels!=0]
                    weighted_data = (weighted_data - weighted_data.mean(1, keepdims=True)) / weighted_data.std(1, keepdims=True)
                    u, _, _ = svd(weighted_data.T @ weighted_data)
                    cached_ref = u[:,0] * (weighted_data @ u[:,0][:,np.newaxis])

                ref_channels[channel] = 1
                nonzero_channel_indices = np.argwhere(ref_channels != 0).squeeze()
                this_ref_which_index = list(nonzero_channel_indices).index(channel)
                ref = cached_ref[:,this_ref_which_index]

            elif method == 'med':
                if not is_cached:
                    weighted_data = data_arr[:,ref_channels!=0]
                    cached_ref = np.nanmedian(weighted_data, axis=1)

                ref = cached_ref

            else:
                raise ValueError(f'Invalid rereference method. Got "{method}"')
            
            if return_ref:
                re_ref_data[:,channel] = ref
            else:
                re_ref_data[:,channel] = data_to_use[:,channel] - ref

        return re_ref_data
        
    data_rereferenced = concat_apply(data_, _rereference, function_kwargs=dict(method=method))

    if return_reference:
        reference_subtracted = concat_apply(data_, _rereference, function_kwargs=dict(method=method, return_ref=True))
        return data_rereferenced, reference_subtracted
    return data_rereferenced


def make_contact_rereference_arr(channelnames, extent=None, grid_sizes={}):
    """
    Create grid which defines re-referencing scheme based on electrodes being on the same contact as
    each other.
    
    Parameters
    ----------
    channelnames : list or array-like
        Channelname of each electrode. They must follow the following scheme: 1) All channelnames must be
        be alphanumeric, with any numbers only being on the right. 2) The numeric portion specifies a
        different electrode number, while the character portion in the left of the channelname specifies the
        contact name. E.g. ['RT1','RT2','RT3','Ls1','Ls2'] indicates two contacts, the first with 3 electrodes
        and the second with 2 electrodes.
    extent : int, optional, default=None
        If provided, then only contacts from the same group which are within ``extent`` electrodes away
        from each other (inclusive) are still grouped together. For example, if ``extent=1``, only the
        nearest electrode on either side of a given electrode on the same contact is still grouped with it.
        This ``extent=1`` produces the traditional local average reference scheme.
    grid_sizes : dict, optional, default={}
        If provided, contains {'contact_name': (nrow, ncol)} values for any known ECoG grid sizes.
        E.g. {'GridA': (8, 16)} indicates that electrodes on contact 'GridA' are arranged in an 8 x 16 grid, 
        which is needed to determine adjacent electrodes for local average referencing with ``extent >= 1``.
    
    Returns
    -------
    arr : np.ndarray
        Square matrix of rereference connections.

    See Also
    --------
    rereference
    """
    def _find_adjacent_numbers(a, b, number, extent):
        '''
        Used to determine electrodes for local averaging ECoG grid"
        '''
        # Validate if the number is within the valid range
        if number < 1 or number > a * b:
            raise ValueError("The number is outside the range of the grid.")
    
        # Calculate the row and column of the given number
        row = (number - 1) // b
        col = (number - 1) % b
    
        # Find all adjacent numbers within the extent
        adjacent_numbers = []
        for dr in range(-extent, extent + 1):  # Rows within the extent
            for dc in range(-extent, extent + 1):  # Columns within the extent
                if dr == 0 and dc == 0:
                    continue  # Skip the number itself
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < a and 0 <= new_col < b:
                    adjacent_num = new_row * b + new_col + 1
                    adjacent_numbers.append(adjacent_num)
    
        return np.array(adjacent_numbers, dtype=int)
    connections = np.zeros((len(channelnames),) * 2, dtype=float)
    channelnames = np.array(channelnames)
    contact_arrays = np.array([x.rstrip('0123456789') for x in channelnames])
    contacts, ch_per_contact = np.unique([x.rstrip('0123456789') for x in channelnames], return_counts=True)
    if extent is None:
        # Common average referencing per electrode array (ECoG grid or sEEG shank)
        # CAR will end up subtracting parts of channel ch from itself
        for contact, num_ch in zip(contacts, ch_per_contact):
            for ch in range(1,num_ch+1):
                curr = np.where(channelnames==f'{contact}{ch}')[0]
                inds = np.where(contact_arrays==contact)[0]
                connections[curr,inds] = 1
    elif extent < 1:
        raise ValueError(f'Invalid extent. Must be no less than 1 but got extent={extent}')
    else:
        # Local average referencing within each electrode array
        # LAR will NOT subtract parts of channel ch from itself
        for contact, num_ch in zip(contacts, ch_per_contact):
            for ch in range(1,num_ch+1):
                # Local referencing for ECoG grids
                if 'grid' in contact.lower():
                    side = np.sqrt(num_ch)
                    half_side = np.sqrt(num_ch/2)
                    # Check grid_sizes dict
                    if contact in grid_sizes:
                        nrows, ncols = grid_sizes[contact]
                    # Assume a square
                    elif np.isclose(side, int(side)):
                        nrows, ncols = side, side
                    # Assume a 1 x 2 rectangle
                    elif np.isclose(half_side, int(half_side)):
                        nrows, ncols = half_side, half_side*2
                    else:
                        raise Exception(f'Cannot determine {contact} layout. Please include layout in `grid_sizes`')
                    adjacent = _find_adjacent_numbers(nrows, ncols, ch, extent)
                    curr = np.where(channelnames==f'{contact}{ch}')[0]
                    inds = []
                    for adj in adjacent:
                        inds.append(np.where(channelnames==f'{contact}{adj}')[0])
                    
                # Local referencing for sEEG shanks and strips
                else:
                    curr = np.where(channelnames==f'{contact}{ch}')[0]
                    inds = []
                    for cc in range(ch-extent, ch+extent+1):
                        if cc != ch:
                            inds.append(np.where(channelnames==f'{contact}{cc}')[0])

                inds = np.concatenate(inds)
                connections[curr,inds] = 1
                
    return connections
