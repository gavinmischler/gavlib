from collections.abc import Iterable, Sequence
import numpy as np

class OutStruct(Iterable):
    '''
    Class for storing electrode response data along with
    task- and electrode-related variables. Under the hood, it consists
    of a list of dictionaries where each dictionary contains all the data
    for one trial.
    
    Parameters
    ----------
    data : dict or list of dictionaries
        The Nth ictionary defines the Nth trial data, typically for the Nth stimulus.
        Each dictionary must contain the same keys if passed in a list of multiple trials.
    strict : bool, default=True
        If True, requires strict adherance to the following standard
            - Each trial must contain at least the following fields:
              ['name','sound','soundf','resp','dataf']
            - Each trial must contain the exact same set of fields
        
    Methods
    -------
    set_field(field, fieldname)
    get_field(fieldname)
    append(trial_data)
    
    Attributes
    ----------
    fields : list of strings
        Names of all fields stored in this OutStruct
    data : list of dictionaries
        Data for each stimulus response and all associated variables
    
    '''
    def __init__(self, data, strict=True):
        
        if isinstance(data, dict):
            data = [data]
            self._data = data
        elif isinstance(data, list):
            self._data = data
        else:
            raise TypeError(f'Can only create OutStruct from a dict or a list '
                            f'of dicts, but found type {type(data)}')
        self._strict = strict
        self._validate_new_out_data(data, strict=strict)

                
    def set_field(self, fielddata, fieldname):
        '''
        Parameters
        ----------
        fielddata : list
            List containing data to add to each trial for this field. Must 
            be same length as this object
        fieldname : string
            Name of field to add. If this field already exists in the OutStruct
            then the current field will be overwritten.
        Returns
        -------
        '''
        if not isinstance(fielddata, list):
            raise TypeError(f'Input data must be a list, but found {type(fielddata)}')
        if len(fielddata) != len(self):
            raise Exception('Length of field is not equal to length of this OutStruct')
        for i in range(len(self.data)):
            self.data[i][fieldname] = fielddata[i]
            
    def get_field(self, fieldname):
        '''
        Return all trials for a single field.
        
        Parameters
        ----------
        fieldname : string
            Which field to get.
        Returns
        -------
        field : list
            List containing each trial's value for this field.
        '''
        try:
            return [tmp[fieldname] for tmp in self.data]
        except KeyError:
            raise KeyError(f'Invalid fieldname: {fieldname} not found in data.')
            
    def __getitem__(self, index):
        '''
        Parameters
        ----------
        index : int
            Which trial to get.
        Returns
        -------
        data : dict, list, or naplib.OutStruct
            If index is an integer, returns the corresponding trial as a dict. If index
            is a string, returns the corresponding field, and if it is a list of strings,
            returns those fields together in a new OutStruct object.
        '''
        if isinstance(index, slice):
            return OutStruct(self.data[index], strict=self._strict)
        if isinstance(index, str):
            return self.get_field(index)
        if isinstance(index, list) or isinstance(index, np.ndarray):
            if isinstance(index[0], str):
                return OutStruct([dict([(field, x[field]) for field in index]) for x in self], strict=False)
            else:
                return OutStruct([self.data[i] for i in index], strict=False)
#             else:
#                 raise IndexError(f'Cannot index from a list if it is not a list of '
#                                  f'strings or integers, found list of {type(index[0])}')
        try:
            # TODO: change this to return a type OutStruct if you do slicing - problem with trying to
            # print because it says KeyError for self.data[0] for key 0
            return self.data[index]
        except IndexError:
            raise IndexError(f'Index invalid for this data. Tried to index {index} but length is {len(self)}.')
        

            
    def __setitem__(self, index, data):
        '''
        Parameters
        ----------
        index : int or string
            Which trial to set, or which field to set.
        data : dict or list of data
            Either trial data to add or field data to add. If index is an
            integer, dictionary should contain all the same fields as
            current OutStruct object.
        Returns
        -------
        '''
        if isinstance(index, str):
            self.set_field(data, index)
        else:
            if index > len(self):
                raise IndexError((f'Index is too large. Current data is length {len(self)} '
                    'but tried to set index {index}. If you want to add to the end of the list '
                    'of trials, use the OutStruct.append() method.'))
            elif index == len(self):
                self.append(data)
            else:
                self.data[index] = data
     
    def append(self, trial_data, strict=None):
        '''
        Append trial data to end of OutStruct.
        
        Parameters
        ----------
        trial_data : dict
            Dictionary containing all the same fields as current OutStruct object.
        strict : bool, default=self._strict
            If true, enforces that new data contains the exact same set of fields as
            the current OutStruct. Default value is self._strict, which is set based
            on the input when creating a new OutStruct from scratch with __init__()

        Returns
        -------
        '''
        if strict is None:
            strict = self._strict
        self._validate_new_out_data([trial_data], strict=strict)
        self.data.append(trial_data)
        
    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return self.__str__() # until we can think of a better __repr__
    
    def __str__(self):
        to_return = f'OutStruct of {len(self)} trials containing {len(self.fields)} fields\n['
        
        to_print = 2 if len(self) > 3 else 3
        for trial_idx, trial in enumerate(self[:to_print]):
            fieldnames = list(trial.keys())
            to_return += '{'
            for f, fieldname in enumerate(fieldnames):
#                 to_return += f'"{fieldname}": {trial[fieldname].__str__()}'
                to_return += f'"{fieldname}": {type(trial[fieldname])}'
                if f < len(fieldnames)-1:
                    to_return += ', '
            if trial_idx < len(self)-1:
                to_return += '}\n'
            else:
                to_return += '}'
        if to_print == 3:
             to_return += ']\n'
        elif to_print == 2:
            to_return += '\n...\n{'
            fieldnames = list(self[-1].keys())
            for f, fieldname in enumerate(fieldnames):
#                 to_return += f'"{fieldname}": {self[-1][fieldname].__str__()}'
                to_return += f'"{fieldname}": {type(self[-1][fieldname])}'
                if f < len(fieldnames)-1:
                    to_return += ', '
            to_return += '}]'
        return to_return
    
    def _validate_new_out_data(self, input_data, strict=True):
        
        first_trial_fields = set(self.fields)
        for t, trial in enumerate(input_data):
            if not isinstance(trial, dict):
                raise TypeError(f'input data is not a list of dicts, found {type(trial)}')
            trial_fields = set(trial.keys())
            if strict and trial_fields != first_trial_fields:
                raise ValueError(f'New data does not contain the same fields as the first trial.')        
        
    @property
    def fields(self):
        '''Get names of all fields in this object'''
        return [k for k, _ in self.data[0].items()]
    
    @property
    def data(self):
        return self._data

    
def join_fields(outstructs, fieldname='resp', axis=-1, return_outstruct=False):
    '''
    Join trials from a field of multiple OutStruct objects by zipping them
    together and concatenating each trial together. The field must be of type
    np.ndarray and concatenation is done with np.concatenate().
    
    Parameters
    ----------
    outstructs : sequence of OutStructs
        Sequence containing the different outstructs to join
    fieldname : string, default='resp'
        Name of the field to concatenate from each OutStruct. For each trial in
        each outstruct, this field must be of type np.ndarray or something which
        can be input to np.concatenate().
    axis : int, default = -1
        Axis along which to concatenate each trial's data. The default corresponds
        to the channel dimension of the conventional 'resp' field of an OutStruct.
    return_outstruct : bool, default=False
        If True, returns data as an OutStruct with a single field named fieldname.

    Returns
    -------
    joined_data : list of np.ndarrays, or OutStruct
        Joined data of same length as each of the outstructs containing concatenated data
        for each trial.
    '''
    
    for out in outstructs:
        if not isinstance(out, OutStruct):
            raise TypeError(f'All inputs must be an OutStruct but found {type(out)}')
        field = out.get_field(fieldname)
        if not isinstance(field[0], np.ndarray):
            raise TypeError(f'Can only concatenate np.ndarrays, but found {type(field[0])} in this field')

    starting_fields = [out.get_field(fieldname) for out in outstructs] # each one should be a list of np.arrays
    
    to_return = []
    
    zipped_fields = list(zip(*starting_fields))
    for i, field_set in enumerate(zipped_fields):
        to_return.append(np.concatenate(field_set, axis=axis))
        
    if return_outstruct:
        return OutStruct([dict([(fieldname, x)]) for x in to_return], strict=False)
    return to_return
        
    
