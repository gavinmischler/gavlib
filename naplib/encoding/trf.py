import copy
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from mne.decoding import TimeDelayingRidge, ReceptiveField
from mne.decoding.time_delaying_ridge import _compute_corrs, _fit_corrs, _compute_reg_neighbors
from mne.decoding.receptive_field import _reshape_for_est, _delays_to_slice, _times_to_delays, _delay_time_series, _corr_score, _r2_score

from ..out_struct import OutStruct
from ..utils import _parse_outstruct_args

class TRF(BaseEstimator):
    '''
    Allows the fitting of temporal receptive field (TRF) models
    to one or more targets at a time using cross-validation. These can be
    encoding models (stimulus-to-brain) or decoding (brain-to-stimulus) models.
    
    Internally, fits several mne.decoding.ReceptiveField models, one for each
    target variable.
    
    Parameters
    ----------
    tmin : float
        The starting lag (inclusive), in seconds (or samples if ``sfreq`` == 1).
    tmax : float
        The ending lag (noninclusive), in seconds (or samples if ``sfreq`` == 1).
        Must be > tmin.
    sfreq : float
        The sampling frequency used to convert times into samples.
    reg_type : str, default='ridge'
        Regularization type. Can be "ridge" (default) or "laplacian".
    alpha : float | list or array-like of floats, default=np.logspace(2, 9, 8).
        Regularization strength. If a list or array-like of values, then the best
        one will be fit with cross-validation. If a list is given, alpha is
        optimized for each target variable individually.
    xval_test_portion : float, default=0.25
        If multiple alpha values are given, cross-validataion is performed to
        choose the best, using this portion as withheld test data on each
        cross-validation loop.
    fit_intercept : bool | None, default=False
        If True, the sample mean is removed before fitting.
    scoring : str, default='corrcoef'
        Defines how predictions will be scored. Currently must be one of
        'r2' (coefficient of determination) or 'corrcoef' (the correlation
        coefficient).
    n_jobs : int | str
        Number of jobs to run in parallel. Can be 'cuda' if CuPy
        is installed properly and ``estimator is None``.
    random_state : int, default=1
        Random state for generation of random gaussian noise
        if ``noise_between_trials`` is True in self.fit().

    Notes
    -----
    For a causal system, the encoding model would have significant
    non-zero values only at positive lags. In other words, lags point
    backwards in time relative to the input, so positive lags correspond
    to previous time samples, while negative lags correspond to future
    time samples. In most cases, an encoding model should use tmin=0
    and tmax>0, while a decoding model should use tmin<0 and tmax=0.
    
    '''
    def __init__(self,
                 tmin,
                 tmax,
                 sfreq,
                 reg_type='ridge',
                 alpha=None,
                 xval_test_portion=0.25,
                 fit_intercept=False,
                 scoring='corrcoef',
                 n_jobs=1,
                 random_state=1):
        
        if tmin >= tmax:
            raise ValueError(f'tmin must be less than tmax, but got {tmin} and {tmax}')
        
        if alpha is None:
            alpha = [round(x, 2) for x in np.logspace(2, 9, 8)]
        
        self.sfreq = float(sfreq)
        self.tmin = tmin
        self.tmax = tmax
        self.reg_type = reg_type
        self.alpha = np.array([alpha]) if isinstance(alpha, float) or isinstance(alpha, int) else alpha
        self.xval_test_portion = xval_test_portion
        self.fit_intercept = fit_intercept
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        if len(self.alpha) > 1 and self.xval_test_portion > 0.5:
            raise ValueError(f'xval_test_portion must be no more than 0.5 if multiple alphas were given,'+
                            f' so that cross validation can occur correctly, but got {xval_test_portion}')
        
    @property
    def _smin(self):
        return int(round(self.tmin * self.sfreq))

    @property
    def _smax(self):
        return int(round(self.tmax * self.sfreq)) + 1
        
    def _delay_and_reshape(self, X, y=None):
        """Delay and reshape the variables.
        X and y should be arrays.
        """
        if not isinstance(self.estimator_, TimeDelayingRidge):
            # X is now shape (n_times, n_epochs, n_feats, n_delays)
            X = _delay_time_series(X, self.tmin, self.tmax, self.sfreq,
                                   fill_mean=self.fit_intercept)
            X = _reshape_for_est(X)
            # Concat times + epochs
            if y is not None:
                y = y.reshape(-1, y.shape[-1], order='F')
        return X, y
    
    def fit(self, outstruct=None, X='aud', y='resp'):
        '''
        Fit a multi-output model to the data in X and y, which contain multiple trials.
        
        Parameters
        ----------
        outstruct : naplib.OutStruct object, optional
            OutStruct containing data to be normalized in one of the field.
            If not given, must give the X and y data directly as the ``X``
            and ``y`` arguments. 
        X : str | list of np.ndarrays or a multidimensional np.ndarray
            Data to be used as predictor in the regression. Once arranged,
            should be of shape (time, num_features).
            If a string, it must specify one of the fields of the outstruct
            provided in the first argument. If a multidimensional array, first dimension
            indicates the trial/instances which will be concatenated over to compute
            normalization statistics.
        y : str | list of np.ndarrays or a multidimensional np.ndarray
            Data to be used as target(s) in the regression. Once arranged,
            should be of shape (time, num_targets[, num_features_y]).
            If a string, it must specify one of the fields of the outstruct
            provided in the first argument. If a multidimensional array, first dimension
            indicates the trial/instances which will be concatenated over to compute
            normalization statistics.
        
                
        Returns
        -------
        self : returns an instance of self
        
        '''
        
        X_, y_ = _parse_outstruct_args(outstruct, copy.deepcopy(X), copy.deepcopy(y))
        
        X_ = np.concatenate(X_, axis=0)

        if y_[0].ndim == 1:
            y_ = [yy.reshape(-1,1) for yy in y_]
            
        # smooth beginning and ending of each target
        hamming = np.hamming(2*int((self.tmax-self.tmin) * self.sfreq))
        hamming_start = hamming[:len(hamming)//2].reshape(-1,1)
        hamming_end = hamming[-len(hamming)//2:].reshape(-1,1)
        if y_[0].ndim == 3:
            hamming_start = np.concatenate([hamming_start.reshape(-1,1,1) for _ in range(y_[0].shape[-1])], axis=-1)
            hamming_end = np.concatenate([hamming_end.reshape(-1,1,1) for _ in range(y_[0].shape[-1])], axis=-1)
        for i in range(len(y_)):
            y_[i][:len(hamming_start)] = np.multiply(y_[i][:len(hamming_start)], hamming_start)
            y_[i][-len(hamming_end):] = np.multiply(y_[i][-len(hamming_end):], hamming_end)

            
        y_ = np.concatenate(y_, axis=0)
        
        
        self.ndim_y_ = y_.ndim # either 2 or 3, depending on whether each samples of y has multiple output features
        if self.ndim_y_ == 3:
            X_ = X_[:, np.newaxis, :]
        # X and y should now both be arrays of shape (time, num_features) and (time, num_targets), respectively
        
        self.models_ = []
        
        # Initialize delays
        self.delays_ = _times_to_delays(self.tmin, self.tmax, self.sfreq)
        
        # Define the slice that we should use in the middle
        self.valid_samples_ = _delays_to_slice(self.delays_)
        
        # for each target variable, fit a TRF model with cross-validation
        for target_idx in range(y_.shape[1]):
            
            print(f'Fitting model for output variable {target_idx}...')
            
            y_thistarget = y_[:,target_idx]
            if self.ndim_y_ == 2:
                y_thistarget = y_thistarget[:,np.newaxis]
            else:
                y_thistarget = y_thistarget[:,np.newaxis,:]
            
            if len(self.alpha) > 1:
                print(f'  Running cross-validation on alphas...')
                
                scores = []
                
                # for each alpha, do cross-validation to get score
                
                X_generator = _xval_time_splits(X_, self.xval_test_portion)
                y_generator = _xval_time_splits(y_thistarget, self.xval_test_portion)
                for (X_train, X_test), (y_train, y_test) in zip(X_generator, y_generator):
                    scores_thissplit = []
                                        
                    rf = self._get_single_RF_model(verbose=False)
                    rf._y_dim = self.ndim_y_
                    self.estimator_ = rf.estimator
                    
                    # take the expensive computation of covariance outside the alpha loop,
                    # and only do it once for each train-test split
                    
                    rf.delays_ = self.delays_
                    rf.valid_samples_ = self.valid_samples_
                    
                    # Create input features
                    X_train_delayed, y_train_delayed = self._delay_and_reshape(X_train, y_train)

                    cov_, x_y_, n_ch_x, X_offset, y_offset = _compute_corrs(
                        X_train_delayed, y_train_delayed, self._smin, self._smax, self.n_jobs,
                        self.fit_intercept, edge_correction=True)
                    rf.cov_ = cov_
                    rf.estimator.cov_ = cov_
                    rf.estimator_ = rf.estimator
                    
                    for ii, alpha_ in enumerate(self.alpha):
                        
                        coef_ = _fit_corrs(cov_, x_y_, n_ch_x,
                                self.reg_type, alpha_, n_ch_x)
                        coef_ = coef_[:,:-1]
                        
                        rf.coef_ = coef_
                        rf.estimator_.coef_ = coef_

                        if self.fit_intercept:
                            rf.estimator_.intercept_ = y_offset - np.dot(X_offset, self.coef_.sum(-1).T)
                        else:
                            rf.estimator_.intercept_ = 0.

                        # Now make predictions about the model output, given input stimuli.
                        y_pred_test = rf.predict(X_test)
                        scores_thissplit.append(self._score_predictions([y_pred_test], [y_test], [1.]))

                    scores.append(scores_thissplit)
                scores = np.array(scores).squeeze()
                mean_scores = scores.mean(0) # average score for each alpha over all xval folds
                
                # choose best alpha
                ix_best_alpha_lap = np.argmax(mean_scores)
                best_alpha_thistarget = self.alpha[ix_best_alpha_lap]
                _print_alpha_xvals(self.alpha, mean_scores)
                print('chose alpha={:.2e}'.format(best_alpha_thistarget))
            else:
                best_alpha_thistarget = self.alpha[0]
                
            rf = self._get_single_RF_model(alpha_val=best_alpha_thistarget)
            rf.fit(X_, y_thistarget)
            # remove last lag of weights because it has significant edge effect
            rf.estimator_.coef_ = rf.coef_[:,:,:-1]
            rf.coef_ = rf.coef_[:,:,:-1] 
            self.models_.append(rf)
            
        return self
    
    @property
    def coef_(self):
        if not hasattr(self, 'ndim_y_'):
            raise ValueError(f'Must call fit() first before accessing coef_ attribute.')
        coefs_ = []
        for mdl in self.models_:
            if self.ndim_y_ == 2:
                coefs_.append(mdl.coef_)
            else:
                coefs_.append(mdl.coef_[np.newaxis,:,:,:])
                
        return np.concatenate(coefs_, axis=0) # shape (num_outputs, freqs, lags)
            
    def _get_single_RF_model(self, alpha_val=0, verbose=0):
        estimator = TimeDelayingRidge(tmin=self.tmin, tmax=self.tmax, sfreq=self.sfreq, reg_type=self.reg_type,
                                      fit_intercept=self.fit_intercept, alpha=alpha_val)
        rf = ReceptiveField(tmin=self.tmin, tmax=self.tmax, sfreq=self.sfreq, estimator=estimator,
                            fit_intercept=self.fit_intercept, scoring=self.scoring, n_jobs=self.n_jobs, verbose=verbose)
        return rf
    
    def predict(self, outstruct=None, X='aud'):
        '''
        Parameters
        ----------
        outstruct : naplib.OutStruct object, optional
            OutStruct containing data to be normalized in one of the field.
            If not given, must give the X and y data directly as the ``X``
            and ``y`` arguments. 
        X : str | list of np.ndarrays or a multidimensional np.ndarray
            Data to be used as predictor in the regression. Once arranged,
            should be of shape (time, num_features).
            If a string, it must specify one of the fields of the outstruct
            provided in the first argument. If a multidimensional array, first dimension
            indicates the trial/instances which will be concatenated over to compute
            normalization statistics.
        
        Returns
        -------
        y_pred : np.ndarray or list of np.ndarrays, each with shape (time, num_targets[, num_features_y])
            Predicted target for each trial in X.
        '''
        
        if not hasattr(self, 'models_'):
            raise ValueError(f'Must call .fit() before can call .predict()')
        
        X = _parse_outstruct_args(outstruct, X)
        
        y_pred = [[] for _ in X]
        for ii, x_trial in enumerate(X):
            for mdl in self.models_:
                tmp_pred = mdl.predict(x_trial)
                
                if self.ndim_y_ == 2:
                    tmp_pred = tmp_pred.reshape(-1,1)
                else:
                    tmp_pred = tmp_pred.reshape(tmp_pred.shape[0],1,-1)
                y_pred[ii].append(tmp_pred)
            y_pred[ii] = np.concatenate(y_pred[ii], axis=1)
        
        return y_pred
    
    def score(self, outstruct=None, X='aud', y='resp', weight_trial_duration=True):
        '''
        Get scores from predictions of the model.
        
        Parameters
        ----------
        outstruct : naplib.OutStruct object, optional
            OutStruct containing data to be normalized in one of the field.
            If not given, must give the X and y data directly as the ``X``
            and ``y`` arguments. 
        X : str | list of np.ndarrays or a multidimensional np.ndarray
            Data to be used as predictor in the regression. Once arranged,
            should be of shape (time, num_features).
            If a string, it must specify one of the fields of the outstruct
            provided in the first argument. If a multidimensional array, first dimension
            indicates the trial/instances which will be concatenated over to compute
            normalization statistics.
        y : str | list of np.ndarrays or a multidimensional np.ndarray
            Data to be used as target(s) in the regression. Once arranged,
            should be of shape (time, num_targets[, num_features_y]).
            If a string, it must specify one of the fields of the outstruct
            provided in the first argument. If a multidimensional array, first dimension
            indicates the trial/instances which will be concatenated over to compute
            normalization statistics.
        weight_trial_duration : bool, default=True
            Scores are computed for each trial and then aggregated by
            averaging across trials in X. If set to True (default), this
            averaging is weighted by the duration of each trial.

        Returns
        -------
        scores : np.array of float, shape (n_outputs,)
            The scores estimated by the model for each output (e.g. mean
            R2 of ``predict(X)``).
        '''
        
        if not hasattr(self, 'models_'):
            raise ValueError(f'Must call .fit() before can call .score()')
        
        X_, y_ = _parse_outstruct_args(outstruct, copy.deepcopy(X), copy.deepcopy(y))        

        y_preds = []
        for x, yy in zip(X_, y_):
            if yy.ndim != self.ndim_y_:
                raise ValueError(f'True y values are incorrect shape. Should be {self.ndim_y_} dimensions, but found a trial with {yy.ndim}')
            y_preds.append(self.predict(X=[x])[0])
            
            
        if weight_trial_duration:
            weights = [x.shape[0] for x in X_]
            weights = [w / float(sum(weights)) for w in weights]
        else:
            weights = [1. for _ in X_]
            
        return self._score_predictions(y_preds, y_, weights)
    

    def _score_predictions(self, y_preds_list, y_true_list, weights):
        
        scorer = _SCORERS[self.scoring]
        
        # for each trial, compute scores
        y_scores = []
        for y_pred, y_true in zip(y_preds_list, y_true_list):
            assert y_true.ndim == self.ndim_y_
            if self.ndim_y_ == 3:
                y_pred = y_pred.transpose(0,2,1).reshape(-1, y_pred.shape[1]) # move output channel dimension to end, then flatten all other features over time
                y_true_reshaped = y_true.transpose(0,2,1).reshape(-1, y_true.shape[1])
            else:
                y_true_reshaped = y_true
            assert y_pred.shape[1] == y_true.shape[1]
            y_scores.append(scorer(y_pred, y_true_reshaped, multioutput='raw_values')[:,np.newaxis])
        y_scores = np.concatenate(y_scores, axis=1)
        
        return np.average(y_scores, axis=1, weights=weights)
            
def _print_alpha_xvals(alphas, alpha_scores):
    
    print('Cross-validation scores:')
    for alpha_i, score_i in zip(alphas, alpha_scores):
        print('alpha={:.2e} : {:.4f}'.format(alpha_i, score_i))
                
def _xval_time_splits(data, test_portion):
    '''
    data : np.array of shape (time, num_features)
    test_portion : float between 0 and 1
    '''
    win_size = np.floor(data.shape[0] * test_portion).astype('int')
    
    num_chunks = int(((data.shape[0]-win_size)/win_size)+1)
    
    idx = np.arange(num_chunks*win_size)
    
    for i in range(0,num_chunks*win_size,win_size):
        start_idx = int(idx[i])
        stop_idx = int(idx[i] + win_size)
        train_data = np.delete(data, slice(start_idx, stop_idx), axis=0)
        test_data = data[start_idx:stop_idx]
        yield train_data, test_data

_SCORERS = {'r2': _r2_score, 'corrcoef': _corr_score}

