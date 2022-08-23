import pytest
import numpy as np
from scipy.signal import convolve

from naplib.encoding import TRF

@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(1)
    x = rng.random(size=(10000,1))
    coef = np.array([[1.],[-0.5]])
    y1 = convolve(x, coef, mode='same')
    x_zeros = np.concatenate([x, np.zeros_like(x)], axis=1)
    return {'coef': coef,'X1': [x], 'X2': [np.concatenate((x,x), axis=1)], 'X3': [x_zeros], 'y1': [y1], 'y2': [np.concatenate((y1,y1), axis=1)]}

def test_1D_input_1D_output_trf(data):
    model = TRF(tmin=0, tmax=0.02, sfreq=100, alpha=[0])
    model.fit(X=data['X1'], y=data['y1'])
    assert np.allclose(data['coef'].reshape(1,1,2), model.coef_, rtol=1e-3)

def test_1D_input_2D_output_trf(data):
    model = TRF(tmin=0, tmax=0.02, sfreq=100, alpha=[0])
    model.fit(X=data['X1'], y=data['y2'])
    reshaped_coef = np.concatenate([data['coef'].reshape(1,1,2), data['coef'].reshape(1,1,2)], axis=1)
    assert np.allclose(reshaped_coef, model.coef_, rtol=1e-3)

def test_2D_input_1D_output_trf(data):
    model = TRF(tmin=0, tmax=0.02, sfreq=100, alpha=[0])
    model.fit(X=data['X2'], y=data['y1'])
    reshaped_coef = np.concatenate([data['coef'].reshape(1,1,2), data['coef'].reshape(1,1,2)], axis=1)/2
    assert np.allclose(reshaped_coef, model.coef_, rtol=1e-3)

def test_2D_input_1D_output_trf_withzeros(data):
    model = TRF(tmin=0, tmax=0.02, sfreq=100, alpha=[0])
    model.fit(X=data['X3'], y=data['y1'])
    truth_coef = np.zeros((1,2,2))
    truth_coef[0,0,:] = data['coef'].squeeze()
    assert np.allclose(truth_coef, model.coef_, rtol=1e-3)

def test_1D_input_1D_output_trf_with_xval(data):
    model = TRF(tmin=0, tmax=0.02, sfreq=100, alpha=[0, 0.0001, 0.0002])
    model.fit(X=data['X1'], y=data['y1'])
    assert np.allclose(data['coef'].reshape(1,1,2), model.coef_, rtol=1e-3)

def test_good_coef_shape_STRF():
    X = [np.random.rand(1000,20), np.random.rand(900,20)]
    y = [np.random.rand(1000,3), np.random.rand(900,3)] # 3 channels of output

    model = TRF(tmin=0, tmax=0.1, sfreq=100, alpha=[200])
    model.fit(X=X, y=y)
    assert model.coef_.shape == (3, 20, 10)

    model = TRF(tmin=0, tmax=0.24, sfreq=100, alpha=[100])
    model.fit(X=X, y=y)
    assert model.coef_.shape == (3, 20, 24)

def test_good_coef_shape_stim_recon():
    X = [np.random.rand(1000, 1, 20), np.random.rand(900, 1, 20)] # 1 channel with 20 features
    y = [np.random.rand(1000, 3), np.random.rand(900, 3)]

    model = TRF(tmin=-0.05, tmax=0, sfreq=100, alpha=[200])
    model.fit(X=y, y=X)
    assert model.coef_.shape == (1, 20, 3, 5)

    model = TRF(tmin=-0.25, tmax=0, sfreq=100, alpha=[100])
    model.fit(X=y, y=X)
    assert model.coef_.shape == (1, 20, 3, 25)

def test_good_pred_stim_recon():
    rng = np.random.default_rng(1)
    X = [rng.random(size=(1000, 1, 10)), rng.random(size=(900, 1, 10))] # 1 channel with 20 features
    y = [rng.random(size=(1000, 6)), rng.random(size=(900, 6))]

    model = TRF(tmin=-0.05, tmax=0, sfreq=100, alpha=[200, 300])
    model.fit(X=y, y=X)
    pred = model.predict(X=[rng.random(size=(100,6))])
    assert isinstance(pred, list)
    assert pred[0].shape == (100, 1, 10)

def test_good_pred_STRF():
    rng = np.random.default_rng(1)
    X = [rng.random(size=(1000, 10)), rng.random(size=(900, 10))]
    y = [rng.random(size=(1000, 6)), rng.random(size=(900, 6))]

    model = TRF(tmin=-0.05, tmax=0, sfreq=100, alpha=[200, 300])
    model.fit(X=X, y=y)
    pred = model.predict(X=[rng.random(size=(100,10))])
    assert isinstance(pred, list)
    assert len(pred) == 1
    assert pred[0].shape == (100, 6)

def test_scoring_STRF():
    rng = np.random.default_rng(1)
    X = [rng.random(size=(1000, 10)), rng.random(size=(900, 10))]
    y = [rng.random(size=(1000, 6)), rng.random(size=(900, 6))]

    model = TRF(tmin=-0.05, tmax=0, sfreq=100, alpha=[0])
    model.fit(X=X, y=y)
    score = model.score(X=X, y=y)
    assert score.shape == (6,)
