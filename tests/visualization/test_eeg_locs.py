import numpy as np

from naplib.visualization import eeg_locs

def test_correct_locs_gtec62():
    pos = eeg_locs()

    expected = np.array([[-3.00089184e-02,  9.23579542e-02],
       [ 5.94632764e-18,  9.71109000e-02],
       [ 3.00089184e-02,  9.23579542e-02],
       [-5.97874400e-02,  7.65244335e-02],
       [-3.05203601e-02,  7.19014626e-02],
       [ 3.05203601e-02,  7.19014626e-02],
       [ 5.97874400e-02,  7.65244335e-02],
       [-7.85643684e-02,  5.70803549e-02],
       [-5.97483198e-02,  5.19384220e-02],
       [-3.98565595e-02,  4.92187520e-02],
       [-1.97710617e-02,  4.89350949e-02],
       [ 2.97322199e-18,  4.85564000e-02],
       [ 1.97710617e-02,  4.89350949e-02],
       [ 3.98565595e-02,  4.92187520e-02],
       [ 5.97483198e-02,  5.19384220e-02],
       [ 7.85643684e-02,  5.70803549e-02],
       [-9.23579542e-02,  3.00089184e-02],
       [-6.99658781e-02,  2.68573843e-02],
       [-4.66003846e-02,  2.47778640e-02],
       [-2.38847943e-02,  2.38847943e-02],
       [ 1.48661100e-18,  2.42782000e-02],
       [ 2.38847943e-02,  2.38847943e-02],
       [ 4.66003846e-02,  2.47778640e-02],
       [ 6.99658781e-02,  2.68573843e-02],
       [ 9.23579542e-02,  3.00089184e-02],
       [-9.71109000e-02,  1.18926553e-17],
       [-7.28327000e-02,  8.91943329e-18],
       [-4.85564000e-02,  5.94644398e-18],
       [-2.42782000e-02,  2.97322199e-18],
       [ 0.00000000e+00, -0.00000000e+00],
       [ 2.42782000e-02, -0.00000000e+00],
       [ 4.85564000e-02, -0.00000000e+00],
       [ 7.28327000e-02, -0.00000000e+00],
       [ 9.71109000e-02, -0.00000000e+00],
       [-9.23579542e-02, -3.00089184e-02],
       [-6.99658781e-02, -2.68573843e-02],
       [-4.66003846e-02, -2.47778640e-02],
       [-2.38847943e-02, -2.38847943e-02],
       [ 1.48661100e-18, -2.42782000e-02],
       [ 2.38847943e-02, -2.38847943e-02],
       [ 4.66003846e-02, -2.47778640e-02],
       [ 6.99658781e-02, -2.68573843e-02],
       [ 9.23579542e-02, -3.00089184e-02],
       [-7.85643684e-02, -5.70803549e-02],
       [-5.97483198e-02, -5.19384220e-02],
       [-3.98565595e-02, -4.92187520e-02],
       [-1.97710617e-02, -4.89350949e-02],
       [ 2.97322199e-18, -4.85564000e-02],
       [ 1.97710617e-02, -4.89350949e-02],
       [ 3.98565595e-02, -4.92187520e-02],
       [ 5.97483198e-02, -5.19384220e-02],
       [ 7.85643684e-02, -5.70803549e-02],
       [-5.70803549e-02, -7.85643684e-02],
       [-3.05203601e-02, -7.19014626e-02],
       [ 4.45971665e-18, -7.28327000e-02],
       [ 3.05203601e-02, -7.19014626e-02],
       [ 5.70803549e-02, -7.85643684e-02],
       [-3.00089184e-02, -9.23579542e-02],
       [ 5.94632764e-18, -9.71109000e-02],
       [ 3.00089184e-02, -9.23579542e-02],
       [-1.15447895e-01, -3.75112948e-02],
       [ 1.15447895e-01, -3.75112948e-02]])

    assert np.allclose(pos, expected)
