import numpy as np
from numpy import newaxis, pi
from numpy.linalg import inv, norm
from numpy.random import rand
from scipy.special import logsumexp
import scipy.fftpack

#############################################
# subset of stolen functions
#############################################


def logpdf_gauss(x, mu, cov):
    assert(mu.ndim == 1 and len(mu) == len(cov) and (cov.ndim == 1 or cov.shape[0] == cov.shape[1]))
    x = np.atleast_2d(x) - mu
    if cov.ndim == 1:
        return -0.5*(len(mu)*np.log(2 * pi) + np.sum(np.log(cov)) + np.sum((x**2)/cov, axis=1))
    else:
        return -0.5*(len(mu)*np.log(2 * pi) + np.linalg.slogdet(cov)[1] + np.sum(x.dot(inv(cov)) * x, axis=1))

# GMM distributions related functions

def logpdf_gmm(x, ws, mus, covs):
    return logsumexp([np.log(w) + logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)], axis=0)


def train_gmm(x, ws, mus, covs):
    """
    TRAIN_GMM Single iteration of EM algorithm for training Gaussian Mixture Model
    [Ws_new,MUs_new, COVs_new, TLL]= TRAIN_GMM(X,Ws,NUs,COVs) performs single
    iteration of EM algorithm (Maximum Likelihood estimation of GMM parameters)
    using training data X and current model parameters Ws, MUs, COVs and returns
    updated model parameters Ws_new, MUs_new, COVs_new and total log likelihood
    TLL evaluated using the current (old) model parameters. The model
    parameters are mixture component mean vectors given by columns of M-by-D
    matrix MUs, covariance matrices given by M-by-D-by-D matrix COVs and vector
    of weights Ws.
    """   
    gamma = np.vstack([np.log(w) + logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)])
    logevidence = logsumexp(gamma, axis=0)
    gamma = np.exp(gamma - logevidence)
    tll = logevidence.sum()
    gammasum = gamma.sum(axis=1)
    ws = gammasum / len(x)
    mus = gamma.dot(x)/gammasum[:,np.newaxis]

    if covs[0].ndim == 1: # diagonal covariance matrices
      covs = gamma.dot(x**2)/gammasum[:,np.newaxis] - mus**2
    else:
      covs = np.array([(gamma[i]*x.T).dot(x)/gammasum[i] - mus[i][:, newaxis].dot(mus[[i]]) for i in range(len(ws))])        
    return ws, mus, covs, tll

# Speech feature extraction (spectrogram, mfcc, ...)

def mel_inv(x):
    return (np.exp(x/1127.)-1.)*700.

def mel(x):
    return 1127.*np.log(1. + x/700.)

def mel_filter_bank(nfft, nbands, fs, fstart=0, fend=None):
    """Returns mel filterbank as an array (nfft/2+1 x nbands)
    nfft   - number of samples for FFT computation
    nbands - number of filter bank bands
    fs     - sampling frequency (Hz)
    fstart - frequency (Hz) where the first filter strats
    fend   - frequency (Hz) where the last  filter ends (default fs/2)
    """
    if not fend:
      fend = 0.5 * fs

    cbin = np.round(mel_inv(np.linspace(mel(fstart), mel(fend), nbands + 2)) / fs * nfft).astype(int)
    mfb = np.zeros((nfft // 2 + 1, nbands))
    for ii in range(nbands):
        mfb[cbin[ii]:  cbin[ii+1]+1, ii] = np.linspace(0., 1., cbin[ii+1] - cbin[ii]   + 1)
        mfb[cbin[ii+1]:cbin[ii+2]+1, ii] = np.linspace(1., 0., cbin[ii+2] - cbin[ii+1] + 1)
    return mfb

def framing(a, window, shift=1):
    shape = ((a.shape[0] - window) // shift + 1, window) + a.shape[1:]
    strides = (a.strides[0]*shift,a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def spectrogram(x, window, noverlap=None, nfft=None):
    if np.isscalar(window): window = np.hamming(window)
    if noverlap is None:    noverlap = window.size // 2
    if nfft     is None:    nfft     = window.size
    x = framing(x, window.size, window.size-noverlap)
    x = scipy.fftpack.fft(x*window, nfft)
    return x[:,:x.shape[1]//2+1]

def mfcc(s, window, noverlap, nfft, fs, nbanks, nceps):
    #MFCC Mel Frequency Cepstral Coefficients
    #   CPS = MFCC(s, FFTL, Fs, WINDOW, NOVERLAP, NBANKS, NCEPS) returns 
    #   NCEPS-by-M matrix of MFCC coeficients extracted form signal s, where
    #   M is the number of extracted frames, which can be computed as
    #   floor((length(S)-NOVERLAP)/(WINDOW-NOVERLAP)). Remaining parameters
    #   have the following meaning:
    #
    #   NFFT          - number of frequency points used to calculate the discrete
    #                   Fourier transforms
    #   Fs            - sampling frequency [Hz]
    #   WINDOW        - window lentgth for frame (in samples)
    #   NOVERLAP      - overlapping between frames (in samples)
    #   NBANKS        - numer of mel filter bank bands
    #   NCEPS         - number of cepstral coefficients - the output dimensionality
    #
    #   See also SPECTROGRAM

    # Add low level noise (40dB SNR) to avoid log of zeros 
    snrdb = 40
    noise = rand(s.shape[0])
    s = s + noise.dot(norm(s, 2)) / norm(noise, 2) / (10 ** (snrdb / 20))

    mfb = mel_filter_bank(nfft, nbanks, fs, 32)
    dct_mx = scipy.fftpack.idct(np.eye(nceps, nbanks), norm='ortho') # the same DCT as in matlab

    S = spectrogram(s, window, noverlap, nfft)
    return dct_mx.dot(np.log(mfb.T.dot(np.abs(S.T)))).T
