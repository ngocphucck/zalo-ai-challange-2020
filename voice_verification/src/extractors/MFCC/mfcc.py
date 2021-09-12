from __future__ import division
import numpy
from scipy.fftpack import dct
from voice_verification.src.extractors.MFCC import sigproc
from scipy.io import wavfile


def calculate_fft_len(sample_rate, win_len):
    window_length_samples = win_len * sample_rate
    fft_len = 1
    while fft_len < window_length_samples:
        fft_len *= 2

    return fft_len


def mfcc(signal, sample_rate=16000, win_len=0.025, win_step=0.01, num_cepstrum=13,
         n_filters=26, fft_len=None, low_freq=0, high_freq=None, pre_emphasize=0.97, cep_lifter=22, append_energy=True,
         win_func=lambda x: numpy.ones((x,))):
    fft_len = fft_len or calculate_fft_len(sample_rate, win_len)
    feat, energy = filter_bank(signal, sample_rate, win_len, win_step, n_filters, fft_len, low_freq, high_freq,
                               pre_emphasize, win_func)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :num_cepstrum]
    feat = lifter(feat, cep_lifter)
    if append_energy:
        feat[:, 0] = numpy.log(energy)  # replace first cepstral coefficient with log of frame energy

    return feat


def filter_bank(signal, sample_rate=16000, win_len=0.025, win_step=0.01,
                n_filters=26, fft_len=512, low_freq=0, high_freq=None, pre_emphasize=0.97,
                win_func=lambda x: numpy.ones((x,))):
    high_freq = high_freq or sample_rate / 2
    signal = sigproc.pre_emphasis(signal, pre_emphasize)
    frames = sigproc.frame_signal(signal, win_len * sample_rate, win_step * sample_rate, win_func)
    power_spec = sigproc.power_spec(frames, fft_len)
    energy = numpy.sum(power_spec, 1)  # this stores the total energy in each frame
    energy = numpy.where(energy == 0, numpy.finfo(float).eps, energy)  # if energy is zero, we get problems with log

    fb = get_filter_banks(n_filters, fft_len, sample_rate, low_freq, high_freq)
    feat = numpy.dot(power_spec, fb.T)  # compute the filterbank energies
    feat = numpy.where(feat == 0, numpy.finfo(float).eps, feat)  # if feat is zero, we get problems with log

    return feat, energy


def log_filter_bank(signal, sample_rate=16000, win_len=0.025, win_step=0.01,
                    n_filters=26, fft_len=512, low_freq=0, high_freq=None, pre_emphasize=0.97,
                    win_func=lambda x: numpy.ones((x,))):

    feat, energy = filter_bank(signal, sample_rate, win_len, win_step, n_filters, fft_len,
                               low_freq, high_freq, pre_emphasize, win_func)

    return numpy.log(feat)


def hz2mel(hz):

    return 2595 * numpy.log10(1 + hz / 700.)


def mel2hz(mel):

    return 700 * (10 ** (mel / 2595.0) - 1)


def get_filter_banks(n_filters=20, fft_len=512, sample_rate=16000, low_freq=0, high_freq=None):
    high_freq = high_freq or sample_rate / 2
    assert high_freq <= sample_rate / 2

    # compute points evenly spaced in mels
    low_mel = hz2mel(low_freq)
    high_mel = hz2mel(high_freq)
    mel_points = numpy.linspace(low_mel, high_mel, n_filters + 2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = numpy.floor((fft_len + 1) * mel2hz(mel_points) / sample_rate)

    fbank = numpy.zeros([n_filters, fft_len // 2 + 1])
    for j in range(0, n_filters):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])

    return fbank


def lifter(cepstral, lifter_coefficient=22):
    if lifter_coefficient > 0:
        n_frames, n_coefficient = numpy.shape(cepstral)
        n = numpy.arange(n_coefficient)
        lift = 1 + (lifter_coefficient / 2.) * numpy.sin(numpy.pi * n / lifter_coefficient)
        return lift * cepstral
    else:

        return cepstral


def delta(feat, step):
    if step < 1:
        raise ValueError('N must be an integer >= 1')
    n_frames = len(feat)
    denominator = 2 * sum([i ** 2 for i in range(1, step + 1)])
    delta_feat = numpy.empty_like(feat)
    padded = numpy.pad(feat, ((step, step), (0, 0)), mode='edge')  # padded version of feat
    for t in range(n_frames):
        delta_feat[t] = numpy.dot(numpy.arange(-step, step + 1),
                                  padded[t: t + 2 * step + 1]) / denominator  # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]

    return delta_feat


if __name__ == '__main__':
    wav_file = '../data/808-27.wav'
    sampling_rate, signal = wavfile.read(wav_file)
    features = mfcc(signal)
    print(features.shape)
