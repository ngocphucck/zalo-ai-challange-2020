import decimal
import numpy
import math
import logging
import matplotlib.pyplot as plt

from scipy.io import wavfile


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def rolling_window(a, window, step=1):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


def frame_signal(signal, frame_len, frame_step, win_func=lambda x: numpy.ones((x,)), stride_trick=True):
    signal_len = len(signal)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))

    if signal_len <= frame_len:
        num_frames = 1
    else:
        num_frames = 1 + int(math.ceil((1.0 * signal_len - frame_len) / frame_step))

    pad_len = int((num_frames - 1) * frame_step + frame_len)

    zeros = numpy.zeros((pad_len - signal_len,))
    pad_signal = numpy.concatenate((signal, zeros))

    if stride_trick:
        win = win_func(frame_len)
        frames = rolling_window(pad_signal, window=frame_len, step=frame_step)
    else:
        indices = numpy.tile(numpy.arange(0, frame_len), (num_frames, 1)) + numpy.tile(
            numpy.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
        indices = numpy.array(indices, dtype=numpy.int32)
        frames = pad_signal[indices]
        win = numpy.tile(win_func(frame_len), (num_frames, 1))

    return frames * win


def de_frame_signal(frames, signal_len, frame_len, frame_step, win_func=lambda x: numpy.ones((x,))):
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    num_frames = numpy.shape(frames)[0]
    assert numpy.shape(frames)[1] == frame_len

    indices = numpy.tile(numpy.arange(0, frame_len), (num_frames, 1)) + numpy.tile(
        numpy.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
    indices = numpy.array(indices, dtype=numpy.int32)
    pad_len = (num_frames - 1) * frame_step + frame_len

    if signal_len <= 0:
        signal_len = pad_len

    rec_signal = numpy.zeros((pad_len,))
    window_correction = numpy.zeros((pad_len,))
    win = win_func(frame_len)

    for i in range(0, num_frames):
        window_correction[indices[i, :]] = window_correction[
                                               indices[i, :]] + win + 1e-15  # add a little bit so it is never zero
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal = rec_signal / window_correction
    return rec_signal[0:signal_len]


def mag_spec(frames, fft_len):
    if numpy.shape(frames)[1] > fft_len:
        logging.warning(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            numpy.shape(frames)[1], fft_len)
    complex_spec = numpy.fft.rfft(frames, fft_len)
    return numpy.absolute(complex_spec)


def power_spec(frames, fft_len):

    return 1.0 / fft_len * numpy.square(mag_spec(frames, fft_len))


def log_power_spec(frames, fft_len, norm=1):
    ps = power_spec(frames, fft_len)
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * numpy.log10(ps)
    if norm:
        return lps - numpy.max(lps)
    else:
        return lps


def pre_emphasis(signal, coefficient=0.95):

    return numpy.append(signal[0], signal[1:] - coefficient * signal[:-1])


if __name__ == '__main__':
    wav_file = '/home/doanphu/Documents/Code/VND_project/zalo-ai-challange-2020/voice_verification/data/808-27.wav'
    sampling_rate, signal = wavfile.read(wav_file)

    print(signal)
    frames = frame_signal(signal, frame_len=25, frame_step=15)
    print(frames)
    plt.plot(frames[0])
    plt.show()
    pass
