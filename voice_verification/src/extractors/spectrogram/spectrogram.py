from scipy.signal import spectrogram
import numpy
import librosa


def get_spec(wav_path, max_time=3):
    audio, sample_rate = librosa.load(wav_path, sr=16000)
    audio_size = len(audio)

    max_audio = max_time * 16000 + 160 * 2
    if audio_size <= max_audio:
        shortage = max_audio - audio_size
        audio = numpy.pad(audio, (0, shortage), 'wrap')
    else:
        audio = audio[: max_audio]

    f, t, spec = spectrogram(audio, fs=16000, window="hamming", nperseg=400, noverlap=240, nfft=1024)
    spec = spec[1:]
    spec = (spec - numpy.mean(spec, axis=0)) / numpy.std(spec, axis=0)
    print(spec.shape)

    return spec


if __name__ == '__main__':
    wav_path = '/home/doanphu/Documents/Code/VND_project/zalo-ai-challange-2020/voice_verification/data/808-27.wav'
    get_spec(wav_path=wav_path)
    
    pass
