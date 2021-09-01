import torch
from torch.utils.data import Dataset
import random
import numpy
from scipy.io import wavfile
from extractors.MFCC.mfcc import mfcc


class VoicePairDataset(Dataset):
    def __init__(self, data, max_sequence_len):
        super(VoicePairDataset, self).__init__()
        self.length = len(data)
        self.data = self.get_data(data)
        self.max_sequence_len = max_sequence_len

    def __getitem__(self, item):
        label_voice1 = random.choice(list(self.data.keys()))
        value_voice1 = random.choice(self.data[label_voice1])

        should_get_the_same_class = random.randint(0, 1)
        if should_get_the_same_class:
            limit = 0
            while True:
                value_voice2 = random.choice(self.data[label_voice1])
                limit += 1
                if value_voice2 != value_voice1 or limit > 3:
                    break

        else:
            while True:
                label_voice2 = random.choice(list(self.data.keys()))
                if label_voice2 == label_voice1:
                    continue
                value_voice2 = random.choice(self.data[label_voice2])
                break

        voice_1 = self.transform(value_voice1)
        voice_1 = torch.from_numpy(voice_1.astype(numpy.float32))
        voice_1 = voice_1.unsqueeze(0)

        voice_2 = self.transform(value_voice2)
        voice_2 = torch.from_numpy(voice_2.astype(numpy.float32))
        voice_2 = voice_2.unsqueeze(0)

        return voice_1, voice_2, torch.tensor(should_get_the_same_class, dtype=torch.long)

    def __len__(self):

        return self.length

    @staticmethod
    def get_data(self, data):
        dict_data = {}
        for elem in data:
            if elem[1] not in dict_data.keys():
                dict_data[elem[1]] = [elem[0]]
            else:
                dict_data[elem[1]].append(elem[0])

        return dict_data

    def transform(self, voice_path):
        sampling_rate, signal = wavfile.read(voice_path)
        features = mfcc(signal)

        if features.shape[0] < self.max_sequence_len:
            new_features = numpy.zeros((self.max_sequence_len - features.shape[0], features.shape[1]))
            new_features = numpy.concatenate((features, new_features), axis=0)
        else:
            new_features = features[:self.max_sequence_len, :]

        return new_features


if __name__ == '__main__':
    data = [('../data/812.wav', 0), ('../data/808-27.wav', 1), ('../data/812.wav', 1)]
    dataset = VoicePairDataset(data, max_sequence_len=2048)
    print(dataset[0])
    pass
