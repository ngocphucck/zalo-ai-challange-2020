from glob import glob
import os


def get_voice_names(voice_folder):

    return glob(voice_folder + "/*.wav")


def encode_name(data_folder):

    return {
        os.listdir(data_folder)[i]: i for i in range(len(os.listdir(data_folder)))
    }


def get_data(data_folder):
    data = []
    dictionary = encode_name(data_folder)

    for person_name in os.listdir(data_folder):
        for voice_name in os.listdir(os.path.join(data_folder, person_name)):
            data.append((os.path.join(data_folder, person_name, voice_name), dictionary[person_name]))

    return data

