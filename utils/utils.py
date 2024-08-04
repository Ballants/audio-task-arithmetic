import os
import pickle

import numpy as np


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def save_data(train_batches=None, val_batches=None, test_batches=None, train_path=None, val_path=None, test_path=None):
    if train_batches is not None and val_batches is not None:
        with open(train_path, 'wb') as f:
            pickle.dump(train_batches, f)
        with open(val_path, 'wb') as f:
            pickle.dump(val_batches, f)
    with open(test_path, 'wb') as f:
        pickle.dump(test_batches, f)


def load_data(dir_data, train=True):
    if train:
        with open(dir_data + 'train_batches.pkl', 'rb') as f:
            train_batches = pickle.load(f)
        with open(dir_data + 'val_batches.pkl', 'rb') as f:
            val_batches = pickle.load(f)
        return train_batches, val_batches
    else:
        with open(dir_data + 'test_batches.pkl', 'rb') as f:
            test_batches = pickle.load(f)
        return test_batches


def one_hot_encoding(index, max_len):
    one_hot = np.zeros(max_len)
    one_hot[index] = 1
    return one_hot


def process_batches(audio_data, label_data, batch_size, processor, candidate_labels):
    for i in range(0, len(audio_data), batch_size):
        batch_audio = audio_data[i:i + batch_size]
        batch_labels = label_data[i:i + batch_size]

        processed_audio = processor(text=candidate_labels, audios=batch_audio, return_tensors="pt",
                                    padding=True, sampling_rate=48000)

        yield processed_audio, batch_labels
