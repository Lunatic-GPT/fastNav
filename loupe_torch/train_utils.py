import numpy as np
import torch


def data_preprocess_train(data_path):
    data = np.load(data_path).astype(float)

    data = np.pad(data, ((0, 0), (3, 3), (7, 7), (5, 5), (0, 0)), 'constant', constant_values=(0, 0))
    data = torch.tensor(data)

    return data


def single_data_preprocess_train(data_path):
    data = np.load(data_path).astype(float)
    data = np.expand_dims(data, 0)

    data = np.pad(data, ((0, 0), (3, 3), (7, 7), (5, 5), (0, 0)), 'constant', constant_values=(0, 0))
    data = torch.tensor(data)

    return data


def data_preprocess_label(data_path):
    data = np.load(data_path).astype(float)
    data = torch.tensor(data)

    return data
