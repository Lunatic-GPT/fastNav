from torch.utils import data
import torch

from loupe_torch import train_utils

"""
simplely get the label_data from the source
for a self-supervised task, labels are equal to the label_data
"""


class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, train_path, label_path):
        super(Dataset, self).__init__()
        self.labels = torch.tensor(train_utils.data_preprocess_label(label_path))
        # self.raw = self.labels
        self.raw = torch.tensor(train_utils.data_preprocess_train(train_path))
    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.labels)

    def __getitem__(self, index):
        """Generates one sample of label_data"""
        # Load label_data and get label
        raw = self.raw[index]
        label = self.labels[index]

        return raw, label
