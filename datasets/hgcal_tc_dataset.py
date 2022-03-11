
import pickle # switch to something else (npz or hdf5 probably)
import numpy as np
import torch
from torch.utils.data import Dataset

class HGCalTCModuleDataset(Dataset):
    '''
    Data format for individual wafers using (cellu, cellv) formatting.  This is
    the first prototype of the econ dataset and will not include layer of wafer
    location information.
    '''
    def __init__(self, input_files, targets=None, transform=None):
        self.input_files = input_files

        # add something meaningful for these
        self.targets     = targets
        self.transform   = transform

        # unpack the data into 8x8 tensors
        self.wafer_data = self.unpack_wafer_data()

    def __getitem__(self, index):
        x = self.wafer_data[index]
        #y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x#, y

    def __len__(self):
        return len(self.wafer_data)

    def unpack_wafer_data(self):
        wafers = []
        for filename in self.input_files:
            f = open(filename, 'rb')
            data = pickle.load(f)
            for (event, zside), event_data in data.items():
                for (waferu, waferv), tc_stack in event_data.items():
                    wafers.append(tc_stack)

        # temporary: stack all wafers and remove empty wafers (those should be trained on though)
        wafers = np.vstack(wafers)
        wafers_sums = wafers.sum(axis=(1, 2))
        wafers = wafers[wafers_sums > 20]

        return torch.tensor(wafers).float()
