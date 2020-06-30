import os
import pandas as pd
from PIL.Image import open
from torch.utils.data import Dataset


class XRayDataSet(Dataset):
    def __init__(self, filedir, metafile, transforms=None):
        self.filedir = filedir
        self.filelist = os.listdir(filedir)
        self.metadata = pd.read_csv(metafile)
        self.transforms = transforms
        self.label_dict = {'Normal': 0, 'Other Pnemonia': 1, 'COVID-19': 2}

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        data = open(os.path.join(self.filedir, self.filelist[index]))
        data = data.convert('RGB')
        if self.transforms:
            data = self.transforms(data)
        meta = self.metadata[self.metadata['Filename'] == self.filelist[index]]
        label = self.label_dict[meta['Label'].values[0]]
        label = int(label)
        return data, label
