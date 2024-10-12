import json
import os
import numpy as np
from torch.utils.data import Dataset


def data_load(DATA_PATH):
    """
    According to the path, load the teeth data from the preprocessed json file.
    Return: feature (8-d vector), label (int:0-32), category ((1, 0) for mandible / (0, 1) for maxillary)
    """
    f = open(DATA_PATH, 'r')
    teeth_dict = json.load(f)
    feature = teeth_dict['feature']
    label = teeth_dict['label']
    category = teeth_dict['category']
    f.close()
    feature = np.array(feature).astype(np.float32)
    label = np.array(label).astype(np.int64)
    category = np.array(category).astype(np.float32)

    return feature, label, category


class Teeth(Dataset):
    def __init__(self, num_points, ROOT_PATH, partition):
        np.random.seed(42)
        self.num_points = num_points
        self.data_paths = []
        self.data_paths_pure = []
        for folder in os.listdir(ROOT_PATH):
            FOLDER_PATH = os.path.join(ROOT_PATH, folder)
            if os.path.isdir(FOLDER_PATH):
                symbol = os.listdir(FOLDER_PATH)[0][0]
                self.data_paths_pure.append(os.path.join(FOLDER_PATH, symbol + "_aligned.json"))

        if partition == "val":
            self.data_paths = self.data_paths_pure[:2000]
            return
        if partition == "test":
            self.data_paths = self.data_paths_pure[-2000:]
            return
        self.data_paths = self.data_paths_pure[0:4000]  # use 4000 samples for training

    def __getitem__(self, item):
        pointcloud, label, category = data_load(self.data_paths[item])
        indices = np.random.choice(pointcloud.shape[0], self.num_points, replace=False)
        pointcloud = pointcloud[indices]
        label = label[indices]

        return pointcloud, category, label

    def __len__(self):
        return len(self.data_paths)


if __name__ == '__main__':
    ROOT_PATH = '/data'
    val = Teeth(10000, ROOT_PATH, partition="val")
    for a in val:
        pass
