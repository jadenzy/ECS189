'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import pickle
import numpy as np

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()


        train_images = np.array([instance['image'] for instance in data['train']])
        train_labels = np.array([instance['label'] for instance in data['train']])

        test_images = np.array([instance['image'] for instance in data['test']])
        test_labels = np.array([instance['label'] for instance in data['test']])

        print(f"Loaded {len(train_images)} training images, {len(test_images)} testing images")

        return (train_images, train_labels), (test_images, test_labels)

