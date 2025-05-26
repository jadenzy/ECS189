import csv
import random
from local_code.base_class.dataset import dataset

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self, split_ratio=0.9, seed=42):
        print('Loading text data for generation...')

        all_texts = []
        with open(self.dataset_source_folder_path, encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2 and row[1].strip():
                    all_texts.append(row[1].strip())

        print(f'Total samples loaded: {len(all_texts)}')

        random.seed(seed)
        random.shuffle(all_texts)
        split_index = int(len(all_texts) * split_ratio)
        train_texts = all_texts[:split_index]
        test_texts = all_texts[split_index:]

        print(f'Split into {len(train_texts)} training and {len(test_texts)} testing samples.')
        return train_texts, test_texts
