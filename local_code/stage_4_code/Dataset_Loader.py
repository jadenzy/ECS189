import os
from local_code.base_class.dataset import dataset

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('Loading IMDb text data...')

        def load_reviews_from_folder(folder_path, label):
            texts = []
            labels = []
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    with open(file_path, encoding='utf-8') as f:
                        texts.append(f.read())

                        labels.append(label)
            return texts, labels

        train_pos_path = os.path.join(self.dataset_source_folder_path, 'train', 'pos')
        train_neg_path = os.path.join(self.dataset_source_folder_path, 'train', 'neg')
        test_pos_path = os.path.join(self.dataset_source_folder_path, 'test', 'pos')
        test_neg_path = os.path.join(self.dataset_source_folder_path, 'test', 'neg')

        train_texts_pos, train_labels_pos = load_reviews_from_folder(train_pos_path, 1)
        train_texts_neg, train_labels_neg = load_reviews_from_folder(train_neg_path, 0)
        test_texts_pos, test_labels_pos = load_reviews_from_folder(test_pos_path, 1)
        test_texts_neg, test_labels_neg = load_reviews_from_folder(test_neg_path, 0)

        train_texts = train_texts_pos + train_texts_neg
        train_labels = train_labels_pos + train_labels_neg
        test_texts = test_texts_pos + test_texts_neg
        test_labels = test_labels_pos + test_labels_neg

        print(f"Loaded {len(train_texts)} training samples and {len(test_texts)} testing samples.")

        return (train_texts, train_labels), (test_texts, test_labels)
