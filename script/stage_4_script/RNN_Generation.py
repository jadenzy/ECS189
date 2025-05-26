import numpy as np
import torch
from local_code.stage_4_code.Gen_Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation

if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load generative text dataset
    data_obj = Dataset_Loader('Joke_Generation', '')

    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_generation/data'
    train_texts, test_texts = data_obj.load()

    # Instantiate generative RNN model
    model = Method_RNN_Generation(vocab_size=20000)
    model.to(device)

    print('************ Start Training and Generating ************')
    model.run(train_texts, test_texts)
    print('************ Done ************')
