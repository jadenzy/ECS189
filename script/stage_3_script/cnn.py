from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import matplotlib.pyplot as plt


DATASET = 'ORL'
#DATASET = 'MNIST'
#DATASET = 'CIFAR'

def load_run_evaluate(method_obj, evaluate_obj, X_train, y_train, X_test, y_test):
    method_obj.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
    learned_result, loss = method_obj.run()

    evaluate_obj.data = learned_result
    score = evaluate_obj.evaluate()

    torch.save(method_obj.state_dict(), '../../result/stage_3_result/best_mlp_' + method_obj.method_description + '_model.pt')

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(loss)), loss, marker='o', color='blue', linestyle='-')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../../result/stage_3_result/loss_plot.png', dpi=300)
    plt.show()

    return score

if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_obj = Dataset_Loader('stage3_data', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = DATASET
    (train_X, train_y), (test_X, test_y) = data_obj.load()

    if data_obj.dataset_source_file_name == 'ORL':
        train_y = train_y - 1
        test_y = test_y - 1

    #method_obj = Method_CNN('CNN', data_obj.dataset_source_file_name, 3, 32, 32, 10)
    method_obj = Method_CNN('CNN', data_obj.dataset_source_file_name, 3, 112, 92, 40)
    method_obj.to(device)

    eva = Evaluate_Accuracy('accuracy, precision_score, recall_score, f1_score', '')

    print('************ Start ************')
    score = load_run_evaluate(
        method_obj, eva,
        train_X, train_y, test_X, test_y
    )

    print('************ Final Performance ************')
    print('Accuracy, Precision_score, Recall_score, F1_score: ' + ', '.join([f'{s:.4f}' for s in score]))
    print('************ Finish ************')