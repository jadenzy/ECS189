from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import matplotlib.pyplot as plt

def load_run_save_evaluate(method_obj, result_obj, evaluate_obj, X_train, y_train, X_test, y_test):

    method_obj.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
    learned_result, loss = method_obj.run()

    result_obj.data = learned_result
    result_obj.save()

    evaluate_obj.data = learned_result
    score = evaluate_obj.evaluate()


    torch.save(method_obj.state_dict(), '../../result/stage_2_result/best_mlp_model.pt')

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(loss)), loss, marker='o', color='blue', linestyle='-')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../../result/stage_2_result/loss_plot.png', dpi=300)
    plt.show()

    return score

if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_obj = Dataset_Loader('stage2_data', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.dataset_source_file_name = 'train.csv'
    train_data = data_obj.load()


    test_data_obj = Dataset_Loader('stage2_test_data', '')
    test_data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    test_data_obj.dataset_source_file_name = 'test.csv'
    test_data = test_data_obj.load()

    method_obj = Method_MLP('multi-layer perceptron', '')
    method_obj.to(device)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    eva = Evaluate_Accuracy('accuracy, precision_score, recall_score, f1_score', '')

    print('************ Start ************')
    score = load_run_save_evaluate(
        method_obj, result_obj, eva,
        train_data['X'], train_data['y'], test_data['X'], test_data['y']
    )

    print('************ Final Performance ************')
    print('Accuracy, Precision_score, Recall_score, F1_score: ' + ', '.join([f'{s:.4f}' for s in score]))
    print('************ Finish ************')



