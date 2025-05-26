import numpy as np
import torch
import matplotlib.pyplot as plt
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy

def load_run_evaluate(method_obj, evaluate_obj, X_train, y_train, X_test, y_test):
    method_obj.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
    learned_result, loss = method_obj.run()

    evaluate_obj.data = learned_result
    score = evaluate_obj.evaluate()

    # Save model
    torch.save(method_obj.state_dict(), '../../result/stage_4_result/best_rnn_' + method_obj.method_description + '_model.pt')

    # Plot loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(loss)), loss, marker='o', color='blue', linestyle='-')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../../result/stage_4_result/loss_plot.png', dpi=300)
    plt.show()

    return score


if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load text classification dataset
    data_obj = Dataset_Loader('IMDb_text_data', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification/'
    (train_X, train_y), (test_X, test_y) = data_obj.load()

    # Create and run RNN
    method_obj = Method_RNN_Classification('RNN', 'RNN for IMDb reviews', vocab_size=25000)
    method_obj.to(device)

    # Evaluation
    eva = Evaluate_Accuracy('accuracy, precision_score, recall_score, f1_score, roc_auc', '')

    print('************ Start ************')
    score = load_run_evaluate(
        method_obj, eva,
        train_X, train_y, test_X, test_y
    )

    print('************ Final Performance ************')
    print('Accuracy, Precision_score, Recall_score, F1_score, ROC_AUC: ' + ', '.join([f'{s:.4f}' for s in score]))
    print('************ Finish ************')
