import numpy as np
import torch
from local_code.stage_5_code.Dataset_Loader import Dataset_Loader  # updated loader
from local_code.stage_5_code.Method_GCN import Method_GCN_Classification
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import matplotlib.pyplot as plt

def load_run_evaluate(method_obj, evaluate_obj, features, labels, adj, train_idx, test_idx):
    logits, losses = method_obj.fit(features, adj, labels, train_idx, test_idx)

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(losses)), losses, marker='o', color='blue', linestyle='-')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'../../result/stage_5_result/{method_obj.method_description}_loss_plot.png', dpi=800)
    plt.show()

    pred_y = logits[test_idx].argmax(dim=1).cpu().numpy()
    true_y = labels[test_idx].cpu().numpy()

    evaluate_obj.data = {'pred_y': pred_y, 'true_y': true_y}
    score = evaluate_obj.evaluate()

    torch.save(method_obj.state_dict(), f'../../result/stage_5_result/best_gcn_{method_obj.method_description}_model.pt')
    return score

if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = 'pubmed'  # 'cora', 'citeseer', 'pubmed'

    data_obj = Dataset_Loader()
    data_obj.dataset_name = dataset_name
    data_obj.dataset_source_folder_path = f'../../data/stage_5_data/{dataset_name}'

    data = data_obj.load()
    graph = data['graph']
    train_idx = data['train_test_val']['idx_train']
    test_idx = data['train_test_val']['idx_test']

    features = graph['X']
    labels = graph['y']
    adj = graph['utility']['A']

    input_dim = features.shape[1]
    num_classes = labels.max().item() + 1

    method_obj = Method_GCN_Classification('GCN', f'GCN on {dataset_name}', input_dim=input_dim, num_classes=num_classes)
    method_obj.to(device)

    eva = Evaluate_Accuracy('accuracy, precision_score, recall_score, f1_score', '')

    print('************ Start ************')
    score = load_run_evaluate(method_obj, eva, features, labels, adj, train_idx, test_idx)
    print('************ Final Performance ************')
    print('Accuracy, Precision_score, Recall_score, F1_score, ROC_AUC: ' + ', '.join([f'{s:.4f}' for s in score]))
    print('************ Finish ************')
