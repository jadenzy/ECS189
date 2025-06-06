from local_code.base_class.evaluate import evaluate
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score


class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate(self):
        print('Evaluating performance on multi-class classification...')

        y_true = self.data['true_y']
        y_pred = self.data['pred_y']

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # ROC AUC is not directly applicable to multi-class unless using one-vs-rest strategy.
        roc_auc = float('nan')

        print(f'Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
        return [acc, precision, recall, f1, roc_auc]
