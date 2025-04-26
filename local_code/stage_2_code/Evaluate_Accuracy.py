'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):

        print('evaluating performance...')

        acc_test = accuracy_score(self.data['true_y'], self.data['pred_y'])
        precision_test = precision_score(self.data['true_y'], self.data['pred_y'], average='weighted', zero_division=0)
        recall_test = (recall_score(self.data['true_y'], self.data['pred_y'], average='weighted', zero_division=0))
        f1_test = (f1_score(self.data['true_y'], self.data['pred_y'], average='weighted', zero_division=0))

        return [acc_test, precision_test, recall_test, f1_test]


        