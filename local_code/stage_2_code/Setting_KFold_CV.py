'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np
import torch
import matplotlib.pyplot as plt

class Setting_KFold_CV(setting):
    fold = 3

    
    def load_run_save_evaluate(self):
        global best_model
        best_score = -1
        best_fold_data = None
        best_loss = []
        # load dataset
        loaded_data = self.dataset.load()

        
        kf = KFold(n_splits=self.fold, shuffle=True)
        
        fold_count = 0
        score_list = []
        for train_index, test_index in kf.split(loaded_data['X']):
            fold_count += 1
            print('************ Fold:', fold_count, '************')
            X_train, X_test = np.array(loaded_data['X'])[train_index], np.array(loaded_data['X'])[test_index]
            y_train, y_test = np.array(loaded_data['y'])[train_index], np.array(loaded_data['y'])[test_index]
        
            # run MethodModule
            self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
            learned_result, loss = self.method.run()
            
            # save raw ResultModule
            self.result.data = learned_result
            self.result.fold_count = fold_count
            self.result.save()
            
            self.evaluate.data = learned_result
            score = self.evaluate.evaluate()
            score_list.append(score)

            if score[0] > best_score:
                best_loss = loss
                best_score = score[0]
                best_model = self.method.state_dict()


            torch.save(best_model, '../../result/stage_2_result/best_mlp_model.pt')

        plt.figure(figsize=(8, 5))
        plt.plot(range(len(best_loss)), best_loss, marker='o', color='blue', linestyle='-')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('../../result/stage_2_result/loss_plot.png', dpi=300)
        plt.show()

        return np.mean(score_list, axis=0), np.std(score_list, axis=0)



        