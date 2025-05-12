from torch.onnx.symbolic_opset14 import batch_norm

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 50
    learning_rate = 1e-3
    batch_size = 128

    def __init__(self, mName, mDescription, input_channels, image_height, image_width, num_classes):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.pool5 = nn.MaxPool2d(2, 2)



        input_size = torch.zeros(1, input_channels, image_height, image_width)
        output_size = self._forward_conv(input_size)
        flattened_size = output_size.view(1, -1).shape[1]

        self.fc1 = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Linear(128, num_classes)
        self.to(self.device)

    def _forward_conv(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        if x.shape[2] >= 2 and x.shape[3] >= 2:
            x = self.pool5(self.conv5(x))
        else:
            x = self.conv5(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def fit(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('testing evaluator', '')

        X_tensor = torch.FloatTensor(np.array(X))
        if X_tensor.dim() == 3:
            X_tensor = X_tensor.unsqueeze(1)
        elif X_tensor.dim() == 4:
            X_tensor = X_tensor.permute(0, 3, 1, 2)
        y_tensor = torch.LongTensor(np.array(y))
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        loss_list = []

        for epoch in range(self.max_epoch):
            self.train()  # explicitly set to train mode
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                y_pred = self.forward(batch_X)
                train_loss = loss_function(y_pred, batch_y)
                train_loss.backward()
                optimizer.step()

                epoch_loss += train_loss.item() * batch_X.size(0)
                preds = y_pred.max(1)[1]
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)


            test_pred_y = self.test(self.data['test']['X'])
            accuracy_evaluator.data = {
                'true_y': self.data['test']['y'],
                'pred_y': test_pred_y.numpy()
            }
            acc, _, _, f1 = accuracy_evaluator.evaluate()
            avg_loss = epoch_loss / total

            print(f'Epoch: {epoch}, Train Accuracy: {acc:.4f}, F1-score: {f1:.4f}, Train Loss: {avg_loss:.4f}')
            loss_list.append(avg_loss)

        return loss_list

    def test(self, X):
        X_tensor = torch.FloatTensor(np.array(X))
        if X_tensor.dim() == 3:
            X_tensor = X_tensor.unsqueeze(1)
        elif X_tensor.dim() == 4:
            X_tensor = X_tensor.permute(0, 3, 1, 2)
        X_tensor = X_tensor.to(self.device)
        preds_list = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), self.batch_size):
                batch_X = X_tensor[i:i+self.batch_size]
                y_pred = self.forward(batch_X)
                preds = y_pred.max(1)[1]
                preds_list.append(preds.cpu())
        return torch.cat(preds_list, dim=0)

    def run(self):
        print('method running...')
        print('--start training...')
        loss = self.fit(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}, loss
