import torch
import torch.nn as nn
from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, A_hat):
        return self.linear(torch.matmul(A_hat, X))


class Method_GCN_Classification(method, nn.Module):
    max_epoch = 150
    learning_rate = 0.01
    hidden_dim = 32
    dropout_rate = 0.3

    def __init__(self, mName, mDescription, input_dim, num_classes):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gc1 = GCNLayer(input_dim, self.hidden_dim)
        self.gc2 = GCNLayer(self.hidden_dim, self.hidden_dim)
        self.gc3 = GCNLayer(self.hidden_dim, num_classes)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.to(self.device)

    def forward(self, X, A_hat):
        X = self.gc1(X, A_hat)
        X = torch.relu(X)
        X = self.dropout(X)
        X = self.gc2(X, A_hat)
        X = torch.relu(X)
        X = self.dropout(X)
        X = self.gc3(X, A_hat)
        return X

    def fit(self, features, adj_matrix, labels, train_idx, test_idx):
        losses = []
        features = features.to(self.device)
        labels = labels.to(self.device)
        A_hat = adj_matrix.to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        evaluator = Evaluate_Accuracy('eval', '')

        best_val_f1 = 0
        best_epoch = 0
        best_val_loss = 100

        for epoch in range(self.max_epoch):
            self.train()
            logits = self.forward(features, A_hat)
            loss = criterion(logits[train_idx], labels[train_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluation
            self.eval()
            with torch.no_grad():
                preds = logits[train_idx].argmax(dim=1).cpu().numpy()
                evaluator.data = {
                    'true_y': labels[train_idx].cpu().numpy(),
                    'pred_y': preds
                }
                acc, _, _, f1 = evaluator.evaluate()
                print(f"Epoch {epoch + 1}: Accuracy={acc:.4f}, F1={f1:.4f}, Loss={loss.item():.4f}")

            if f1 > best_val_f1:
                best_val_f1 = f1
                best_epoch = epoch
            # if loss.item() < best_val_loss:
            #     best_val_loss = loss.item()
            #     best_epoch = epoch
            elif epoch - best_epoch >= 10:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            losses.append(loss.item())

        return logits, losses
