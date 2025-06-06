import os
import re
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.data = [self.text_to_sequence(text, vocab, max_len) for text in texts]
        self.labels = labels
        self.max_len = max_len

    def text_to_sequence(self, text, vocab, max_len):
        tokens = text
        seq = [vocab.get(word, vocab['<UNK>']) for word in tokens]
        if len(seq) < max_len:
            seq += [vocab['<PAD>']] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        return torch.tensor(seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class Method_RNN_Classification(method, nn.Module):
    max_epoch = 10
    learning_rate = 2e-3
    batch_size = 128
    max_len = 800
    embed_dim = 120
    hidden_dim = 128

    def __init__(self, mName, mDescription, vocab_size, num_classes=2):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding = nn.Embedding(vocab_size, self.embed_dim, padding_idx=0)
        self.rnn = nn.GRU(self.embed_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.rnn2 = nn.GRU(self.hidden_dim * 2, self.hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)

        self.to(self.device)

    def forward(self, x):
        x = self.embedding(x)
        out1, _ = self.rnn(x)

        #For LSTM
        #out2, (hn, _) = self.rnn2(out1)

        # For GRU
        out2, hn = self.rnn2(out1)

        last_hidden = torch.cat([hn[-2], hn[-1]], dim=1)

        dropped = self.dropout(last_hidden)
        out = self.fc(dropped)

        return out

    def fit(self, X, y, vocab):
        dataset = TextDataset(X, y, vocab, self.max_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        evaluator = Evaluate_Accuracy('eval', '')

        loss_list = []
        for epoch in range(self.max_epoch):
            self.train()
            total_loss, total_correct, total = 0, 0, 0

            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.forward(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_X.size(0)
                preds = logits.argmax(dim=1)
                total_correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

            test_preds = self.test(self.data['train']['X'], vocab)
            evaluator.data = {'true_y': self.data['train']['y'], 'pred_y': test_preds.numpy()}
            acc, _, _, f1 = evaluator.evaluate()
            print(f"Epoch {epoch}: Accuracy={acc:.4f}, F1={f1:.4f}, Loss={total_loss/total:.4f}")
            loss_list.append(total_loss/total)

        return loss_list

    def test(self, X, vocab):
        self.eval()
        dataset = TextDataset(X, [0] * len(X), vocab, self.max_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        preds = []

        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.forward(batch_X)
                batch_preds = outputs.argmax(dim=1).cpu()
                preds.append(batch_preds)

        return torch.cat(preds, dim=0)

    def run(self):
        print('-- Cleaning and Training --')
        self.build_vocab_and_clean_data()
        loss = self.fit(self.data['train']['X'], self.data['train']['y'], self.vocab)
        pred_y = self.test(self.data['test']['X'], self.vocab)
        return {'pred_y': pred_y.cpu().numpy(), 'true_y': np.array(self.data['test']['y'])}, loss


    def build_vocab_and_clean_data(self, min_freq=5, max_vocab_size=25000):
        def clean_text(text):
            text = text.lower()
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        print('-- Cleaning and Building Vocabulary --')

        all_texts = self.data['train']['X'] + self.data['test']['X']
        all_tokens = []

        cleaned_texts = []

        for text in tqdm(all_texts, desc='Cleaning text'):
            text = clean_text(text)
            tokens = text.split()
            cleaned_texts.append(tokens)
            all_tokens.extend(tokens)

        # Save cleaned tokens back to dataset
        train_size = len(self.data['train']['X'])
        self.data['train']['X'] = cleaned_texts[:train_size]
        self.data['test']['X'] = cleaned_texts[train_size:]

        print('-- Building Vocabulary --')

        counter = Counter(all_tokens)
        most_common = [word for word, freq in counter.items() if freq >= min_freq]
        most_common = most_common[:max_vocab_size - 2]  # leave space for PAD and UNK

        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        for idx, word in enumerate(most_common, start=2):
            self.word2idx[word] = idx

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        self.vocab = self.word2idx

        print(f'Final vocab size: {self.vocab_size}')
