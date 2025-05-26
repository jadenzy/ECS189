import os
import re
import torch
import json
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, texts, vocab, max_len):
        self.vocab = vocab
        self.max_len = max_len
        self.data = self.prepare_sequences(texts)

    def prepare_sequences(self, texts):
        sequences = []
        for tokens in texts:
            token_ids = [self.vocab.get(tok, self.vocab['<UNK>']) for tok in tokens]
            token_ids = [self.vocab.get('<START>', self.vocab['<UNK>'])] + token_ids + \
                        [self.vocab.get('<END>', self.vocab['<UNK>'])]

            if len(token_ids) < self.max_len + 1:
                padding = [self.vocab['<PAD>']] * (self.max_len + 1 - len(token_ids))
                token_ids += padding
            else:
                token_ids = token_ids[:self.max_len + 1]

            input_seq = token_ids[:-1]
            target_seq = token_ids[1:]

            sequences.append((torch.tensor(input_seq), torch.tensor(target_seq)))
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Method_RNN_Generation(nn.Module):
    max_epoch = 250
    learning_rate = 1e-3
    batch_size = 64
    max_len = 40
    embed_dim = 300
    hidden_dim = 256
    dropout = 0.2
    num_layers = 2

    def __init__(self, vocab_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding = nn.Embedding(vocab_size, self.embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            self.embed_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.hidden_dim, vocab_size)

        self.to(self.device)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        logits = self.fc(output)
        return logits, hidden

    def fit(self, X, vocab):
        dataset = TextDataset(X, vocab, self.max_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])

        for epoch in range(self.max_epoch):
            self.train()
            total_loss = 0
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                logits, _ = self.forward(inputs)

                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)

            print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

    def generate(self, start_words, vocab, idx2word, max_gen_len=40, temperature=0.7):
        self.eval()
        with torch.no_grad():
            input_ids = [vocab.get('<START>', vocab['<UNK>'])] + \
                        [vocab.get(w, vocab['<UNK>']) for w in start_words]

            input_seq = torch.tensor([input_ids[-self.max_len:]], device=self.device)
            generated = []
            hidden = None

            for _ in range(max_gen_len):
                logits, hidden = self.forward(input_seq, hidden)
                logits = logits[:, -1, :]

                prob = torch.softmax(logits.squeeze(0) / temperature, dim=-1)
                next_word_id = torch.multinomial(prob, num_samples=1).item()

                if next_word_id == vocab.get('<END>', -1):
                    break

                generated.append(next_word_id)
                input_seq = torch.tensor([[next_word_id]], device=self.device)

            filtered = [idx2word.get(idx, '<UNK>') for idx in generated
                        if idx not in {vocab.get('<START>', -1), vocab.get('<END>', -1), vocab['<PAD>']}]
            return ' '.join(filtered)

    def build_vocab_and_clean_data(self, train_texts, test_texts, max_vocab_size=20000, min_freq=1):
        def clean_text(text):
            text = text.lower()
            text = re.sub(r"[^a-zA-Z0-9\s.,!?\'\"-]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        print('-- Cleaning and Building Vocabulary --')
        all_texts = train_texts + test_texts
        all_tokens = []
        cleaned_texts = []

        for text in tqdm(all_texts, desc='Cleaning text'):
            tokens = clean_text(text).split()
            cleaned_texts.append(tokens)
            all_tokens.extend(tokens)

        train_size = len(train_texts)
        self.train_tokens = cleaned_texts[:train_size]
        self.test_tokens = cleaned_texts[train_size:]

        # Count and keep most frequent words
        counter = Counter(all_tokens)

        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>', '<PUNCHLINE>']
        filtered_words = [word for word, count in counter.items()
                          if count >= min_freq and word not in special_tokens]
        filtered_words = sorted(filtered_words, key=lambda w: counter[w], reverse=True)[
                         :max_vocab_size - len(special_tokens)]

        self.word2idx = {token: i for i, token in enumerate(special_tokens)}
        self.word2idx.update({word: i + len(special_tokens)
                              for i, word in enumerate(filtered_words)})
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab = self.word2idx  # Add this line to store the vocab

    def run(self, train_texts, test_texts):
        print('-- Cleaning and Training --')
        self.build_vocab_and_clean_data(train_texts, test_texts)

        # Fit and generate
        self.fit(self.train_tokens, self.vocab)  # Pass self.vocab here

        torch.save(self.state_dict(), "../../result/stage_4_result/Gen_model.pt")
        with open("../../result/stage_4_result/vocab.json", "w", encoding="utf-8") as f:
            json.dump({k: int(v) for k, v in self.vocab.items()}, f)

        print('\n-- Sample Generations --')
        prompts = [["what", "did", "the"]]
        for p in prompts:
            output = self.generate(p, self.vocab, self.idx2word)
            print(f'Prompt: {" ".join(p)}\nGenerated: {output}\n')




