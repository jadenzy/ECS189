import json
import torch
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation

def load_model(model_path, vocab_size):
    model = Method_RNN_Generation(vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    return model

with open("../../result/stage_4_result/vocab.json", encoding="utf-8") as f:
    vocab = json.load(f)

idx2word = {v: k for k, v in vocab.items()}

model_path = '../../result/stage_4_result/Gen_model.pt'
model = load_model(model_path, 20000)
model.vocab = vocab
model.idx2word = idx2word

while True:
    prompt = input("Enter prompt (or 'exit'): ")
    if prompt.strip().lower() == "exit":
        break
    prompt_words = prompt.strip().split()
    output = model.generate(prompt_words, vocab, idx2word)
    print(f"Generated: {output}")
