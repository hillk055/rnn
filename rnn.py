from transformers import AutoTokenizer
from pathlib import Path
import torch.nn as nn

text = Path('corpus.txt').read_text(encoding='utf-8')   # Encoding in utf-8 is standard practice

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encoded = tokenizer(text, return_tensors='pt', truncation=True, add_special_tokens=True, return_overflowing_tokens=True,
                    stride=32, max_length=96, padding='max_length')
# Set truncation True and Max Length to limit the sie of the text
# Special tokens are either sentence beginners or sentences breakers, marked in so the model has extra context

# Next we need to batch inputs, we need some overlap so that the model can learn conditionality.
# E.g. I liked up at the sky. The stars looked good. The model now knows that sky and starts can be used together

print(encoded['input_ids'].shape) # Returns number of batches and the size of each batch
vocab_sz = tokenizer.vocab_size


class RNNLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embedding_dim=64)
        # Expects the first dimension of the tensor to be batch size
        self.rnn = nn.RNN(embedding_dim=64, batch_first=True, hidden_size=10)
        # Fully connected layer take the in_features and returns a word in the completes the
        # sequence with a word in the vocabulary
        self.fc = nn.Linear(10, vocab_size)


    def forward(self, x):

        pass








