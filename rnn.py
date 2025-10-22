import time
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch


class CustomDataset(Dataset):

    """
    Data is split into windows where the time step = size of the window. If we had a sequence and one target variable we
    miss out on t – 1 pair of data that could be used for training.

    E.G. input = [4, 3, 1, 2], target = [5] Given t = 5 datapoints we waste t – 1 = 4 datapoints.
    If we set up like input = [4, 3, 1, 2], target = [3, 1, 2, 5] we have access to t-1 pairs of predictions.
    This allows the model to learn dependencies and not waste time steps.
    """

    def __init__(self, token_ids, block_size: int = 5, stride: int = 2) -> None:

        self.token_ids = token_ids
        self.block_size = block_size

        # Create window slices in innit, executed once and then fetch from get_item

        step = max(1, self.block_size - stride)
        self.starts = list(range(0, len(token_ids) - 1, step))

    def __len__(self) -> int:

        """
        When calling the dataloader it neets to know the length of the dataset to know how many times to call the
        dataloader.
        """

        return len(self.starts)

    def __getitem__(self, item: int) -> dict:

        """
        The dataloader iteratively accesses the windows. If it has a batch_size argument it will collect
        multiple windows into a single tensor.

        For padded tensors we also need to return a padded mask, that tells the model not to penalise predictions on
        padded data
        """

        s = self.starts[item]
        window = self.token_ids[s:s+self.block_size+1]
        x = window[:-1]
        y = window[1:]

        if (length_x := len(x)) < self.block_size:
            pad_len = self.block_size - length_x
            x = F.pad(x, (0, pad_len), mode='constant', value=-100)
            y = F.pad(y, (0, pad_len), mode='constant', value=-100)

        # Why do we need long tensors
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        # Return padded mask as well
        return {'input_ids': x, 'labels': y}


class RNNLanguageModel(nn.Module):

    def __init__(self, tokeniser) -> None:
        super().__init__()

        vocab_size = tokeniser.vocab_size
        self.embed = nn.Embedding(vocab_size, embedding_dim=128)
        # Expects the first dimension of the tensor to be batch size
        self.rnn = nn.RNN(128, batch_first=True, hidden_size=10)
        # Fully connected layer take the in_features and returns a word in the completes the
        # sequence with a word in the vocabulary
        self.fc = nn.Linear(10, vocab_size)

    def forward(self, x):

        out = self.embed(x)
        out, hidden = self.rnn(out)
        out = self.fc(out)
        return out


text = Path('corpus.txt').read_text(encoding='utf-8')   # Encoding in utf-8 is standard practice

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encoded = tokenizer.encode(text)

ds = CustomDataset(encoded)

train_size = int(len(ds) * 0.8)
test_size = len(ds) - train_size
train_ds, test_ds = random_split(ds, [train_size, test_size])

train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=2, shuffle=False)

# Set up the model, criterion and optimiser for the model
model = RNNLanguageModel(tokeniser=tokenizer)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimiser = Adam(model.parameters(), lr=0.001)


def train(model, dataloader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch['input_ids']        # [batch, seq_len]
            targets = batch['labels']          # [batch, seq_len]

            optimizer.zero_grad()
            outputs = model(inputs)            # [batch, seq_len, vocab_size]

            # Flatten for CrossEntropyLoss
            batch_size, seq_len, vocab_size = outputs.shape
            outputs = outputs.view(batch_size * seq_len, vocab_size)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss = {total_loss / len(dataloader):.4f}")



# Run training
train(model, train_dl, criterion, optimiser, epochs=40)



