import pathlib
import time
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from pathlib import Path
import torch.nn as nn
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

        # Why do we need long tensors
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        # Return padded mask as well
        return {'input_ids': x, 'labels': y}


text = pathlib.Path('corpus.txt').read_text('utf-8')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

encode = tokenizer.encode(text)
ds = CustomDataset(encode)

dl = DataLoader(ds, batch_size=1)
print(next(iter(dl)))


