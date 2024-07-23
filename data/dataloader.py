import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def get_dataloader(file_path, seq_len, batch_size):
    dataset = VideoDataset(file_path, seq_len)
    # Convert batch of torch Tensors to numpy arrays for Flax
    _collate_fn = lambda batch: [x.numpy() for x in batch]
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
    )


class VideoDataset(Dataset):
    def __init__(self, file_path, seq_len):
        self.seq_len = seq_len
        self.data_shape = np.load(file_path, mmap_mode="r").shape
        self.data_file = file_path

    def __len__(self):
        return self.data_shape[0]

    def __getitem__(self, index):
        video = np.load(self.data_file, mmap_mode="r")[index]
        total_frames = video.shape[0]
        # Generate a random start index for the sequence
        seq_start = np.random.randint(0, total_frames - self.seq_len + 1)
        # Extract the sequence from the video
        sequence = video[seq_start : seq_start + self.seq_len]
        # Convert Numpy array to Torch tensor
        return torch.from_numpy(sequence).clone()
