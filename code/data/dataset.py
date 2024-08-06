import numpy as np
from data.load import load_metadata, load_song
from torch.utils.data import Dataset


class SingleTrackDataset(Dataset):
    def __init__(
        self,
        mode="train",
        dataset="medley",
        song="TablaBreakbeatScience_RockSteady",
        train_dataset_len=500,
        audio_len=114000,
        sr=30000,
        load_kwargs={"target_tracks": ["mix"]},
        eval_warmup_sec=1,
        batch_size=1,
    ):
        super().__init__()
        self.mode = mode
        self.dataset = dataset
        self.song = song
        self.train_dataset_len = train_dataset_len
        self.audio_len = audio_len
        self.sr = sr
        self.load_kwargs = load_kwargs
        self.metadata = load_metadata(dataset, song)
        self.eval_warmup = eval_warmup_sec * sr
        self.batch_size = batch_size

    def __getitem__(self, idx):
        match self.mode:
            case "train":
                start = None
            case "valid" | "test":
                start = idx * (self.audio_len - self.eval_warmup)
        data = load_song(
            metadata=self.metadata,
            audio_len=self.audio_len,
            start=start,
            **self.load_kwargs,
        )
        return data

    def __len__(self):
        match self.mode:
            case "train":
                return self.train_dataset_len * self.batch_size
            case "valid" | "test" | "all":
                n = self.metadata["total_len"] / (self.audio_len - self.eval_warmup)
                return int(np.ceil(n))
