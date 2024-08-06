import pytorch_lightning as pl
import torch
from data.dataset import SingleTrackDataset
from torch.utils.data.dataloader import DataLoader


class SingleTrackOverfitDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.__dict__.update(**args)
        self.common_dataset_kwargs = dict(
            dataset=self.dataset,
            song=self.song,
            audio_len=self.audio_len,
            batch_size=self.batch_size,
        )
        self.common_loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self):
        train_dataset = SingleTrackDataset(
            mode="train",
            train_dataset_len=self.steps_per_epoch,
            **self.common_dataset_kwargs
        )
        return DataLoader(train_dataset, **self.common_loader_kwargs)

    def val_dataloader(self):
        valid_dataset = SingleTrackDataset(mode="valid", **self.common_dataset_kwargs)
        return DataLoader(valid_dataset, **self.common_loader_kwargs)

    def test_dataloader(self):
        test_dataset = SingleTrackDataset(mode="test", **self.common_dataset_kwargs)
        return DataLoader(test_dataset, **self.common_loader_kwargs)

    def collate_fn(self, data_list):
        if len(data_list) == 1:
            return data_list[0]
        else:
            batch = {}
            batch["source"] = torch.stack([d["source"] for d in data_list], 0)
            batch["mix"] = torch.stack([d["mix"] for d in data_list], 0)
            return batch
