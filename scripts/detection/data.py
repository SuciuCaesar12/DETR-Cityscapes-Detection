from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict
from cityscapes_helper import CityscapesDetectionAPI
import lightning.pytorch as pl


class CollateDetection:

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        pixel_values, labels = list(zip(*batch))
        encoding = self.processor.pad(pixel_values, return_tensors="pt")
        return {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': labels
        }


class CityscapesDetectionDataset(Dataset):

    def __init__(self, cityscapesAPI: CityscapesDetectionAPI, processor=None):
        self.cityscapesAPI = cityscapesAPI
        self.processor = processor

        self.image_ids = self.cityscapesAPI.getImageIds()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img, target = self.cityscapesAPI[image_id]

        # -------------------------------------------------------
        # DATA AUGMENTATION
        # -------------------------------------------------------

        if self.processor is not None:
            target = {'image_id': image_id, 'annotations': target}
            encoding = self.processor(images=img, annotations=target, return_tensors="pt")

            return encoding["pixel_values"].squeeze(), encoding["labels"][0]

        return img, target


class CityscapesDetectionDataModule(pl.LightningDataModule):

    def __init__(self,
                 root: Path,
                 annFiles: Dict[str, Path],
                 processor=None,
                 batch_size: int = 1,
                 num_workers: int = 0):
        super().__init__()

        self.root = root
        self.cityscapesAPIs = {split: CityscapesDetectionAPI(self.root, annFile) for split, annFile in annFiles.items()}
        self.datasets = {}
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = CollateDetection(processor)

        # metadata related to the dataset
        self.categories = self.cityscapesAPIs['train'].categories
        self.MAX_CLASS_ID = max([cat['id'] for cat in self.categories])

    def setup(self, stage):
        if stage == 'fit':
            for split in ['train', 'val']:
                if split in self.cityscapesAPIs:
                    self.datasets[split] = CityscapesDetectionDataset(
                        cityscapesAPI=self.cityscapesAPIs[split], processor=self.processor)

        if stage == 'test':
            if 'test' in self.cityscapesAPIs:
                self.datasets['test'] = CityscapesDetectionDataset(
                    cityscapesAPI=self.cityscapesAPIs['test'], processor=self.processor)

    def train_dataloader(self):
        if 'train' not in self.datasets:
            return None
        return DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True)

    def val_dataloader(self):
        if 'val' not in self.datasets:
            return None
        return DataLoader(
            self.datasets['val'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn)

    def test_dataloader(self):
        if 'test' not in self.datasets:
            return None
        return DataLoader(
            self.datasets['val'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn)
