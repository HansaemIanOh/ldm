import os
from typing import *
import json
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import datasets

class ImageFolderDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_dir: str, 
        json_dir: Optional[str] = None,
        patch_size: int = 224, 
        batch_size: int = 32, 
        num_workers: int = 4, 
        val_split: float = 0.1, 
        test_split: float = 0.1, 
        use_manual_split: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.json_dir = json_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.use_manual_split = use_manual_split

        self.transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage: Optional[str] = None):
        if self.use_manual_split:
            self.full_dataset = ImageFolder(self.data_dir, transform=self.transform)
            if self.json_dir is not None:
                self.full_dataset = self._load_and_apply_json_mapping(self.full_dataset, self.json_dir)
            self._split_dataset()
            self.num_classes = len(self.full_dataset.classes)
            
        else:
            self.train_dataset = ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.transform)
            self.valid_dataset = ImageFolder(os.path.join(self.data_dir, 'valid'), transform=self.transform)
            self.test_dataset = ImageFolder(os.path.join(self.data_dir, 'test'), transform=self.transform)
            if self.json_dir is not None:
                self.train_dataset = self._load_and_apply_json_mapping(self.train_dataset, self.json_dir)
                self.valid_dataset = self._load_and_apply_json_mapping(self.valid_dataset, self.json_dir)
                self.test_dataset = self._load_and_apply_json_mapping(self.test_dataset, self.json_dir)
            self.num_classes = len(self.train_dataset.classes)

    def _split_dataset(self):
        total_size = len(self.full_dataset)
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.val_split)
        train_size = total_size - test_size - val_size

        self.train_dataset, self.valid_dataset, self.test_dataset = random_split(
            self.full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def _load_and_apply_json_mapping(self, dataset, json_path):
        with open(json_path, 'r') as f:
            class_to_idx = json.load(f)['class_to_idx']
        dataset.class_to_idx = class_to_idx
        dataset.classes = list(class_to_idx.keys())
        dataset.samples = [(s[0], class_to_idx[dataset.classes[s[1]]]) for s in dataset.samples]
        dataset.targets = [class_to_idx[dataset.classes[t]] for t in dataset.targets]
        return dataset

class DataModule(pl.LightningDataModule):
    def __init__(
            self, 
            dataset_name: str,
            load_cache: str,
            save_cache: str,
            json_dir: Optional[str] = None,
            patch_size: int = 224, 
            batch_size: int = 32, 
            num_workers: int = 4, 
            val_split: float = 0.1, 
            test_split: float = 0.1, 
            use_manual_split: bool = False,
            **kwargs,
        ) -> None:
        super().__init__()
        self.json_dir = json_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.use_manual_split = use_manual_split
        self.dataset_name = dataset_name
        self.load_cache = load_cache
        self.save_cache = save_cache
        transforms = []
        if patch_size:
            transforms.append(Resize((patch_size, patch_size)))
        transforms.append(ToTensor())
        self.transform = Compose(transforms)

    def setup(
        self, 
        **kwargs
        ) -> None:
        if self.load_cache and os.path.exists(self.load_cache):
            dataset = datasets.load_from_disk(self.load_cache)
            self.split_dataset(dataset)
        else:
            dataset = datasets.load_dataset(self.dataset_name)

            def preprocess(examples):
                images = [self.transform(img.convert("RGB")) for img in examples["img"]]
                return {"imgs": images, "labels": examples["label"]}

            processed_datasets = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)
            processed_datasets.set_format("torch")

            self.save_processed_data(processed_datasets)
            self.split_dataset(processed_datasets)

    def split_dataset(self, dataset):
        self.train_dataset = dataset["train"]
        self.valid_dataset = dataset["validation"] if "validation" in dataset else dataset["test"]
        self.test_dataset = dataset["test"]

    @rank_zero_only
    def save_processed_data(self, tokenized_dataset):
        if self.save_cache:
            rank_zero_info(f"Saving processed dataset to {self.save_cache}")
            tokenized_dataset.save_to_disk(self.save_cache)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
