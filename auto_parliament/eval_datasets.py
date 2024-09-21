from typing import Literal, Callable
from dataclasses import dataclass
import os
from pathlib import Path

import inspect_ai
from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, hf_dataset
from datasets import load_dataset
import inspect_ai.dataset


ETHICS_CATEGORIES = ["commonsense", "deontology", "justice", "utilitarianism", "virtue"]


def download_ethics(subset: str = "commonsense", overwrite: bool = False):
    """
    Download the ethics dataset from Hugging Face and save it locally.
    
    Args:
        subset (str): The subset of the ethics dataset to download.
        overwrite (bool): If True, download and save even if the file already exists.
    """
    save_path = Path("data/datasets") / subset
    if save_path.exists() and not overwrite:
        print(f"Dataset '{subset}' already exists. Skipping download.")
        return
    
    dataset = load_dataset("hendrycks/ethics", subset, split="validation")
    
    # Create the directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the dataset
    dataset.save_to_disk(str(save_path))
    print(f"Dataset '{subset}' downloaded and saved to {save_path}")


def record_to_sample_base(record: dict):
    return Sample(
        input=record["input"],
        target=record["label"],
        choices=[0,1],
    )


@dataclass
class InspectHFDataset:
    name: str
    path: str = "hendrycks/ethics"
    dataset: inspect_ai.dataset.Dataset | None = None
    record_to_sample: Callable[[dict], Sample] = record_to_sample_base

    def __post_init__(self) -> None:
        # download_ethics(self.name, overwrite=True)
        self.path = "hendrycks/ethics"
        self.dataset = hf_dataset(
            path=self.path,
            name=self.name,
            split="validation",
            sample_fields=self.record_to_sample,
            trust=True,
            limit=4,
        )

@dataclass
class InspectEthicsDataset(InspectHFDataset):
    # name: ETHICS_CATEGORIES
    # path: str = ""

    def __post_init__(self) -> None:
        assert self.name in ETHICS_CATEGORIES
        self.path = f"hendrycks/ethics/{self.name}"
        super().__post_init__()

ethics_datasets = [
    InspectEthicsDataset(
        name="commonsense",
    ),
]
