from typing import Literal, Callable
from dataclasses import dataclass
import os
from pathlib import Path
import random

import inspect_ai
from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, hf_dataset
from datasets import load_dataset
import inspect_ai.dataset

import prompts

ETHICS_CATEGORIES = ["commonsense", "deontology", "justice", "utilitarianism", "virtue"]
N_SAMPLES: int = 4

def record_to_sample_base(record: dict):
    return Sample(
        input=record["input"],
        target=str(record["label"]),
        choices=["0","1"],
    )

class CommonSenseConfig:
    target_map = {
        0: "Reasonable",
        1: "Unreasonable",
    }
    choices = ["Unreasonable","Reasonable"]

    @staticmethod
    def record_to_sample(record: dict):
        random.shuffle(CommonSenseConfig.choices)
        return Sample(
            input=record["input"],
            target=CommonSenseConfig.target_map[record["label"]],
            choices=CommonSenseConfig.choices,
        )



@dataclass
class InspectHFDataset:
    name: str
    path: str
    n_samples: int | None = 10
    dataset: inspect_ai.dataset.Dataset | None = None
    record_to_sample: Callable[[dict], Sample] = record_to_sample_base
    system_message: str = prompts.SYSTEM_HHH

    def __post_init__(self) -> None:
        self.dataset = hf_dataset(
            path=self.path,
            name=self.name,
            split="validation",
            sample_fields=self.record_to_sample,
            trust=True,
            limit=self.n_samples,
            shuffle=True,
            seed=42,
            # cache_dir=Path("../data/hf_cache"),
        )

@dataclass
class InspectEthicsDataset(InspectHFDataset):
    path: str = "hendrycks/ethics"

    def __post_init__(self) -> None:
        assert self.name in ETHICS_CATEGORIES
        super().__post_init__()

ethics_datasets = [
    InspectEthicsDataset(
        name="commonsense",
        system_message=prompts.SYSTEM_ETHICS,
        record_to_sample=CommonSenseConfig.record_to_sample,
        n_samples=N_SAMPLES,
    ),
]
