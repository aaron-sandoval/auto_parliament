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


def record_to_sample_base(record: dict):
    return Sample(
        input=record["input"],
        target=str(record["label"]),
        choices=["0","1"],
    )


@dataclass
class InspectHFDataset:
    name: str
    path: str = "hendrycks/ethics"
    dataset: inspect_ai.dataset.Dataset | None = None
    record_to_sample: Callable[[dict], Sample] = record_to_sample_base

    def __post_init__(self) -> None:
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
    def __post_init__(self) -> None:
        assert self.name in ETHICS_CATEGORIES
        self.path = f"hendrycks/ethics"
        super().__post_init__()

ethics_datasets = [
    InspectEthicsDataset(
        name="commonsense",
    ),
]
