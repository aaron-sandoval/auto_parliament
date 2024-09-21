from typing import Literal, Callable
from dataclasses import dataclass

import inspect_ai
from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, hf_dataset
import inspect_ai.dataset


def record_to_sample_base(record: dict):
    return Sample(
        input=record["input"],
        target=record["label"],
        choices=[0,1],
    )


@dataclass
class InspectHFDataset:
    dataset: inspect_ai.dataset.Dataset | None = None
    record_to_sample: Callable[[dict], Sample] = record_to_sample_base
    path: str
    name: str

    def __post_init__(self) -> None:
        self.dataset = hf_dataset(
            path=self.path,
            split="validation",
            sample_fields=self.record_to_sample,
            trust=True,
        )

ETHICS_CATEGORIES = Literal["commonsense", "deontology", "justice", "utilitarianism", "virtue"]

@dataclass
class InspectEthicsDataset(InspectHFDataset):
    name: ETHICS_CATEGORIES

    def __post_init__(self) -> None:
        self.path = f"hendrycks/ethics/{self.name}"
        self.dataset = hf_dataset(
            path=self.path,
            split="validation",
            sample_fields=self.record_to_sample,
            trust=True,
        )

ethics_datasets = [
    InspectEthicsDataset(
        name="commonsense",
    ),
]
