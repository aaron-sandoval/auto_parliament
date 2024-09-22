from typing import Literal, Callable, Any
from dataclasses import dataclass, field
import os
import abc
import itertools
from pathlib import Path
import random

import inspect_ai
from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, hf_dataset
from datasets import load_dataset
import inspect_ai.dataset

import prompts

ETHICS_CATEGORIES = ["commonsense", "deontology", "justice", "utilitarianism", "virtue"]
N_SAMPLES: int = 10

# def record_to_sample_base(record: dict):
#     return Sample(
#         input=record["input"],
#         target=str(record["label"]),
#         choices=["0","1"],
#     )

@dataclass
class InspectHFDataset:
    name: str
    path: str
    n_samples: int | None = N_SAMPLES
    dataset: inspect_ai.dataset.Dataset | None = None
    system_prompt: str = prompts.SYSTEM_HHH
    choices: tuple[str, ...] = field(default_factory=tuple)
    choices_permutations: list[list[str]] | None = None

    def __post_init__(self) -> None:
        if self.choices_permutations is None:
            self.choices_permutations = list(itertools.permutations(self.choices))

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

    @staticmethod
    @abc.abstractmethod
    def record_to_sample(record: dict[str, Any]) -> Sample:
        pass

@dataclass
class InspectBasicLabelDataset(InspectHFDataset):
    path: str = "hendrycks/ethics"
    target_map: dict = field(default_factory=dict)

    @staticmethod
    def record_to_sample(record: dict[str, Any]) -> Sample:
        return Sample(
            input=record["input"],
            target=InspectBasicLabelDataset.target_map[record["label"]],
            choices=random.sample(InspectBasicLabelDataset._choices_permutations, 1)[0],
        )

    def __post_init__(self) -> None:
        assert self.name in ETHICS_CATEGORIES
        self.choices = tuple(self.target_map.values())
        super().__post_init__()

ethics_datasets = [
    InspectBasicLabelDataset(
        name="commonsense",
        target_map={
            0: "Reasonable",
            1: "Unreasonable",
        },
        system_prompt=prompts.SYSTEM_COMMONSENSE,
    ),
]
