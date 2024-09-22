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
    # record_to_sample: Callable[[dict], Sample] = record_to_sample_base
    system_prompt: str = prompts.SYSTEM_HHH

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

    @staticmethod
    @abc.abstractmethod
    def record_to_sample(record: dict[str, Any]) -> Sample:
        pass

@dataclass
class InspectEthicsDataset(InspectHFDataset):
    path: str = "hendrycks/ethics"
    target_map: dict = field(
        default_factory=lambda: {
            0: "Reasonable",
            1: "Unreasonable",
        }
    )
    choices: tuple[str] = ("Unreasonable","Reasonable")
    choices_permutations: list[list[str]] | None = None

    @staticmethod
    def record_to_sample(record: dict[str, Any]) -> Sample:
        # random.shuffle(InspectEthicsDataset.choices)
        return Sample(
            input=record["input"],
            target=InspectEthicsDataset._target_map[record["label"]],
            choices=random.sample(InspectEthicsDataset._choices_permutations, 1)[0],
        )

    def __post_init__(self) -> None:
        assert self.name in ETHICS_CATEGORIES
        if self.choices_permutations is None:
            self.choices_permutations = list(itertools.permutations(self.choices))
        if not hasattr(InspectEthicsDataset, "_target_map"):
            InspectEthicsDataset._target_map = self.target_map
        if not hasattr(InspectEthicsDataset, "_choices_permutations"):
            InspectEthicsDataset._choices_permutations = self.choices_permutations
        super().__post_init__()

ethics_datasets = [
    InspectEthicsDataset(
        name="commonsense",
        system_prompt=prompts.SYSTEM_ETHICS,
    ),
]
