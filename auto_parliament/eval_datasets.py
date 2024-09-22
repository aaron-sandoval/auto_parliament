from typing import Literal, Callable, Any, override
from dataclasses import dataclass, field
import os
import abc
import itertools
from pathlib import Path
from functools import partial
import random

import inspect_ai
from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, hf_dataset
from datasets import load_dataset
import inspect_ai.dataset

import prompts

ETHICS_CATEGORIES = ["commonsense", "deontology", "justice", "utilitarianism", "virtue"]
N_SAMPLES: int = 4

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
    record_to_sample: Callable[[dict[str, Any]], Sample] | None = field(default=None)
    n_samples: int | None = N_SAMPLES
    dataset: inspect_ai.dataset.Dataset | None = None
    system_prompt: str = prompts.SYSTEM_HHH
    choices: tuple[str, ...] = field(default_factory=tuple)
    choices_permutations: list[list[str]] | None = None

    def __post_init__(self) -> None:
        if self.choices_permutations is None:
            self.choices_permutations = list(itertools.permutations(self.choices))
        if self.record_to_sample is None:
            self.record_to_sample = partial(self.record_to_sample_object, self)

        self.dataset = hf_dataset(
            path=self.path,
            name=self.name,
            split="validation",
            sample_fields=self.record_to_sample_object,
            trust=True,
            limit=self.n_samples,
            shuffle=True,
            seed=42,
            # cache_dir=Path("../data/hf_cache"),
        )

    @abc.abstractmethod
    def record_to_sample_object(self, record: dict[str, Any]) -> Sample:
        pass

@dataclass
class InspectBasicLabelDataset(InspectHFDataset):
    path: str = "hendrycks/ethics"
    target_map: dict = field(default_factory=dict)

    @override
    def record_to_sample_object(self, record: dict[str, Any]) -> Sample:
        return Sample(
            input=record["input"],
            target=self.target_map[record["label"]],
            choices=random.sample(self.choices_permutations, 1)[0],
        )

    def __post_init__(self) -> None:
        assert self.name in ETHICS_CATEGORIES
        self.choices = tuple(self.target_map.values())
        super().__post_init__()


@dataclass
class InspectDeontologyDataset(InspectBasicLabelDataset):
    @override
    def record_to_sample_object(self, record: dict[str, Any]) -> Sample:
        return Sample(
            input="\n".join([record["scenario"], record["excuse"]]),
            target=self.target_map[record["label"]],
            choices=random.sample(self.choices_permutations, 1)[0],
        )


ethics_datasets = [
    InspectBasicLabelDataset(
        name="commonsense",
        target_map={
            0: "Reasonable",
            1: "Unreasonable",
        },
        system_prompt=prompts.SYSTEM_COMMONSENSE,
    ),
    InspectDeontologyDataset(
        name="deontology",
        target_map={
            1: "Reasonable",
            0: "Unreasonable",
        },
        system_prompt=prompts.SYSTEM_DEONTOLOGY,
    ),
]
