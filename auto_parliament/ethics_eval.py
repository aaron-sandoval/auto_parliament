from typing import Literal
from functools import cache
from datasets import load_dataset, DatasetDict, Dataset
from autogen import AssistantAgent, ChatResult

from parliament import agents

categories = ["commonsense", "deontology", "justice", "utilitarianism", "virtue"]

@cache
def load_validation_set(category: Literal["commonsense", "deontology", "justice", "utilitarianism", "virtue"]) -> Dataset:
    return load_dataset("hendrycks/ethics", category, split="validation")

def dataset_iterator(dataset: Dataset, shuffle: bool | int = 42, n_items: int = 10) -> Dataset:
    if shuffle:
        dataset = dataset.shuffle(seed=int(shuffle) if not isinstance(shuffle, bool) else None)
    return dataset[:n_items]

def eval_loop(agents: list[AssistantAgent], dataset: Dataset):
    for item in dataset:
        ...