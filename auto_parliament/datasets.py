from typing import Literal

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, hf_dataset


ETHICS_CATEGORIES = Literal["commonsense", "deontology", "justice", "utilitarianism", "virtue"]

@task
def ethics():
    def record_to_sample(record: dict):
        return Sample(
            input=record["input"],
            target=record["label"],
            choices=[0,1],
        )
    dataset = hf_dataset(
        path="hendrycks/ethics",
        split="validation",
        sample_fields=record_to_sample,
        trust=True
    )