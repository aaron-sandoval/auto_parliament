from typing import Literal
from datetime import datetime
from functools import cache
from pathlib import Path

from autogen import AssistantAgent, ChatResult
from inspect_ai import Task
from inspect_ai.scorer import choice, match
from inspect_ai.solver import (
    multiple_choice, 
    Plan, 
    system_message, 
    prompt_template,
    generate,
)

from parliament import agents
from datasets import ethics_datasets, InspectEthicsDataset, load_dataset, DatasetDict, Dataset
import prompts


def ethics_task(dataset: Dataset, max_messages: int = 10):
    return Task(
        dataset,
        plan=Plan([
            system_message(prompts.SYSTEM_HHH),
            multiple_choice(prompts.MULTIPLE_CHOICE_FORMAT_TEMPLATE),
            prompt_template(prompts.MAKE_CHOICE_PROMPT),
            generate()
        ]),
        scorer=match(),
        max_messages=max_messages,
    )

def run_eval(dataset: InspectEthicsDataset, model: str):
    eval(
        ethics_task(dataset),
        model=model,
        log_dir=str(Path("data")/"eval_logs"/f"{dataset.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"),

    )
