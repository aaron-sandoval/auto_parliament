from typing import Literal
from datetime import datetime
from functools import cache
from pathlib import Path

from autogen import AssistantAgent, ChatResult
from inspect_ai import Task, eval
from inspect_ai.log import EvalLog
from inspect_ai.dataset import Dataset
from inspect_ai.scorer import choice, match
from inspect_ai.solver import (
    Plan, 
    system_message, 
    prompt_template,
    generate,
    chain_of_thought,
)

from parliament import agents
from eval_datasets import InspectEthicsDataset
import prompts


def ethics_task(dataset: Dataset, max_messages: int = 10):
    return Task(
        dataset,
        plan=Plan([
            system_message(prompts.SYSTEM_ETHICS),
            prompts.multiple_choice_format(template=prompts.MULTIPLE_CHOICE_FORMAT_TEMPLATE),
            prompt_template(prompts.COT_TEMPLATE),
            generate(),
            # prompt_template(prompts.MAKE_CHOICE_PROMPT),
            # generate()
        ]),
        scorer=match(location="end", ignore_case=False),
        max_messages=max_messages,
    )

def run_eval(dataset: InspectEthicsDataset, model: str) -> EvalLog:
    return eval(
        ethics_task(dataset.dataset),
        model=model,
        log_dir=str(Path("data/eval_logs")/f"{dataset.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"),

    )
