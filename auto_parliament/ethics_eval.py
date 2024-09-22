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
    multiple_choice,
)

import single_llms
from eval_datasets import InspectHFDataset
import prompts


TEMPERATURE: float = 1.0


def ethics_task(dataset: InspectHFDataset, model: single_llms.InspectModel, max_messages: int = 10):
    return Task(
        dataset.dataset,
        plan=Plan([
            system_message(model.system_prompt),
            system_message(dataset.system_prompt),
            prompts.multiple_choice_format(),
            prompt_template(prompts.COT_TEMPLATE),
            model.generate_callable(),
            prompts.append_user_message(prompts.MAKE_CHOICE_PROMPT),
            model.generate_callable(),
        ]),
        scorer=match(location="end", ignore_case=False),
        max_messages=max_messages,
    )

def run_eval(dataset: InspectHFDataset, model: single_llms.InspectModel) -> tuple[EvalLog, Path]:
    log_dir = Path("../data/eval_logs")/f"{dataset.name}/{model.belief_name}"
    return eval(
        ethics_task(dataset, model),
        model=model.inspect_path,
        log_dir=str(log_dir),
        temperature=TEMPERATURE,
    ), log_dir
