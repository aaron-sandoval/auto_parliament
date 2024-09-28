from typing import Literal, Any
from datetime import datetime
from functools import cache
from pathlib import Path
import pandas as pd
import numpy as np
import json

from autogen import AssistantAgent, ChatResult
from inspect_ai import Task, eval
from inspect_ai.log import EvalLog
from inspect_ai.dataset import Dataset
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
from single_llms import ParliamentBasic
from log_utils import (
    get_dataset_name, 
    get_model_name, 
    get_num_samples,
    save_eval_dfs,
    get_latest_filenames,
    SCORE_TO_FLOAT,
)
TEMPERATURE: float = 1.0


def ethics_task(dataset: InspectHFDataset, model: single_llms.InspectModel, max_messages: int = 10):
    return Task(
        dataset.dataset,
        plan=Plan([
            system_message(model.system_prompt),
            system_message(dataset.system_prompt),
            dataset.mcq_format,
            prompt_template(prompts.COT_TEMPLATE),
            model.generate_callable(),
            prompts.append_user_message(prompts.MAKE_CHOICE_PROMPT),
            model.generate_callable(),
        ]),
        scorer=dataset.scorer,
        max_messages=max_messages,
        name=f"{dataset.name}_{model.abbv}",
    )

def run_eval(dataset: InspectHFDataset, model: single_llms.InspectModel) -> tuple[EvalLog, Path]:
    log_dir = Path("../data/eval_logs")/f"{dataset.name}/{model.belief_name}"
    return eval(
        ethics_task(dataset, model),
        model=model.inspect_path,
        log_dir=str(log_dir),
        temperature=TEMPERATURE,
    ), log_dir


def logs_to_dfs(single_llm_logs: list[Path]) -> dict[str, pd.DataFrame]:
    """Transforms a list of EvalLog paths into a list of DataFrames.

    Each DataFrame contains the following columns:
    - question: The question that was asked.
    - target: The target answer that was asked for.
    - <model_name>: Variable number of columns, each with the score given by a model.
    Returns a list of DataFrames, one for each dataset.
    """
    single_llm_names = list({get_model_name(log) for log in single_llm_logs})
    single_llm_names.sort()
    dataset_names = list({get_dataset_name(log) for log in single_llm_logs})
    dataset_names.sort()

    dfs = {}
    for dataset_name in dataset_names:
        dataset_log_names = [log for log in single_llm_logs if get_dataset_name(log) == dataset_name]
        dataset_logs: list[dict[str, Any]] = []
        for log_name in dataset_log_names:
            with open(log_name, "r", encoding="utf-8") as f:
                dataset_logs.append(json.load(f))

        # Load questiosn and targets
        num_samples = get_num_samples(dataset_logs[0])
        df = pd.DataFrame(
            columns=["question", "target"] + single_llm_names,
            dtype=np.float16,
            index=range(1, get_num_samples(dataset_logs[0]) + 1),
        )
        df["question"] = [sample["input"] for sample in dataset_logs[0]["samples"]]
        df["target"] = [sample["target"] for sample in dataset_logs[0]["samples"]]

        # Load scores
        for log, model_name in zip(dataset_logs, single_llm_names):
            if get_num_samples(log) != num_samples:
                raise ValueError("All logs must have the same number of samples")
            if not all(sample["input"] == df["question"].iloc[i] for i, sample in enumerate(log["samples"])):
                raise ValueError("Questions do not match")
            if not all(sample["target"] == df["target"].iloc[i] for i, sample in enumerate(log["samples"])):
                raise ValueError("Targets do not match")
            df.loc[:, model_name] = [SCORE_TO_FLOAT[sample["scores"]["match"]["value"]] for sample in log["samples"]]
        dfs[dataset_name] = df

    return dfs


def postprocess_logs(single_llm_logs: list[Path], parliaments: list[ParliamentBasic]):
    """
    Args:
        single_llm_logs: List of Paths to single LLM logs.
        credences: List of dictionaries containing the credence values for each belief.
    """
    log_dfs: dict[str, pd.DataFrame] = logs_to_dfs(single_llm_logs)
    save_eval_dfs(log_dfs)


if __name__ == "__main__":
    single_llm_logs = get_latest_filenames()
    logs_to_dfs(single_llm_logs)
