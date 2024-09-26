from typing import Sequence
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from autogen import ChatResult, AssistantAgent
from copy import deepcopy

from inspect_ai.log import EvalLog


SCORE_TO_FLOAT = {
    "I": 0.0,
    "C": 1.0,
}


def log_chat_history(chat_history: ChatResult, agents: Sequence[AssistantAgent], filename: str | None = None, file_suffix: str | None = None) -> dict:
    if filename is None:
        filename = "".join([datetime.now().strftime('%Y-%m-%d-%H%M%S'), f"_{file_suffix}" if file_suffix is not None else "", ".json"])
    filename = Path(".")/"data"/"chat_logs"/filename
    log = {
        "timestamp": datetime.now().strftime('%Y-%m-%d-%H:%M'),
        "participants": [],
        "messages": []
    }
    
    for agent in agents.values():
        LLM_config = deepcopy(agent.llm_config["config_list"][0])
        LLM_config.pop("api_key")
        log["participants"].append({
            "name": agent.name,
            "system_prompt": agent.system_message,
            "LLM_config": LLM_config
        })

    for message in chat_history.chat_history:
        participant = message["name"]
        # for msg in message:
        log["messages"].append({
            "sender": participant,
            "content": message['content'],
            # "role": message['role']
        })
    
    with open(filename, "w+", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    return log

def get_dataset_name(log: EvalLog) -> str:
    return log.eval.task.split("_")[0]

def get_model_name(log: EvalLog) -> str:
    return log.eval.task.split("_")[1]

def get_num_samples(log: EvalLog) -> int:
    return log.eval.dataset.samples

def get_questions(log: EvalLog) -> list[str]:
    return [sample.input for sample in log.eval.dataset.samples]

def get_targets(log: EvalLog) -> list[str]:
    return [sample.target for sample in log.eval.dataset.samples]

def get_latest_filenames(log_dir: Path, only_latest_run: bool = False) -> list[Path]:
    """Gets the latest log files in the given directory for each dataset and model.

    Args:
        log_dir: The directory to search for log files.
        only_latest_run: Whether to only return log files with the single latest runtime among all, or the latest for each dataset and model.
    """
    if only_latest_run:
        log_files = log_dir.glob("**/*.json")
        # Count the number of leaf directories
        num_leaf_dirs = sum(1 for _ in log_dir.rglob('*') if _.is_dir() and not any(_.iterdir()))
        
        log_files = sorted(log_files, key=lambda x: x.name)[-num_leaf_dirs:]
        return log_files
    else:
        log_files: list[Path] = []
        for subdir in log_dir.iterdir():
            if subdir.is_dir():
                latest_file = sorted(subdir.glob('*.json'))[-1]
                # if latest_file:
                log_files.append(latest_file)
        return sorted(log_files, key=lambda x: x.name)
