from typing import Sequence, Any
import json
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from autogen import ChatResult, AssistantAgent
from copy import deepcopy

from inspect_ai.log import EvalLog

EVAL_LOG_DIR = Path(".")/"data"/"eval_logs"
ANALYSIS_LOG_DIR = Path(".")/"data"/"eval_dfs"

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

def get_dataset_name(log_path: Path) -> str:
    return log_path.parent.parent.name

def get_model_name(log_path: Path) -> str:
    return log_path.parent.name

def get_num_samples(log: dict[str, Any]) -> int:
    return log.eval.dataset.samples

def get_questions(log: EvalLog) -> list[str]:
    return [sample.input for sample in log.eval.dataset.samples]

def get_targets(log: EvalLog) -> list[str]:
    return [sample.target for sample in log.eval.dataset.samples]

def get_latest_filenames(log_dir: Path = EVAL_LOG_DIR, only_latest_run: bool = False) -> list[Path]:
    """Gets the latest log files in the given directory for each dataset and model.

    Args:
        log_dir: The directory to search for log files.
        only_latest_run: Whether to only return log files with the single latest runtime among all, or the latest for each dataset and model.
    """
    leaf_dirs = [Path(dirpath) for dirpath, dirnames, _ in os.walk(log_dir) if not dirnames]
    if not only_latest_run:
        return [sorted(leafdir.glob('*.json'))[-1] for leafdir in leaf_dirs]
    else:
        raise NotImplementedError("Only latest run not implemented")

def save_eval_dfs(dfs: dict[str, pd.DataFrame]):
    for dataset_name, df in dfs.items():
        df.to_pickle(ANALYSIS_LOG_DIR/f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{dataset_name}.pkl")

