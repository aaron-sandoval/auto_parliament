from typing import Sequence, Any, Literal
import json
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from autogen import ChatResult, AssistantAgent
from copy import deepcopy

from inspect_ai.log import EvalLog

ROOT_DIR = Path(__file__).parent.parent
EVAL_LOG_DIR = ROOT_DIR/"data"/"eval_logs"
EVAL_DF_DIR = ROOT_DIR/"data"/"eval_dfs"
EVAL_AUG_DIR = ROOT_DIR/"data"/"eval_aug_dfs"
PLOTS_DIR = ROOT_DIR/"data"/"plots"

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
    return log["eval"]["dataset"]["samples"]

def get_questions(log: dict[str, Any]) -> list[str]:
    return [sample["input"] for sample in log["eval"]["dataset"]["samples"]]


def get_targets(log: dict[str, Any]) -> list[str]:
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


def save_eval_dfs(dfs: dict[str, pd.DataFrame]) -> Path:
    dt: str = datetime.now().isoformat(timespec='minutes').replace(':', '')
    eval_df_dir = EVAL_DF_DIR / dt
    eval_df_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, df in dfs.items():
        df.to_pickle(eval_df_dir/f"{dataset_name}.pkl")
    return eval_df_dir


def load_eval_dfs(eval_df_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    if eval_df_dir is None:
        eval_df_dir = sorted(EVAL_DF_DIR.iterdir(), key=os.path.getmtime)[-1]
    dfs = {}
    for pkl_file in eval_df_dir.glob('*.pkl'):
        dataset_name = pkl_file.stem
        dfs[dataset_name] = pd.read_pickle(pkl_file)
    return dfs


def save_eval_aug_dfs(dfs: dict[str, pd.DataFrame]) -> Path:
    dt: str = datetime.now().isoformat(timespec='minutes').replace(':', '')
    eval_df_dir = EVAL_AUG_DIR / dt
    eval_df_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, df in dfs.items():
        df.to_pickle(eval_df_dir/f"{dataset_name}.pkl")
    return eval_df_dir


def load_eval_aug_dfs(eval_df_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    if eval_df_dir is None:
        eval_df_dir = sorted(EVAL_AUG_DIR.iterdir(), key=os.path.getmtime)[-1]
    dfs = {}
    for pkl_file in eval_df_dir.glob('*.pkl'):
        dataset_name = pkl_file.stem
        dfs[dataset_name] = pd.read_pickle(pkl_file)
    return dfs

PLOT_FORMAT: type = Literal["pkl", "png"]

def save_plots(plots: dict[str, plt.Figure], plot_dir: Path = PLOTS_DIR, formats: PLOT_FORMAT | list[PLOT_FORMAT] = ["pkl", "png"]) -> Path:
    plot_dir = plot_dir/datetime.now().isoformat(timespec='minutes').replace(':', '')
    plot_dir.mkdir(parents=True, exist_ok=True)
    for name, plot in plots.items():
        if "png" in formats:
            png_filename = plot_dir/f"{name}.png"
            plot.savefig(png_filename, bbox_inches='tight', pad_inches=0.1)
        if "pkl" in formats:
            fig_filename = plot_dir/f"{name}.pkl"
            with open(fig_filename, 'wb') as file:
                pickle.dump(plot, file)
    return plot_dir
