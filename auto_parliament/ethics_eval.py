from typing import Literal, Any, TypeVar, Callable
from datetime import datetime
from functools import cache
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass
from functools import cached_property

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
    load_eval_dfs,
    get_latest_filenames,
    save_plots,
    save_eval_aug_dfs,
    load_eval_aug_dfs,
    SCORE_TO_FLOAT,
)
import plotting

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


def logs_to_dfs(single_llm_logs: list[Path] = get_latest_filenames()) -> dict[str, pd.DataFrame]:
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


def concat_parliament_evs(log_dfs: dict[str, pd.DataFrame], parliaments: list[ParliamentBasic]):
    """Concatenates the expected values of each parliament to each DataFrame.
    """
    for df in log_dfs.values():
        for parliament in parliaments:
            df[parliament.name] = parliament.get_expected_values(df)
    return log_dfs


def postprocess_logs(
        single_llm_logs: list[Path] | None = None, 
        parliaments: list[ParliamentBasic] | None = None,
        compile_json_to_dfs: bool = True,
        ) -> dict[str, pd.DataFrame]:
    """
    Args:
        single_llm_logs: List of Paths to single LLM logs.
        credences: List of dictionaries containing the credence values for each belief.
    """
    if compile_json_to_dfs:
        if single_llm_logs is None:
            single_llm_logs = get_latest_filenames()
        log_dfs: dict[str, pd.DataFrame] = logs_to_dfs(single_llm_logs)
        save_eval_dfs(log_dfs)
    else:
        log_dfs = load_eval_dfs()
    retval = log_dfs
    if parliaments is not None:
        log_dfs = concat_parliament_evs(log_dfs, parliaments)
        retval = save_eval_aug_dfs(log_dfs)
    return retval
        
T = TypeVar("T")

@dataclass
class EvalAnalysis:
    """
    Class for analyzing the results of an evaluation.

    Assumes that the log_dfs have already been concatenated with the parliaments.
    Assumes that the log_dfs are built from eval data from the same single LLMs.
    """
    log_dfs: dict[str, pd.DataFrame]
    parliaments: list[ParliamentBasic]
    NUM_QUESTION_COLS: int = 2  # Number of columns for question data before single LLMs and parliaments

    def __getattr__(self, key: str):
        if key in self.log_dfs.keys():
            return self.log_dfs[key]
        return super().__getattr__(key)

    @property
    def df1(self) -> pd.DataFrame:
        """Get the first DataFrame in the log_dfs, for when any will do"""
        return next(iter(self.log_dfs.values()))

    @property
    def num_single_llms(self) -> int:
        """Number of columns for single LLMs in the evaluation
        """
        return len(self.df1.columns) - self.NUM_QUESTION_COLS - len(self.parliaments)

    @property
    def single_agent_llm_names(self) -> list[str]:
        return list(self.df1.columns)[self.NUM_QUESTION_COLS:self.NUM_QUESTION_COLS+self.num_single_llms]
    
    @property
    def parliament_names(self) -> list[str]:
        return list(self.df1.columns)[-len(self.parliaments):]

    def map_over_datasets(self, func: Callable[[pd.DataFrame], T], datasets: list[str] | None = None) -> dict[str, T]:
        """Apply a function to each DataFrame in log_dfs.
        """
        if datasets is None:
            datasets = self.log_dfs.keys()
        else:
            assert all(dataset in self.log_dfs.keys() for dataset in datasets)
        return {dataset: func(df) for dataset, df in self.log_dfs.items() if dataset in datasets}

    def generate_plots(
            self, 
            show_plots: bool = False,
            ) -> list[plt.Figure]:
        """Calls all analysis and plotting functions and saves the results to the plots directory.
        """
        if show_plots:
            plt.ion()
        figs = {
            "model_performance_by_dataset": self.model_performance_by_dataset(show_plot=show_plots)[1],
            "score_cdf": self.question_performance_counts(show_plot=show_plots)[1],
            "covariance_among_beliefs": self.covariance_among_beliefs(show_plot=show_plots)[1],
        }
        save_plots(figs, formats=["png"])
        

    def get_cols(self, cols: list[str]) -> dict[str, pd.DataFrame]:
        return {key: df.loc[:, cols] for key, df in self.log_dfs.items()}

    def mean_over_questions(self, datasets: list[str] | None = None) -> dict[str, pd.Series]:
        if datasets is None:
            datasets = self.log_dfs.keys()
        return {dataset: df.iloc[:, self.NUM_QUESTION_COLS:].mean(axis=0) for dataset, df in self.log_dfs.items() if dataset in datasets}

    def mean_over_single_llms(self, datasets: list[str] | None = None) -> dict[str, pd.Series]:
        """Mean score by all single LLMs for each dataset in `datasets`.
        """
        if datasets is None:
            datasets = self.log_dfs.keys()
        return {dataset: df.loc[:,self.single_agent_llm_names].mean(axis=1) for dataset, df in self.log_dfs.items() if dataset in datasets}

    def question_performance_counts(
            self, 
            datasets: list[str] | None = None, 
            return_plot: bool = True,
            show_plot: bool = False,
            ) -> dict[str, pd.Series] | tuple[dict[str, pd.DataFrame], plt.Figure]:
        """Returns DataFrames containing histogram data of the mean score for each question.
        """
        if datasets is None:
            datasets = self.log_dfs.keys()
        assert all(dataset in self.log_dfs.keys() for dataset in datasets)
        mean_scores = self.mean_over_single_llms(datasets)
        out = {}
        for dataset in datasets:
            means = mean_scores[dataset]
            out[dataset] = means.value_counts().sort_index()
        
        if return_plot:
            fig = plotting.plot_cdfs(out, show=show_plot)
            return out, fig
        else:
            return out

    def model_performance_by_dataset(
            self, 
            datasets: list[str] | None = None, 
            return_plot: bool = True,
            show_plot: bool = False,
            ) -> pd.DataFrame | tuple[pd.DataFrame, plt.Figure]:
        """Returns a DataFrame with the mean score for each model (columns) for each dataset (rows).
        """
        if datasets is None:
            datasets = self.log_dfs.keys()
        assert all(dataset in self.log_dfs.keys() for dataset in datasets)
        means: dict[str, pd.Series] = self.mean_over_questions(datasets)
        out = pd.DataFrame(
            columns=self.single_agent_llm_names + self.parliament_names, 
            index=means.keys(),
            dtype=np.float16,
        )
        for dataset, series in means.items():
            out.loc[dataset] = series

        if return_plot:
            fig = plotting.plot_model_performance_by_dataset(out, show=show_plot)
            return out, fig
        else:
            return out

    def covariance_among_beliefs(
            self, 
            datasets: list[str] | None = None,
            return_plot: bool = True,
            show_plot: bool = False,
            ) -> pd.DataFrame | tuple[pd.DataFrame, plt.Figure]:
        """Returns a DataFrame with the dot product of each model's scores in question space.
        """
        if datasets is None:
            datasets = self.log_dfs.keys()
        def dot_product_matrix_normalized(df: pd.DataFrame) -> pd.DataFrame:
            # Replace all 0s with -1s in the DataFrame
            df = df.replace(0, -1)
            return pd.DataFrame(df[self.single_agent_llm_names].transpose() @ df[self.single_agent_llm_names]) / len(df)

        cov = self.map_over_datasets(dot_product_matrix_normalized, datasets)
        if return_plot:
            fig = plotting.plot_covariance_among_beliefs(cov, reduce_over_datasets=True, show=show_plot)
            return cov, fig
        else:
            return cov

if __name__ == "__main__":
    aug_dfs: dict[str, pd.DataFrame] = load_eval_aug_dfs()
    analysis = EvalAnalysis(log_dfs=aug_dfs, parliaments=single_llms.parliaments)
    analysis.generate_plots(show_plots=True)