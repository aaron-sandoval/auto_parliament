# %%
import os
from pathlib import Path
import itertools

from ethics_eval import (
    run_eval, 
    postprocess_logs, 
    EvalAnalysis,
)
from eval_datasets import ethics_datasets
from single_llms import inspect_models, parliaments
from log_utils import EVAL_LOG_DIR, get_latest_filenames, load_eval_dfs

RUN_EVAL = False
RUN_POSTPROCESS = True
RUN_PLOTS = True
# %%
# Run evaluations
if RUN_EVAL:
    dataset_model_combos = itertools.product(ethics_datasets, inspect_models)
    logs = []
    for dataset, model in dataset_model_combos:
        log, log_path = run_eval(dataset, model)
        if log[0].status == "success":
            print(f"Success: {dataset.name=}\t{model.belief_name=}")
        else:
            print(f"Failed: {dataset.name=}\t{model.belief_name=}")
        logs.append((log, log_path))

    print("************\nAll done!\n************")
    log_dir = str(EVAL_LOG_DIR)
    # !inspect view start --log-dir $log_dir --port 7575

# %%
# Postprocess logs to augment with model parliaments
if RUN_POSTPROCESS:
    try:
        aug_dfs = postprocess_logs([log[1] for log in logs], parliaments)
    except NameError as e:
        print(f"No `logs` in memory, reading latest logs from disk.")
        aug_dfs = postprocess_logs(get_latest_filenames(EVAL_LOG_DIR), parliaments)

# %%
# Plot results
if RUN_PLOTS:
    try:
        analysis = EvalAnalysis(log_dfs=aug_dfs, parliaments=parliaments)
    except NameError as e:
        print(f"No `aug_dfs` in memory, reading latest logs from disk.")
        aug_dfs = load_eval_dfs()
        analysis = EvalAnalysis(log_dfs=aug_dfs, parliaments=parliaments)
    analysis.generate_plots(show_plots=True)

